"""Join episodes from one run's snapshots into a single labelled video

For each chosen training step, takes the nearest snapshot from ppo/ppo_mario.py
--snapshot-every and replays evaluation episode 0 (seed 0), the same episode
scripts/evaluate.py records for that checkpoint. Each clip is labelled with its
training step and ends on a half-second hold. Frames are scaled up 3x so the
label stays sharp.

Usage:
    uv run python scripts/training_progress_video.py
    uv run python scripts/training_progress_video.py --steps 0 2 5 10 --output videos/short.mp4
"""

import argparse
import subprocess
import wave
from pathlib import Path

import cv2
import imageio
import imageio_ffmpeg
import numpy as np

# scripts/ is on the import path when this file is run directly
from evaluate import SNES_FPS, PPOAgent, run_episode

SCALE = 3
HOLD_FRAMES = 30  # half a second on the last frame of each clip
FONT = cv2.FONT_HERSHEY_SIMPLEX


class LabelledWriter:
    """Recorder for SkipFrames: labels and writes each frame as it arrives.

    Writing as we go keeps memory small; a 10M-step clip alone is 2,000+ frames.
    """

    def __init__(self, writer):
        self.writer = writer
        self.label = ""
        self.audio = []

    def append(self, item):
        frame, audio = item
        frame = cv2.resize(
            frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST
        )
        (width, height), _ = cv2.getTextSize(self.label, FONT, 0.8, 2)
        cv2.rectangle(frame, (0, 0), (width + 16, height + 20), (0, 0, 0), -1)
        cv2.putText(frame, self.label, (8, height + 10), FONT, 0.8, (255, 255, 255), 2)
        self.writer.append_data(frame)
        self.audio.append(audio)
        self.last_frame = frame

    def hold(self, frames):
        for _ in range(frames):
            self.writer.append_data(self.last_frame)
            self.audio.append(np.zeros_like(self.audio[-1]))


def main():
    parser = argparse.ArgumentParser(description="Join snapshots into one video")
    parser.add_argument("--state", default="YoshiIsland1", help="level savestate")
    parser.add_argument(
        "--snapshots",
        default="runs/ppo_mario/YoshiIsland1_seed0_snapshots",
        help="folder of step*.pt files from ppo_mario.py --snapshot-every",
    )
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 10],
        help="training steps to show, in millions (nearest snapshot is used)",
    )
    parser.add_argument("--output", default="videos/progress_yi1_seed0.mp4")
    args = parser.parse_args()

    snapshots = sorted(Path(args.snapshots).glob("step*.pt"))
    snapshot_steps = np.array(
        [int(path.stem.removeprefix("step")) for path in snapshots]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent, sound = output.with_suffix(".silent.mp4"), output.with_suffix(".wav")

    with imageio.get_writer(silent, fps=SNES_FPS) as writer:
        recorder = LabelledWriter(writer)
        for target in args.steps:
            index = np.abs(snapshot_steps - target * 1e6).argmin()
            step = snapshot_steps[index]
            recorder.label = f"{step:,} training steps"
            result = run_episode(PPOAgent(0, snapshots[index]), args.state, 0, recorder)
            recorder.hold(HOLD_FRAMES)
            print(
                f"{step:>10,} training steps: {result['end']:5s} at x "
                f"{result['max_x']:4d} after {result['steps']} steps"
            )

    with wave.open(str(sound), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)  # 16-bit samples
        w.setframerate(32040)  # the SNES audio rate
        w.writeframes(np.concatenate(recorder.audio).tobytes())
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", silent, "-i", sound]
        + ["-c:v", "copy", "-c:a", "aac", "-shortest", output],
        check=True,
    )
    silent.unlink()
    sound.unlink()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
