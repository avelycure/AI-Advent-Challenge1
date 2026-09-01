"""Общий конвейер записи роликов: снять сеанс, сжать паузы, собрать видео."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable, List, Sequence

import imageio_ffmpeg
from frame_renderer import FrameRenderer
from terminal_recorder import Frame, TerminalRecorder, compress


def record(argv: Sequence[str], cwd: str, scenario: Callable[[TerminalRecorder], None],
           cols: int, rows: int, fps: int) -> List[Frame]:
    recorder = TerminalRecorder(cols=cols, rows=rows, fps=fps)
    started = time.time()
    recorder.start(argv, cwd=cwd)
    try:
        scenario(recorder)
    finally:
        recorder.stop()
    print("  снято {} кадров за {:.0f} с".format(len(recorder.frames), time.time() - started))
    return recorder.frames


def encode(images, path: str, fps: int) -> None:
    width, height = images[0].size
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "{}x{}".format(width, height), "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "slow", "-crf", "26", "-tune", "stillimage",
        "-movflags", "+faststart", path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for image in images:
        process.stdin.write(image.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(process.stderr.read().decode("utf-8", "replace")[-2000:])


def build_video(frames: List[Frame], renderer: FrameRenderer, output: str, fps: int,
                max_segment: float, title=None, outro=None,
                title_seconds: float = 2.2, outro_seconds: float = 3.0) -> None:
    compressed = compress(frames, fps, max_segment)
    print("  после сжатия пауз: {} кадров ({:.1f} с)".format(
        len(compressed), len(compressed) / fps))

    print("→ рисую кадры…")
    cache = {}
    images = []
    for frame in compressed:
        key = (frame.snapshot, frame.caption)
        if key not in cache:
            cache[key] = renderer.render(frame.snapshot, frame.caption)
        images.append(cache[key])
    print("  уникальных кадров: {}".format(len(cache)))

    if title is not None:
        images = [title] * int(title_seconds * fps) + images
    if outro is not None:
        images = images + [outro] * int(outro_seconds * fps)

    output = os.path.expanduser(output)
    print("→ кодирую…")
    encode(images, output, fps)
    size = os.path.getsize(output)
    print("\nГотово: {}".format(output))
    print("  длительность {:.1f} с | размер {:.1f} МБ".format(
        len(images) / fps, size / 1024 / 1024))
