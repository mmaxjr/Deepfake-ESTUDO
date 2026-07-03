"""Silent-frame video writing + audio muxing via a bundled ffmpeg binary."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def write_silent_video(
    frames: Iterable[np.ndarray],
    output_path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
) -> Path:
    """`frame_size` is (width, height)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise IOError(f"Could not open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def mux_audio(silent_video_path: str | Path, source_video_path: str | Path, output_path: str | Path) -> Path:
    """Combine the (silent) swapped video with the original audio track from
    `source_video_path`. Falls back to the silent video as-is if muxing
    fails (e.g. the source has no audio stream)."""
    import imageio_ffmpeg

    silent_video_path, source_video_path, output_path = (
        Path(silent_video_path), Path(source_video_path), Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(silent_video_path),
        "-i", str(source_video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not output_path.exists():
        logger.warning(
            "Audio mux failed (falling back to silent video). ffmpeg stderr: %s",
            result.stderr.decode(errors="ignore")[-500:],
        )
        shutil.copy(silent_video_path, output_path)
    return output_path
