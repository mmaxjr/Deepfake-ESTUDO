"""Builds cached, aligned face crops from either a SceneDataset or an IdentityDataset.

The two entrypoints (`build_from_scene` / `build_from_identity`) are kept
separate and each asserts the type of dataset it was given. This is part of
the defense-in-depth for the scene/identity separation: passing a
SceneDataset to `build_from_identity` (or vice versa) is a hard TypeError,
not a silently-accepted mistake.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from datasets.common import iter_video_frames, read_image
from datasets.identity_dataset import IdentityDataset
from datasets.scene_dataset import SceneDataset

from .aligner import align_face, estimate_similarity_transform
from .detector import FaceDetector

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _save_aligned(image, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    cv2.imwrite(str(out_path), image)
    return out_path


def build_from_identity(
    identity: IdentityDataset,
    detector: FaceDetector,
    output_resolution: int,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    if not isinstance(identity, IdentityDataset):
        raise TypeError(
            f"build_from_identity requires an IdentityDataset, got {type(identity).__name__}. "
            "Scene footage is never a valid identity source -- see README."
        )

    out_dir = Path(cache_dir) if cache_dir else identity.folder / "_aligned_cache"
    saved: list[Path] = []
    media = identity.iterate_media()
    logger.info("Aligning %d identity media file(s) for '%s'", len(media), identity.name)

    for media_path in media:
        if media_path.suffix.lower() in VIDEO_EXTENSIONS:
            frames = iter_video_frames(media_path, stride=5)
            prefix = media_path.stem
        else:
            frames = [read_image(media_path)]
            prefix = media_path.stem

        for i, frame in enumerate(frames):
            face = detector.detect_largest(frame)
            if face is None:
                continue
            transform = estimate_similarity_transform(face.kps, output_resolution)
            aligned = align_face(frame, transform, output_resolution)
            saved.append(_save_aligned(aligned, out_dir, f"{prefix}_{i:05d}"))

    logger.info("Saved %d aligned identity face crop(s) to %s", len(saved), out_dir)
    return saved


def build_from_scene(
    scene: SceneDataset,
    detector: FaceDetector,
    output_resolution: int,
    cache_dir: str | Path,
    frame_stride: int = 5,
    max_frames_per_video: int | None = 200,
) -> list[Path]:
    if not isinstance(scene, SceneDataset):
        raise TypeError(
            f"build_from_scene requires a SceneDataset, got {type(scene).__name__}."
        )

    cache_dir = Path(cache_dir)
    saved: list[Path] = []

    for record in scene.iterate_videos():
        video_out_dir = cache_dir / record.dataset_name / record.path.stem
        count = 0
        for i, frame in enumerate(iter_video_frames(record.path, stride=frame_stride)):
            if max_frames_per_video is not None and count >= max_frames_per_video:
                break
            face = detector.detect_largest(frame)
            if face is None:
                continue
            transform = estimate_similarity_transform(face.kps, output_resolution)
            aligned = align_face(frame, transform, output_resolution)
            saved.append(_save_aligned(aligned, video_out_dir, f"frame_{i:06d}"))
            count += 1

    logger.info("Saved %d aligned scene face crop(s) to %s", len(saved), cache_dir)
    return saved
