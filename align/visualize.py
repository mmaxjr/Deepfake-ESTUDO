"""Visual sanity-check helpers: dump landmark/bbox overlays for manual review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detector import DetectedFace, FaceDetector


def draw_face_overlay(image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
    out = image_bgr.copy()
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in face.kps.astype(int):
        cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(
        out,
        f"{face.det_score:.2f}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


def dump_sanity_check(
    images: list[np.ndarray],
    detector: FaceDetector,
    out_dir: str | Path,
    prefix: str = "preview",
) -> list[Path]:
    """Run detection on each image and save an overlay preview for visual review."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, image in enumerate(images):
        face = detector.detect_largest(image)
        overlay = draw_face_overlay(image, face) if face is not None else image
        out_path = out_dir / f"{prefix}_{i:04d}.png"
        cv2.imwrite(str(out_path), overlay)
        saved.append(out_path)
    return saved
