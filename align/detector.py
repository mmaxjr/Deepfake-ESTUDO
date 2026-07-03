"""Face detection wrapper around insightface (RetinaFace + 5-point landmarks)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    kps: np.ndarray         # 5x2 landmarks: left eye, right eye, nose, mouth-left, mouth-right
    det_score: float


class FaceDetector:
    """Lazily-initialized insightface detector.

    The first call downloads detector/landmark model weights to
    `~/.insightface/models` (a face-detector backbone, unrelated to any
    research dataset -- see README for why this doesn't conflict with the
    "no dataset auto-download" rule).
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640), ctx_id: int = -1):
        self.det_size = det_size
        self.ctx_id = ctx_id
        self._app = None

    def _ensure_loaded(self):
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        self._app = app
        logger.info("insightface detector loaded (det_size=%s, ctx_id=%s)", self.det_size, self.ctx_id)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        self._ensure_loaded()
        faces = self._app.get(image_bgr)
        if faces:
            return [
                DetectedFace(bbox=f.bbox, kps=f.kps, det_score=float(f.det_score))
                for f in faces
            ]

        # RetinaFace (and similar scale-based detectors) frequently miss
        # faces that fill almost the entire frame with no margin -- a very
        # common shape for tightly-cropped headshots/avatars used as
        # identity photos. Retry once with padding, then translate the
        # result back into the original image's coordinate space.
        h, w = image_bgr.shape[:2]
        pad = max(h, w) // 3
        padded = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padded_faces = self._app.get(padded)
        if not padded_faces:
            return []

        offset = np.array([pad, pad], dtype=np.float32)
        results = []
        for f in padded_faces:
            bbox = f.bbox.copy()
            bbox[0:2] -= offset
            bbox[2:4] -= offset
            kps = f.kps.copy() - offset
            results.append(DetectedFace(bbox=bbox, kps=kps, det_score=float(f.det_score)))
        return results

    def detect_largest(self, image_bgr: np.ndarray) -> DetectedFace | None:
        """Return the face with the largest bbox area (v1 heuristic for multi-face frames)."""
        faces = self.detect(image_bgr)
        if not faces:
            return None

        def area(face: DetectedFace) -> float:
            x1, y1, x2, y2 = face.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)

        return max(faces, key=area)


def make_detector(ctx_id_from_device: "torch.device | None" = None, det_size: tuple[int, int] = (640, 640)) -> FaceDetector:
    """Build a FaceDetector, mapping a torch device to insightface's ctx_id convention."""
    ctx_id = -1
    if ctx_id_from_device is not None and getattr(ctx_id_from_device, "type", "cpu") == "cuda":
        index = getattr(ctx_id_from_device, "index", None)
        ctx_id = index if index is not None else 0
    return FaceDetector(det_size=det_size, ctx_id=ctx_id)
