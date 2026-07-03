"""FaceSwapper: applies a trained FaceSwapAutoencoder to scene video frames."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from align.aligner import align_face, estimate_similarity_transform, unwarp_face
from align.detector import FaceDetector
from datasets.common import iter_video_frames, video_info
from models.autoencoder import FaceSwapAutoencoder

from .blend import composite_feather, composite_poisson
from .video_writer import mux_audio, write_silent_video

logger = logging.getLogger(__name__)


def _frame_to_tensor(frame_bgr: np.ndarray, device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


class FaceSwapper:
    def __init__(
        self,
        model: FaceSwapAutoencoder,
        detector: FaceDetector,
        resolution: int,
        device: "torch.device",
        blend: str = "feather",
    ):
        self.model = model.to(device).eval()
        self.detector = detector
        self.resolution = resolution
        self.device = device
        if blend not in ("feather", "poisson"):
            raise ValueError("blend must be 'feather' or 'poisson'")
        self.blend = blend

    def swap_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        face = self.detector.detect_largest(frame_bgr)
        if face is None:
            return frame_bgr

        transform = estimate_similarity_transform(face.kps, self.resolution)
        aligned = align_face(frame_bgr, transform, self.resolution)

        input_tensor = _frame_to_tensor(aligned, self.device)
        with torch.no_grad():
            output_tensor = self.model.swap(input_tensor)
        swapped_aligned = _tensor_to_bgr(output_tensor[0])

        h, w = frame_bgr.shape[:2]
        warped_back, mask = unwarp_face(swapped_aligned, transform, (h, w))

        composite_fn = composite_poisson if self.blend == "poisson" else composite_feather
        return composite_fn(frame_bgr, warped_back, mask)

    def _swap_frames(self, video_path: str | Path) -> Iterator[np.ndarray]:
        for frame in iter_video_frames(video_path, stride=1):
            yield self.swap_frame(frame)

    def process_video(
        self,
        scene_video_path: str | Path,
        output_path: str | Path,
        keep_audio: bool = True,
    ) -> Path:
        scene_video_path = Path(scene_video_path)
        output_path = Path(output_path)
        info = video_info(scene_video_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            silent_path = Path(tmp_dir) / "silent.mp4"
            write_silent_video(
                self._swap_frames(scene_video_path),
                silent_path,
                fps=info["fps"] or 25.0,
                frame_size=(info["width"], info["height"]),
            )
            if keep_audio:
                mux_audio(silent_path, scene_video_path, output_path)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy(silent_path, output_path)

        logger.info("Wrote swapped video to %s", output_path)
        return output_path
