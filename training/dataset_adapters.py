"""torch Dataset adapters over cached aligned face crops.

`IdentityFaceSet` and `SceneFaceSet` are intentionally two distinct classes
(not variants of one parametrized class) so that `Trainer.fit` can assert
`isinstance(scene_faces, SceneFaceSet)` / `isinstance(identity_faces,
IdentityFaceSet)` and catch an accidental swap as a hard type error.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.augment import color_jitter, random_warp


def _to_chw_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def _load_augmented_pair(path: Path, augment: bool) -> tuple[torch.Tensor, torch.Tensor]:
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"Could not read cached face crop: {path}")
    if augment:
        image = color_jitter(image)
        warped, target = random_warp(image)
    else:
        warped, target = image, image.copy()
    return _to_chw_tensor(warped), _to_chw_tensor(target)


class IdentityFaceSet(Dataset):
    """Aligned face crops for the consented identity branch."""

    def __init__(self, cache_dir_or_paths: str | Path | list[Path], augment: bool = True):
        self.paths = _resolve_paths(cache_dir_or_paths)
        if not self.paths:
            raise ValueError("IdentityFaceSet received no aligned face crops")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_augmented_pair(self.paths[idx], self.augment)


class SceneFaceSet(Dataset):
    """Aligned face crops sampled from scene/driving video(s)."""

    def __init__(self, cache_dir_or_paths: str | Path | list[Path], augment: bool = True):
        self.paths = _resolve_paths(cache_dir_or_paths)
        if not self.paths:
            raise ValueError("SceneFaceSet received no aligned face crops")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _load_augmented_pair(self.paths[idx], self.augment)


def _resolve_paths(cache_dir_or_paths: str | Path | list[Path]) -> list[Path]:
    if isinstance(cache_dir_or_paths, (list, tuple)):
        return list(cache_dir_or_paths)
    cache_dir = Path(cache_dir_or_paths)
    return sorted(cache_dir.rglob("*.png"))
