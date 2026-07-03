"""Train-time-only augmentation: random local warp + mild color jitter.

`random_warp` follows the classic "deepfakes"/faceswap trick: the network
sees a locally-deformed version of a face as input, but is trained to
reconstruct the *undistorted* aligned crop as target. This keeps the
encoder from overfitting to exact pixel alignment and generalizes better to
unseen poses/expressions at inference time.
"""

from __future__ import annotations

import cv2
import numpy as np


def random_warp(image: np.ndarray, grid_size: int = 5, warp_scale_frac: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Return (warped_input, target) -- both same size as `image`."""
    resolution = image.shape[0]
    assert image.shape[1] == resolution, "random_warp expects a square crop"

    range_ = np.linspace(0, resolution, grid_size, dtype=np.float32)
    mapx = np.broadcast_to(range_, (grid_size, grid_size)).astype(np.float32)
    mapy = mapx.T

    scale = max(1.0, resolution * warp_scale_frac)
    mapx = mapx + np.random.normal(size=(grid_size, grid_size), scale=scale).astype(np.float32)
    mapy = mapy + np.random.normal(size=(grid_size, grid_size), scale=scale).astype(np.float32)

    map_x = cv2.resize(mapx, (resolution, resolution)).astype(np.float32)
    map_y = cv2.resize(mapy, (resolution, resolution)).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, image.copy()


def color_jitter(image_bgr: np.ndarray, max_brightness: float = 0.15, max_saturation: float = 0.15) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + np.random.uniform(-max_saturation, max_saturation)
    hsv[..., 2] *= 1.0 + np.random.uniform(-max_brightness, max_brightness)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
