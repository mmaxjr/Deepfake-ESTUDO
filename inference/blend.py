"""Compositing helpers: color matching + soft/hard mask blending."""

from __future__ import annotations

import cv2
import numpy as np


def color_transfer(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Reinhard-style LAB mean/std color transfer: recolor `source_bgr`
    (the swapped face) to match `target_bgr`'s (the original region's) color
    statistics, reducing visible seams from lighting/skin-tone mismatch."""
    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = src_lab.mean(axis=(0, 1)), src_lab.std(axis=(0, 1)) + 1e-6
    tgt_mean, tgt_std = tgt_lab.mean(axis=(0, 1)), tgt_lab.std(axis=(0, 1)) + 1e-6

    result = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def feather_mask(mask: np.ndarray, erode_px: int = 4, blur_px: int = 15) -> np.ndarray:
    """Erode a binary/soft mask slightly then Gaussian-blur it for a soft edge."""
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    if erode_px > 0:
        kernel = np.ones((erode_px, erode_px), np.uint8)
        mask_u8 = cv2.erode(mask_u8, kernel)
    blur_px = blur_px if blur_px % 2 == 1 else blur_px + 1
    mask_u8 = cv2.GaussianBlur(mask_u8, (blur_px, blur_px), 0)
    return mask_u8.astype(np.float32) / 255.0


def composite_feather(frame_bgr: np.ndarray, swapped_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-composite `swapped_bgr` onto `frame_bgr` using a feathered mask."""
    color_matched = color_transfer(swapped_bgr, frame_bgr)
    alpha = feather_mask(mask)[..., None]
    return (color_matched.astype(np.float32) * alpha + frame_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def composite_poisson(frame_bgr: np.ndarray, swapped_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha-composite via `cv2.seamlessClone` (Poisson blending). Slower,
    sometimes higher quality, occasionally introduces its own artifacts."""
    mask_u8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 10)
    if len(xs) == 0 or len(ys) == 0:
        return frame_bgr
    center = (int(xs.mean()), int(ys.mean()))
    try:
        return cv2.seamlessClone(swapped_bgr, frame_bgr, mask_u8, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone can fail on degenerate masks; fall back to feathering.
        return composite_feather(frame_bgr, swapped_bgr, mask)
