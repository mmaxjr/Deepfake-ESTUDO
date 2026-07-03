"""Similarity-transform face alignment (5-point landmarks -> canonical crop).

Standard ArcFace-style reference points, defined at 112x112 and scaled to
whatever output resolution the pipeline is configured for.
"""

from __future__ import annotations

import cv2
import numpy as np

_REFERENCE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def reference_landmarks(output_resolution: int) -> np.ndarray:
    scale = output_resolution / 112.0
    return _REFERENCE_112 * scale


def estimate_similarity_transform(kps: np.ndarray, output_resolution: int) -> np.ndarray:
    """Estimate a 2x3 similarity transform mapping detected `kps` (5x2) onto
    the canonical reference landmarks for the given output resolution."""
    ref = reference_landmarks(output_resolution)
    matrix, _ = cv2.estimateAffinePartial2D(
        kps.astype(np.float32), ref, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("Could not estimate similarity transform from landmarks")
    return matrix


def align_face(image_bgr: np.ndarray, transform: np.ndarray, output_resolution: int) -> np.ndarray:
    """Warp `image_bgr` into a canonical aligned crop using `transform`."""
    return cv2.warpAffine(
        image_bgr, transform, (output_resolution, output_resolution), flags=cv2.INTER_LINEAR
    )


def unwarp_face(
    aligned_face: np.ndarray,
    transform: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an aligned-space face back into the original frame's coordinate space.

    Returns (warped_back_bgr, mask) where both have shape `target_shape` (H, W),
    warped_back_bgr is 3-channel and mask is single-channel float32 in [0, 1]
    covering the region the aligned crop occupies in the original frame.
    """
    h, w = target_shape
    warped = cv2.warpAffine(
        aligned_face,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    face_res = aligned_face.shape[0]
    ones = np.ones((face_res, face_res), dtype=np.float32)
    mask = cv2.warpAffine(
        ones,
        transform,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, mask
