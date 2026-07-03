"""Alignment smoke tests.

The similarity-transform math only needs numpy/opencv (already required),
so those tests always run. Anything touching the actual insightface model
is import-skipped when the ML stack isn't installed yet (see README: this
project needs a Python 3.10/3.11 venv with `requirements.txt` installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from align.aligner import align_face, estimate_similarity_transform, reference_landmarks, unwarp_face


def test_reference_landmarks_scale_with_resolution():
    ref_128 = reference_landmarks(128)
    ref_256 = reference_landmarks(256)
    assert ref_128.shape == (5, 2)
    np.testing.assert_allclose(ref_256, ref_128 * 2, rtol=1e-5)


def test_align_face_produces_expected_shape():
    resolution = 128
    kps = reference_landmarks(resolution).astype(np.float32)
    image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

    transform = estimate_similarity_transform(kps, resolution)
    aligned = align_face(image, transform, resolution)

    assert aligned.shape == (resolution, resolution, 3)


def test_unwarp_roundtrip_shapes_and_mask_range():
    resolution = 128
    kps = reference_landmarks(resolution).astype(np.float32)
    image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

    transform = estimate_similarity_transform(kps, resolution)
    aligned = align_face(image, transform, resolution)

    warped_back, mask = unwarp_face(aligned, transform, target_shape=(256, 256))

    assert warped_back.shape == (256, 256, 3)
    assert mask.shape == (256, 256)
    assert mask.min() >= 0.0 and mask.max() <= 1.0 + 1e-5
    assert mask.max() > 0.5  # some region should be fully covered


def test_face_detector_importable_or_skipped():
    pytest.importorskip("insightface")
    from align.detector import FaceDetector

    detector = FaceDetector()  # construction is lazy; no model download here
    assert detector._app is None
