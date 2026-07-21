"""Tests for app/blur_check.py — Layer 0c deterministic frame-sharpness gate.

Focus: (1) sharp vs. blurred synthetic images land on the correct side of
`MIN_FACE_SHARPNESS_224` (see app/config.py for the 93.0 sharp / 25.6-89.4
motion-blurred-k9 real-calibration numbers this threshold sits between);
(2) fail-safe on bad input — this layer must NEVER crash the request path,
same bar app/geometry_check.py's own test suite holds.
"""

import cv2
import numpy as np
import pytest

from app.blur_check import check_face_sharpness

THRESHOLD = 60.0


def _checkerboard(size: int = 200) -> np.ndarray:
    """High-frequency synthetic pattern — stands in for a sharp real face
    crop (rich micro-texture => high Laplacian variance)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cell = 8
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y:y + cell, x:x + cell] = 220
    return img


def _blurred(img: np.ndarray, k: int = 21) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


class TestCheckFaceSharpness:
    def test_sharp_synthetic_face_not_flagged(self):
        img = _checkerboard()
        result = check_face_sharpness(bbox=[0, 0, 200, 200], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is True
        assert result.sharpness > THRESHOLD
        assert result.is_blurry is False

    def test_heavily_blurred_synthetic_face_flagged(self):
        img = _blurred(_checkerboard(), k=31)
        result = check_face_sharpness(bbox=[0, 0, 200, 200], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is True
        assert result.sharpness < THRESHOLD
        assert result.is_blurry is True

    def test_blurring_monotonically_reduces_measured_sharpness(self):
        """Sanity check on the metric's own direction — sharper kernels
        (smaller k) must measure higher variance than blurrier ones (larger
        k), matching the real motion-blur-k9 numbers in the module docstring
        (25.6-89.4) sitting below the sharp-original numbers (93.0-437.1)."""
        base = _checkerboard()
        sharpness_by_k = []
        for k in (1, 5, 15, 31):
            img = base if k == 1 else _blurred(base, k)
            result = check_face_sharpness(bbox=[0, 0, 200, 200], image_bgr=img, threshold=THRESHOLD)
            sharpness_by_k.append(result.sharpness)
        assert sharpness_by_k == sorted(sharpness_by_k, reverse=True)

    def test_threshold_boundary_is_exclusive_not_flagged(self):
        """sharpness == threshold must NOT flag (strict <, matching
        app/blur_check.py's `is_blurry = sharpness < threshold`)."""
        img = _checkerboard()
        result = check_face_sharpness(bbox=[0, 0, 200, 200], image_bgr=img, threshold=THRESHOLD)
        # Re-run with threshold pinned exactly to the measured value.
        exact = check_face_sharpness(bbox=[0, 0, 200, 200], image_bgr=img, threshold=result.sharpness)
        assert exact.is_blurry is False

    def test_zero_bbox_dimension_fails_safe(self):
        img = _checkerboard()
        result = check_face_sharpness(bbox=[0, 0, 0, 100], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_negative_bbox_dimension_fails_safe(self):
        img = _checkerboard()
        result = check_face_sharpness(bbox=[0, 0, -10, 100], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is False

    def test_bbox_entirely_outside_frame_fails_safe(self):
        img = _checkerboard(size=50)
        result = check_face_sharpness(bbox=[500, 500, 50, 50], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is False
        assert result.error == "EMPTY_CROP"

    def test_malformed_bbox_fails_safe_not_raises(self):
        img = _checkerboard()
        result = check_face_sharpness(bbox=[0, 0], image_bgr=img, threshold=THRESHOLD)
        assert result.ran is False
        assert "UNEXPECTED" in result.error
