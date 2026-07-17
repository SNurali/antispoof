"""Tests for app/frame_qc.py — Layer 0 per-frame QC gate."""
import cv2
import numpy as np

from app.face_landmarks import FrameFace
from app.frame_qc import MIN_FACE_EDGE_PX, assess_frame


def _make_face_image(size=300, brightness=140) -> np.ndarray:
    img = np.full((size, size, 3), brightness, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (brightness - 40, brightness - 60, brightness - 80), -1)
    # add some texture so the sharpness metric is non-trivial
    noise = (np.random.rand(size, size, 3) * 20).astype(np.uint8)
    return cv2.add(img, noise)


def _kps_for(size: int) -> np.ndarray:
    """Plausible 5-point landmarks (eyes/nose/mouth corners) centered in the image."""
    c = size / 2
    return np.array(
        [[c - 30, c - 20], [c + 30, c - 20], [c, c], [c - 20, c + 30], [c + 20, c + 30]],
        dtype=np.float32,
    )


def _frame_face(edge=200, yaw=0.0, pitch=0.0, kps_size=300) -> FrameFace:
    return FrameFace(
        bbox_xyxy=(50.0, 50.0, 50.0 + edge, 50.0 + edge),
        kps=_kps_for(kps_size),
        pose_pitch=pitch,
        pose_yaw=yaw,
        pose_roll=0.0,
        det_score=0.9,
        n_faces_detected=1,
    )


class TestAssessFrame:
    def test_no_face_rejected(self):
        img = _make_face_image()
        result = assess_frame(img, None)
        assert result.valid is False
        assert result.reason == "NO_FACE"

    def test_multiple_faces_rejected(self):
        img = _make_face_image()
        face = _frame_face()
        object.__setattr__(face, "n_faces_detected", 2)
        result = assess_frame(img, face)
        assert result.valid is False
        assert result.reason == "MULTIPLE_FACES"

    def test_too_small_face_rejected(self):
        img = _make_face_image()
        face = _frame_face(edge=MIN_FACE_EDGE_PX - 10)
        result = assess_frame(img, face)
        assert result.valid is False
        assert result.reason == "TOO_SMALL"

    def test_too_dark_rejected(self):
        img = _make_face_image(brightness=5)
        face = _frame_face()
        result = assess_frame(img, face)
        assert result.valid is False
        assert result.reason == "TOO_DARK"

    def test_too_bright_rejected(self):
        # Near-white frame with a few dark pixels for texture (needs to pass
        # the sharpness gate first, checked before brightness) but still
        # average well above MAX_BRIGHTNESS.
        img = np.full((300, 300, 3), 254, dtype=np.uint8)
        rng = np.random.default_rng(0)
        speckle = (rng.integers(0, 20, size=(300, 300)) == 0) * 200  # rare, high-contrast speckle
        img[..., 0] = np.clip(img[..., 0].astype(int) - speckle, 0, 255).astype(np.uint8)
        img[..., 1] = np.clip(img[..., 1].astype(int) - speckle, 0, 255).astype(np.uint8)
        img[..., 2] = np.clip(img[..., 2].astype(int) - speckle, 0, 255).astype(np.uint8)
        face = _frame_face()
        result = assess_frame(img, face)
        assert result.valid is False
        assert result.reason == "TOO_BRIGHT"

    def test_good_frame_valid(self):
        img = _make_face_image(brightness=140)
        face = _frame_face(edge=200)
        result = assess_frame(img, face)
        assert result.valid is True
        assert result.reason is None
        assert "sharpness" in result.metrics
        assert "brightness" in result.metrics
