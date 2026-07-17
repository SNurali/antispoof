"""Tests for app/active_challenge.py — Layer 2 active challenge verification."""
import numpy as np

from app.active_challenge import verify_challenge
from app.config import Settings
from app.face_landmarks import FrameFace


def _eye_landmarks(mode: str) -> np.ndarray:
    """Synthetic (68, 3) landmark_68 array with only the eye-contour indices
    (36-41 right eye, 42-47 left eye — see face_landmarks.py::FrameFace for
    why these indices) populated; everything else stays zero (unused by
    eye_aspect_ratios). 'open' geometry gives EAR ~0.4 (well above any
    plausible threshold), 'closed' gives EAR ~0.04 (well below) — synthetic
    shapes chosen for a clean separation, NOT measured from a real blink
    (see app/config.py::LIVENESS_EAR_BLINK_MAX for why no real closed-eye
    calibration data exists yet)."""
    lmk = np.zeros((68, 3), dtype=np.float32)
    half_gap = 2.0 if mode == "open" else 0.2
    right_eye = {36: (0.0, 0.0), 37: (2.0, -half_gap), 38: (7.0, -half_gap),
                 39: (10.0, 0.0), 40: (7.0, half_gap), 41: (2.0, half_gap)}
    for i, (x, y) in right_eye.items():
        lmk[i] = (x, y, 0.0)
    for i, base in zip(range(42, 48), range(36, 42)):
        x, y, _ = lmk[base]
        lmk[i] = (x + 50.0, y, 0.0)  # left eye, offset so it doesn't overlap
    return lmk


def _face(yaw: float, pitch: float = 0.0, landmark_68: np.ndarray = None) -> FrameFace:
    return FrameFace(
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        kps=None,
        pose_pitch=pitch,
        pose_yaw=yaw,
        pose_roll=0.0,
        det_score=0.9,
        n_faces_detected=1,
        landmark_68=landmark_68,
    )


def _settings() -> Settings:
    return Settings(SERVICE_TOKEN="")


class TestVerifyChallenge:
    def test_unknown_step_rejected(self):
        frames = [(0, _face(0.0))]
        result = verify_challenge(["NOT_A_REAL_STEP"], frames, _settings())
        assert result.passed is False
        assert result.reason == "UNSUPPORTED_STEP"

    def test_no_frames_rejected(self):
        result = verify_challenge(["TURN_LEFT"], [], _settings())
        assert result.passed is False
        assert result.reason == "NO_FRONTAL_REFERENCE"

    def test_no_frontal_reference_rejected(self):
        # every frame far off-center — no baseline to compare against
        frames = [(0, _face(35.0)), (1, _face(40.0))]
        result = verify_challenge(["TURN_LEFT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "NO_FRONTAL_REFERENCE"

    def test_turn_left_detected_passes(self):
        # frontal reference + a frame turned past the threshold (positive yaw)
        frames = [(0, _face(0.0)), (1, _face(25.0))]
        result = verify_challenge(["TURN_LEFT"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_turn_right_detected_passes(self):
        frames = [(0, _face(0.0)), (1, _face(-25.0))]
        result = verify_challenge(["TURN_RIGHT"], frames, _settings())
        assert result.passed is True

    def test_missing_step_fails_as_challenge_failed(self):
        # frontal + a turn in the WRONG direction only
        frames = [(0, _face(0.0)), (1, _face(25.0))]
        result = verify_challenge(["TURN_RIGHT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "TURN_RIGHT" in result.detail["missing_steps"]

    def test_static_series_fails_multi_step_challenge(self):
        """No rotation anywhere in the series (e.g. a printed photo with
        slight natural pose jitter) — must not pass a multi-step challenge."""
        frames = [(0, _face(-2.2)), (1, _face(-1.2)), (2, _face(3.9)), (3, _face(0.4))]
        result = verify_challenge(["TURN_RIGHT", "TURN_LEFT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_both_steps_satisfied_by_different_frames(self):
        frames = [(0, _face(0.0)), (1, _face(25.0)), (2, _face(0.0)), (3, _face(-25.0))]
        result = verify_challenge(["TURN_LEFT", "TURN_RIGHT"], frames, _settings())
        assert result.passed is True


class TestVerifyChallengeBlink:
    """BLINK is implemented (2026-07-17) but deliberately not in the default
    pool — see app/config.py::LIVENESS_EAR_BLINK_MAX. These tests exercise
    the detection MECHANISM with synthetic geometry (open EAR~0.4 vs closed
    EAR~0.04, both far from the 0.20 cutoff in either direction) — they
    prove the wiring is correct, they do NOT stand in for a real-data
    threshold calibration."""

    def test_blink_is_a_supported_step(self):
        """BLINK must no longer be rejected as UNSUPPORTED_STEP — it moved
        from the old UNSUPPORTED_STEPS set into SUPPORTED_STEPS."""
        frames = [(0, _face(0.0, landmark_68=_eye_landmarks("open")))]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.reason != "UNSUPPORTED_STEP"

    def test_blink_detected_when_ear_dips_passes(self):
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(0.0, landmark_68=_eye_landmarks("closed"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("open"))),
        ]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_eyes_always_open_fails_blink(self):
        """A static photo (or a live face that never blinks in the captured
        window) must NOT pass a BLINK challenge."""
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("open"))),
        ]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "BLINK" in result.detail["missing_steps"]

    def test_missing_landmark_68_treated_as_no_evidence(self):
        """A frame with landmark_68=None (should not happen once the SCRFD
        pass succeeds, kept defensive) must not crash and must not count as
        blink evidence."""
        frames = [(0, _face(0.0, landmark_68=None))]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_blink_combined_with_turn_requires_both(self):
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(25.0, landmark_68=_eye_landmarks("open"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("closed"))),
        ]
        result = verify_challenge(["TURN_LEFT", "BLINK"], frames, _settings())
        assert result.passed is True
