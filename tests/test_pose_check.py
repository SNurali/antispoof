"""Tests for app/pose_check.py — Layer 0d face-angle gate.

Focus: (1) correctness of the yaw/pitch comparison against the real
calibration numbers measured on s001 (see app/pose_check.py and
app/config.py docstrings — n=1 subject, real captured video frames via
insightface landmark_3d_68); (2) fail-safe on missing pose input — this
layer must NEVER crash the request path, same bar the other Layer 0 gates'
test suites hold. `check_face_pose` is a pure comparison (no model, no
image) — the model-calling side (app/main.py::_run_pose_gate) is exercised
in tests/test_pad_check.py instead.
"""

import pytest

from app.pose_check import check_face_pose

YAW_THRESH = 40.0
PITCH_THRESH = 45.0


class TestCheckFacePose:
    def test_frontal_bonafide_not_flagged(self):
        """s001 frontal_1: pose_yaw=-0.54, pose_pitch=3.52."""
        result = check_face_pose(-0.54, 3.52, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is False

    def test_bonafide_30deg_turn_not_flagged(self):
        """s001 left30: pose_yaw=32.79, pose_pitch=10.92 — the widest bona
        fide yaw actually measured; must clear YAW_THRESH=40.0 with margin."""
        result = check_face_pose(32.79, 10.92, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is False

    def test_bonafide_up_tilt_not_flagged(self):
        """s001 up: pose_pitch=36.88, pose_yaw=2.63 — the widest bona fide
        pitch actually measured; must clear PITCH_THRESH=45.0 with margin."""
        result = check_face_pose(2.63, 36.88, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is False

    def test_extreme_yaw_flagged(self):
        result = check_face_pose(55.0, 0.0, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is True

    def test_extreme_pitch_flagged(self):
        result = check_face_pose(0.0, 60.0, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is True

    def test_negative_extreme_yaw_flagged(self):
        """abs() must be applied — a large NEGATIVE yaw (turned the other
        way) must flag exactly like a large positive one."""
        result = check_face_pose(-55.0, 0.0, YAW_THRESH, PITCH_THRESH)
        assert result.ran is True
        assert result.is_off_angle is True

    def test_yaw_threshold_boundary_is_exclusive(self):
        """abs(yaw) == threshold must NOT flag (strict >, matching
        app/pose_check.py's `abs(pose_yaw) > yaw_threshold_deg`)."""
        result = check_face_pose(YAW_THRESH, 0.0, YAW_THRESH, PITCH_THRESH)
        assert result.is_off_angle is False

    def test_none_yaw_fails_safe(self):
        result = check_face_pose(None, 0.0, YAW_THRESH, PITCH_THRESH)
        assert result.ran is False
        assert result.error == "NO_POSE"

    def test_none_pitch_fails_safe(self):
        result = check_face_pose(0.0, None, YAW_THRESH, PITCH_THRESH)
        assert result.ran is False
        assert result.error == "NO_POSE"

    def test_both_none_fails_safe(self):
        result = check_face_pose(None, None, YAW_THRESH, PITCH_THRESH)
        assert result.ran is False
