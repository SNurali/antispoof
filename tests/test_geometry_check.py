"""Tests for app/geometry_check.py — Layer 0a deterministic face-to-frame gate.

Focus: (1) correctness of the ratio math against the real calibration
numbers measured with FaceDetector on incident_urgut; (2) fail-safe on bad
input — this layer must NEVER crash the request path.
"""

import pytest

from app.geometry_check import check_face_geometry


class TestCheckFaceGeometry:
    def test_spoof_calibration_value_flagged_as_document(self):
        """Real numbers from incident_urgut/urgut_v2_passport/passport_style_spoof_01.jpg
        measured with the actual FaceDetector: frame 993x1275, bbox 748x799."""
        result = check_face_geometry(
            bbox=[0, 0, 748, 799], frame_shape=(1275, 993), threshold=0.35,
        )
        assert result.ran is True
        assert result.face_area_ratio == pytest.approx(0.4720, abs=1e-3)
        assert result.is_document is True

    def test_bonafide_max_calibration_value_not_flagged(self):
        """Real numbers from incident_urgut/original/photo_2026-07-06_11-36-03.jpg
        (the highest face-area-ratio bonafide in the calibration set):
        frame 960x1280, bbox 484x545."""
        result = check_face_geometry(
            bbox=[0, 0, 484, 545], frame_shape=(1280, 960), threshold=0.35,
        )
        assert result.ran is True
        assert result.face_area_ratio == pytest.approx(0.2147, abs=1e-3)
        assert result.is_document is False

    def test_all_12_bonafide_calibration_values_pass_below_threshold(self):
        """Every bonafide face_area_ratio measured on incident_urgut/original/
        (via the real FaceDetector) must stay below the default threshold."""
        # (frame_h, frame_w, bbox_w, bbox_h) per file, from live calibration run.
        bonafide_frames = [
            (1440, 1080, 454, 507),
            (1280, 960, 330, 361),
            (1280, 960, 224, 238),
            (1280, 960, 274, 290),
            (1280, 960, 394, 440),
            (1280, 960, 381, 418),
            (1280, 960, 353, 385),
            (1280, 960, 338, 358),
            (1280, 960, 400, 448),
            (1280, 960, 334, 361),
            (1280, 960, 390, 427),
            (1280, 960, 484, 545),
        ]
        for frame_h, frame_w, bbox_w, bbox_h in bonafide_frames:
            result = check_face_geometry(
                bbox=[0, 0, bbox_w, bbox_h], frame_shape=(frame_h, frame_w), threshold=0.35,
            )
            assert result.is_document is False, (
                f"bonafide frame {frame_w}x{frame_h} bbox {bbox_w}x{bbox_h} "
                f"wrongly flagged (ratio={result.face_area_ratio})"
            )

    def test_face_width_ratio_computed(self):
        result = check_face_geometry(bbox=[10, 10, 500, 500], frame_shape=(1000, 1000), threshold=0.35)
        assert result.face_width_ratio == pytest.approx(0.5)

    def test_frame_aspect_ratio_is_diagnostic_only(self):
        """frame_aspect_ratio is computed for observability but must NOT affect
        is_document — verified by holding a passport-like 7:9 aspect frame
        with a small face (should NOT be flagged)."""
        result = check_face_geometry(bbox=[0, 0, 50, 50], frame_shape=(900, 700), threshold=0.35)
        assert result.frame_aspect_ratio == pytest.approx(700 / 900, abs=1e-3)
        assert result.is_document is False

    def test_zero_frame_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, 100, 100], frame_shape=(0, 500), threshold=0.35)
        assert result.ran is False
        assert result.is_document is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_zero_bbox_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, 0, 100], frame_shape=(500, 500), threshold=0.35)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_negative_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, -10, 100], frame_shape=(500, 500), threshold=0.35)
        assert result.ran is False

    def test_malformed_bbox_fails_safe_not_raises(self):
        """Too few elements in bbox => unpacking error must be caught, not raised."""
        result = check_face_geometry(bbox=[0, 0], frame_shape=(500, 500), threshold=0.35)
        assert result.ran is False
        assert "UNEXPECTED" in result.error

    def test_threshold_is_inclusive_boundary(self):
        """face_area_ratio exactly == threshold should flag (>=, not >)."""
        # bbox area / frame area = 100*100 / 200*200 = 0.25
        result = check_face_geometry(bbox=[0, 0, 100, 100], frame_shape=(200, 200), threshold=0.25)
        assert result.is_document is True
