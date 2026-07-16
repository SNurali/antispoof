"""Tests for app/geometry_check.py — Layer 0a deterministic face-to-frame gate.

Focus: (1) correctness of the ratio math against the real calibration
numbers measured with FaceDetector on incident_urgut + the 2026-07-16
19:41 incident photo; (2) fail-safe on bad input — this layer must NEVER
crash the request path.

Thresholds re-calibrated 2026-07-16 evening (see app/geometry_check.py and
app/config.py docstrings for the full margin analysis): AREA=0.27 is the
ONLY ratio wired into production's `is_document` decision (2PAC review,
2026-07-16: face_width_ratio ~= 1.09*sqrt(face_area_ratio) on every
calibration sample, so a width gate adds no attack coverage area misses,
only FRR risk). `face_width_ratio` stays a diagnostic-only field, and
`check_face_geometry`'s optional `width_threshold` parameter is tested here
purely as a function CAPABILITY for a future, independent width
calibration — production (app/main.py) never passes it.
"""

import pytest

from app.geometry_check import check_face_geometry

AREA_THRESH = 0.27


class TestCheckFaceGeometry:
    """Production behavior: area-only decision (matches app/main.py wiring —
    `width_threshold` is never passed in production code)."""

    def test_passport_v2_spoof_flagged_as_document(self):
        """Real numbers from incident_urgut/urgut_v2_passport/passport_style_spoof_01.jpg
        measured with the actual FaceDetector: frame 993x1275, bbox 748x799."""
        result = check_face_geometry(bbox=[0, 0, 748, 799], frame_shape=(1275, 993), threshold=AREA_THRESH)
        assert result.ran is True
        assert result.face_area_ratio == pytest.approx(0.4721, abs=1e-3)
        assert result.face_width_ratio == pytest.approx(0.7533, abs=1e-3)  # diagnostic only
        assert result.is_document is True

    def test_2026_07_16_incident_caught_by_area_alone(self):
        """Real numbers from the 2026-07-16 19:41 incident photo (the one
        that slipped through the OLD 0.35 area-only threshold): measured
        with the actual FaceDetector, frame 993x1275, bbox 611x668.
        area_ratio=0.3224 < old 0.35 (missed) but >= new 0.27 (caught) — with
        NO width gate involved, exactly as production is wired."""
        result = check_face_geometry(bbox=[0, 0, 611, 668], frame_shape=(1275, 993), threshold=AREA_THRESH)
        assert result.ran is True
        assert result.face_area_ratio == pytest.approx(0.3224, abs=1e-3)
        assert result.face_width_ratio == pytest.approx(0.6153, abs=1e-3)  # diagnostic only
        assert result.is_document is True

    def test_bonafide_max_calibration_value_not_flagged(self):
        """Real numbers from incident_urgut/original/photo_2026-07-06_11-36-03.jpg
        (the highest face-area-ratio bonafide in the calibration set):
        frame 960x1280, bbox 484x545."""
        result = check_face_geometry(bbox=[0, 0, 484, 545], frame_shape=(1280, 960), threshold=AREA_THRESH)
        assert result.ran is True
        assert result.face_area_ratio == pytest.approx(0.2147, abs=1e-3)
        assert result.face_width_ratio == pytest.approx(0.5042, abs=1e-3)  # diagnostic only, NOT gated
        assert result.is_document is False

    def test_all_12_bonafide_calibration_values_pass_below_threshold(self):
        """Every bonafide face_area_ratio measured on incident_urgut/original/
        (via the real FaceDetector) must stay below the area threshold —
        area-only, exactly as production decides."""
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
                bbox=[0, 0, bbox_w, bbox_h], frame_shape=(frame_h, frame_w), threshold=AREA_THRESH,
            )
            assert result.is_document is False, (
                f"bonafide frame {frame_w}x{frame_h} bbox {bbox_w}x{bbox_h} "
                f"wrongly flagged (area={result.face_area_ratio})"
            )

    def test_face_width_ratio_always_computed_as_diagnostic_field(self):
        """face_width_ratio must always be present/correct in the result even
        though it never affects is_document in production (width_threshold
        defaults to None)."""
        result = check_face_geometry(bbox=[10, 10, 500, 500], frame_shape=(1000, 1000), threshold=AREA_THRESH)
        assert result.face_width_ratio == pytest.approx(0.5)
        assert result.face_area_ratio == pytest.approx(0.25)
        assert result.is_document is False  # 0.25 < AREA_THRESH(0.27) — area alone decides

    def test_wide_bbox_not_flagged_by_default_area_only_production_behavior(self):
        """A wide-but-short bbox (high width_ratio, low area_ratio) must NOT
        be flagged when width_threshold is not supplied — proves production
        (which never passes width_threshold) is genuinely area-only, not
        secretly gated by width."""
        result = check_face_geometry(bbox=[0, 0, 700, 100], frame_shape=(1000, 1000), threshold=AREA_THRESH)
        assert result.face_area_ratio == pytest.approx(0.07)
        assert result.face_width_ratio == pytest.approx(0.7)  # diagnostic only — high, but ignored
        assert result.is_document is False

    def test_frame_aspect_ratio_is_diagnostic_only(self):
        """frame_aspect_ratio is computed for observability but must NOT affect
        is_document — verified by holding a passport-like 7:9 aspect frame
        with a small face (should NOT be flagged)."""
        result = check_face_geometry(bbox=[0, 0, 50, 50], frame_shape=(900, 700), threshold=AREA_THRESH)
        assert result.frame_aspect_ratio == pytest.approx(700 / 900, abs=1e-3)
        assert result.is_document is False

    def test_zero_frame_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, 100, 100], frame_shape=(0, 500), threshold=AREA_THRESH)
        assert result.ran is False
        assert result.is_document is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_zero_bbox_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, 0, 100], frame_shape=(500, 500), threshold=AREA_THRESH)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_negative_dimension_fails_safe(self):
        result = check_face_geometry(bbox=[0, 0, -10, 100], frame_shape=(500, 500), threshold=AREA_THRESH)
        assert result.ran is False

    def test_malformed_bbox_fails_safe_not_raises(self):
        """Too few elements in bbox => unpacking error must be caught, not raised."""
        result = check_face_geometry(bbox=[0, 0], frame_shape=(500, 500), threshold=AREA_THRESH)
        assert result.ran is False
        assert "UNEXPECTED" in result.error

    def test_threshold_is_inclusive_boundary(self):
        """face_area_ratio exactly == threshold should flag (>=, not >)."""
        # bbox area / frame area = 100*100 / 200*200 = 0.25
        result = check_face_geometry(bbox=[0, 0, 100, 100], frame_shape=(200, 200), threshold=0.25)
        assert result.is_document is True


class TestWidthThresholdCapability:
    """`width_threshold` is a function CAPABILITY kept for a future,
    independently-collected width calibration — NOT wired into production
    (app/main.py never passes it). These tests cover only the parameter's
    own mechanics, not any production decision."""

    def test_width_threshold_none_by_default_matches_production_wiring(self):
        result = check_face_geometry(bbox=[0, 0, 700, 100], frame_shape=(1000, 1000), threshold=AREA_THRESH)
        assert result.is_document is False  # area-only: 0.07 < 0.27, width ignored

    def test_width_threshold_when_explicitly_supplied_can_still_gate(self):
        """Documents that the OR-gate mechanics still work if a future
        calibration explicitly opts back in — not exercised by production."""
        result = check_face_geometry(
            bbox=[0, 0, 700, 100], frame_shape=(1000, 1000), threshold=AREA_THRESH, width_threshold=0.55,
        )
        assert result.face_area_ratio == pytest.approx(0.07)  # well below AREA_THRESH
        assert result.is_document is True  # only true because width_threshold was explicitly passed here

    def test_width_threshold_is_inclusive_boundary(self):
        """face_width_ratio exactly == width_threshold should flag (>=, not >)
        when width_threshold is explicitly supplied."""
        result = check_face_geometry(
            bbox=[0, 0, 50, 10], frame_shape=(200, 100), threshold=0.99, width_threshold=0.5,
        )
        assert result.face_width_ratio == pytest.approx(0.5)
        assert result.is_document is True
