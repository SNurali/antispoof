"""Tests for app/aspect_ratio_check.py — Layer 0g camera-aspect-ratio gate.

Focus: (1) correctness of the min(w,h)/max(w,h) band comparison against the
REAL numbers behind `ASPECT_RATIO_MIN=0.70` / `ASPECT_RATIO_MAX=0.85` — the
real confirmed-fraud sample (real_fake_01.jpg, 720x1280 = 9:16 = 0.5625),
174/199 of faces-dataset/'s Telegram-preview files sharing that same 9:16
shape, and the client/bona-fide ground truth (egaz-mobile's 3:4 capture,
an owner-supplied real bona fide photo at 1920x2560 = 3:4), all documented
in app/aspect_ratio_check.py's module docstring; (2) fail-safe on malformed
input — this layer must NEVER crash the request path, same bar every other
Layer 0 gate's test suite holds. `check_aspect_ratio` is a pure arithmetic
function (no model, no image decode) — the endpoint-wiring side
(app/main.py::_run_aspect_ratio_gate and its call sites) is exercised in
tests/test_pad_check.py and friends instead.
"""

import pytest

from app.aspect_ratio_check import check_aspect_ratio

MIN_RATIO = 0.70
MAX_RATIO = 0.85


class TestCheckAspectRatio:
    def test_client_shaped_image_not_flagged(self):
        """960x1280 (3:4) — egaz-mobile's expected real capture shape."""
        result = check_aspect_ratio(960, 1280, MIN_RATIO, MAX_RATIO)
        assert result.ran is True
        assert result.is_non_camera_geometry is False
        assert result.ratio == pytest.approx(0.75)

    def test_owner_supplied_bona_fide_photo_not_flagged(self):
        """1920x2560 (3:4) — the real live "grandma in car" reference
        photo the owner supplied for comparison."""
        result = check_aspect_ratio(1920, 2560, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is False
        assert result.ratio == pytest.approx(0.75)

    def test_real_confirmed_fraud_sample_flagged(self):
        """720x1280 (9:16) — real_fake_01.jpg, the actual confirmed-fraud
        sample this gate exists for."""
        result = check_aspect_ratio(720, 1280, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True
        assert result.ratio == pytest.approx(0.5625)

    def test_telegram_preview_dominant_shape_flagged(self):
        """450x800 — the single most common shape across faces-dataset/
        (174 of 199 files), same 9:16 ratio as the confirmed fraud sample."""
        result = check_aspect_ratio(450, 800, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True

    def test_camera_shaped_telegram_minority_not_flagged(self):
        """600x800 (3:4-ish) — the ~25-file minority of faces-dataset/ that
        IS camera-shaped, must not be flagged."""
        result = check_aspect_ratio(600, 800, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is False

    def test_square_image_flagged(self):
        """1:1 — not a phone camera still-photo ratio either."""
        result = check_aspect_ratio(1000, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True
        assert result.ratio == pytest.approx(1.0)

    def test_16_9_landscape_flagged(self):
        """16:9 landscape reduces to the SAME ratio as a 9:16 portrait once
        measured as min/max — orientation must not matter."""
        result = check_aspect_ratio(1920, 1080, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True
        assert result.ratio == pytest.approx(0.5625)

    def test_orientation_agnostic_swap_gives_same_result(self):
        """Width/height swapped must give the identical ratio/verdict —
        min(w,h)/max(w,h) by construction."""
        portrait = check_aspect_ratio(960, 1280, MIN_RATIO, MAX_RATIO)
        landscape = check_aspect_ratio(1280, 960, MIN_RATIO, MAX_RATIO)
        assert portrait.ratio == landscape.ratio
        assert portrait.is_non_camera_geometry == landscape.is_non_camera_geometry

    def test_4_5_ratio_not_flagged(self):
        """4:5 = 0.80 — the second camera ratio the task explicitly named."""
        result = check_aspect_ratio(800, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is False
        assert result.ratio == pytest.approx(0.80)

    def test_min_ratio_boundary_at_threshold_not_flagged(self):
        """ratio == min_ratio exactly must NOT flag — inclusive band
        (`min_ratio <= ratio <= max_ratio`)."""
        # 700 x 1000 -> ratio exactly 0.70
        result = check_aspect_ratio(700, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is False

    def test_just_below_min_ratio_flagged(self):
        result = check_aspect_ratio(699, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True

    def test_max_ratio_boundary_at_threshold_not_flagged(self):
        # 850 x 1000 -> ratio exactly 0.85
        result = check_aspect_ratio(850, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is False

    def test_just_above_max_ratio_flagged(self):
        result = check_aspect_ratio(851, 1000, MIN_RATIO, MAX_RATIO)
        assert result.is_non_camera_geometry is True

    def test_zero_width_fails_safe(self):
        result = check_aspect_ratio(0, 1280, MIN_RATIO, MAX_RATIO)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_zero_height_fails_safe(self):
        result = check_aspect_ratio(960, 0, MIN_RATIO, MAX_RATIO)
        assert result.ran is False

    def test_negative_dimensions_fail_safe(self):
        result = check_aspect_ratio(-10, 1280, MIN_RATIO, MAX_RATIO)
        assert result.ran is False

    def test_result_carries_measured_values_for_signals(self):
        result = check_aspect_ratio(960, 1280, MIN_RATIO, MAX_RATIO)
        assert result.width == 960
        assert result.height == 1280
