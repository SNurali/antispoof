"""Tests for app/resolution_check.py — Layer 0e image resolution/weight gate.

Focus: (1) correctness of the min-side/megapixels/byte-size comparison
against the REAL numbers behind `MIN_IMAGE_MIN_SIDE_PX=700` /
`MIN_IMAGE_MEGAPIXELS=0.55` / `MIN_IMAGE_BYTES=15360` — the Telegram-preview
calibration dataset's observed max (623px / 0.498MP / 128.3KB, n=199,
`faces-dataset/`) and the client's expected real output (~960x1280,
~1.23MP, `egaz-mobile/core/core-faceid-capture`), both documented in
app/resolution_check.py's module docstring; (2) fail-safe on malformed
input — this layer must NEVER crash the request path, same bar the other
Layer 0 gates' test suites hold. `check_image_resolution` is a pure
arithmetic function (no model, no image decode) — the endpoint-wiring side
(app/main.py::_run_resolution_gate and its call sites) is exercised in
tests/test_pad_check.py and friends instead.
"""

import pytest

from app.resolution_check import check_image_resolution

MIN_SIDE = 700
MIN_MP = 0.55
MIN_BYTES = 15 * 1024


class TestCheckImageResolution:
    def test_client_shaped_image_not_flagged(self):
        """~960x1280 (3:4, egaz-mobile's expected real output, ~1.23MP),
        300KB — comfortably clears every threshold."""
        result = check_image_resolution(960, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is True
        assert result.is_low_resolution is False
        assert result.min_side == 960
        assert result.megapixels == pytest.approx(1.229, abs=0.001)

    def test_telegram_preview_max_dimensions_flagged(self):
        """623x800 — the WIDEST Telegram-preview file actually measured
        across the whole 199-file calibration dataset (both real/ and
        fake/) — must still be rejected by the min-side check."""
        result = check_image_resolution(623, 800, 90 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is True
        assert result.is_low_resolution is True
        assert "MIN_SIDE" in result.reason
        assert "MEGAPIXELS" in result.reason  # 0.498MP also below 0.55MP

    def test_telegram_preview_typical_dimensions_flagged(self):
        """450x800 — the single most common shape in the calibration
        dataset (162 of 199 files)."""
        result = check_image_resolution(450, 800, 60 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is True
        assert result.is_low_resolution is True

    def test_min_side_boundary_is_exclusive_below(self):
        """699px min side (1px under threshold) must flag."""
        result = check_image_resolution(699, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is True
        assert result.reason == "MIN_SIDE"  # megapixels (699*1280=0.894MP) still clears

    def test_min_side_boundary_at_threshold_not_flagged(self):
        """700px min side (== threshold) must NOT flag — `< min_side_px`,
        not `<=`, matching app/resolution_check.py's comparison."""
        result = check_image_resolution(700, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is False

    def test_megapixels_boundary_is_exclusive_below(self):
        """A wide-but-short frame with min_side well above threshold but
        total area just under 0.55MP must still flag on MEGAPIXELS alone."""
        # 700 x 780 = 0.546MP < 0.55MP, min_side=700 clears MIN_SIDE.
        result = check_image_resolution(700, 780, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is True
        assert result.reason == "MEGAPIXELS"

    def test_bytes_boundary_is_exclusive_below(self):
        """Dimensions clear both pixel checks, but the file is a
        near-blank/corrupted 10KB upload — BYTES alone must flag."""
        result = check_image_resolution(960, 1280, 10 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is True
        assert result.reason == "BYTES"

    def test_bytes_boundary_at_threshold_not_flagged(self):
        result = check_image_resolution(960, 1280, MIN_BYTES, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is False

    def test_all_three_checks_fire_together(self):
        """A genuinely tiny, near-empty upload — every sub-check fires,
        `reason` lists all of them."""
        result = check_image_resolution(100, 100, 2 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is True
        assert result.reason == "MIN_SIDE+MEGAPIXELS+BYTES"

    def test_zero_width_fails_safe(self):
        result = check_image_resolution(0, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_zero_height_fails_safe(self):
        result = check_image_resolution(960, 0, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is False
        assert result.error == "INVALID_DIMENSIONS"

    def test_negative_width_fails_safe(self):
        result = check_image_resolution(-10, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.ran is False

    def test_landscape_orientation_uses_min_side_correctly(self):
        """min(width, height) must be taken regardless of orientation —
        a rotated client frame (1280x960) is exactly as valid as 960x1280."""
        result = check_image_resolution(1280, 960, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.is_low_resolution is False
        assert result.min_side == 960

    def test_result_carries_measured_values_for_signals(self):
        """Fields consumed by app/main.py::_resolution_signals must be
        populated even on the not-flagged path (diagnostic value)."""
        result = check_image_resolution(960, 1280, 300 * 1024, MIN_SIDE, MIN_MP, MIN_BYTES)
        assert result.width == 960
        assert result.height == 1280
        assert result.byte_size == 300 * 1024
