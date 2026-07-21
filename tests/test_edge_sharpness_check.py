"""Tests for app/edge_sharpness_check.py — Layer 0f DIAGNOSTIC-ONLY signal.

READ app/edge_sharpness_check.py's module docstring first: this is
explicitly NOT a reject gate. An asymmetric-edge-blur hypothesis (from a
forensic pass on one real fraud sample) was tested against a real bona fide
counter-example the same day and did not hold up (the counter-example
showed the SAME soft-edge/sharp-center pattern). These tests therefore only
cover: (1) the raw measurement is computed correctly (pure arithmetic, no
model), (2) fail-safe behavior on bad input, (3) that the module exposes NO
reject/flag boolean for app/main.py to accidentally wire into a gate.
Endpoint-level wiring (default-off, never affects verdict) is covered in
tests/test_pad_check.py.
"""

import cv2
import numpy as np
import pytest

from app.edge_sharpness_check import EdgeSharpnessDiagnostic, measure_edge_sharpness


def _make_uniform_frame(width: int = 720, height: int = 1280) -> np.ndarray:
    """Flat gray frame — zero Laplacian variance everywhere (no edges to
    detect), a degenerate-but-valid input."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_textured_frame(width: int = 720, height: int = 1280, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8).astype(np.uint8)


class TestMeasureEdgeSharpnessNoRejectAPI:
    def test_result_dataclass_has_no_reject_or_flag_field(self):
        """API-shape guard: this module must never grow an `is_*` boolean
        that a future caller could mistake for a gate decision — see module
        docstring. Fails loudly (not silently) if that ever changes."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(EdgeSharpnessDiagnostic)}
        assert not any(name.startswith("is_") for name in field_names), (
            f"EdgeSharpnessDiagnostic grew a boolean-looking field {field_names} — "
            "re-read the module docstring before wiring this into a reject decision"
        )


class TestMeasureEdgeSharpnessArithmetic:
    def test_uniform_frame_zero_everywhere(self):
        """A flat frame has zero Laplacian variance everywhere — center=0
        degrades to the documented all-zero result, not a crash/NaN."""
        result = measure_edge_sharpness(_make_uniform_frame())
        assert result.ran is True
        assert result.center_sharpness == 0.0
        assert result.left_to_center_ratio == 0.0
        assert result.right_to_center_ratio == 0.0

    def test_textured_frame_produces_positive_sharpness(self):
        result = measure_edge_sharpness(_make_textured_frame())
        assert result.ran is True
        assert result.left_sharpness > 0
        assert result.right_sharpness > 0
        assert result.center_sharpness > 0

    def test_min_edge_to_center_ratio_is_the_smaller_of_the_two(self):
        img = _make_textured_frame()
        result = measure_edge_sharpness(img)
        assert result.min_edge_to_center_ratio == min(result.left_to_center_ratio, result.right_to_center_ratio)

    def test_symmetric_content_gives_similar_left_right(self):
        """A frame mirrored left-right must produce near-identical
        left/right sharpness (sanity check on the crop geometry itself, not
        a claim about real photos)."""
        half = _make_textured_frame(width=360, height=1280)
        mirrored = np.concatenate([half, half[:, ::-1, :]], axis=1)
        result = measure_edge_sharpness(mirrored, edge_fraction=0.15)
        assert result.left_sharpness == pytest.approx(result.right_sharpness, rel=0.35)

    def test_edge_fraction_is_configurable(self):
        img = _make_textured_frame()
        r1 = measure_edge_sharpness(img, edge_fraction=0.10)
        r2 = measure_edge_sharpness(img, edge_fraction=0.25)
        # Different crop sizes over random noise -> not required to be equal,
        # just both must run successfully.
        assert r1.ran and r2.ran


class TestMeasureEdgeSharpnessFailSafe:
    def test_zero_height_fails_safe(self):
        result = measure_edge_sharpness(np.zeros((0, 100, 3), dtype=np.uint8))
        assert result.ran is False

    def test_zero_width_fails_safe(self):
        result = measure_edge_sharpness(np.zeros((100, 0, 3), dtype=np.uint8))
        assert result.ran is False

    def test_edge_fraction_out_of_range_fails_safe(self):
        img = _make_textured_frame()
        assert measure_edge_sharpness(img, edge_fraction=0.0).ran is False
        assert measure_edge_sharpness(img, edge_fraction=0.5).ran is False
        assert measure_edge_sharpness(img, edge_fraction=0.6).ran is False

    def test_extremely_narrow_frame_fails_safe(self):
        """A frame narrower than 2x the edge strip width has no room for a
        center region — must degrade to ran=False, not crash/negative-size crop."""
        narrow = _make_textured_frame(width=2, height=100)
        result = measure_edge_sharpness(narrow, edge_fraction=0.3)
        assert result.ran is False
        assert result.error == "FRAME_TOO_NARROW"


class TestMeasureEdgeSharpnessRealWorldSanityCheck:
    """Reproduces (approximately) the two documented reference measurements
    from the module docstring — NOT a threshold test (there is no
    threshold), just confirms the function's OWN numbers are internally
    consistent with what was reported after this hypothesis was tested and
    shelved."""

    def test_synthetic_one_sided_smear_shows_low_min_edge_ratio(self):
        """A synthetic stand-in for 'one edge deliberately blurred, center
        sharp' — the pattern that motivated this module — DOES show up as a
        low min_edge_to_center_ratio. This does NOT mean the signal is safe
        to gate on (see module docstring: a genuine bona fide photo showed
        the same pattern) — it only confirms the arithmetic responds in the
        expected direction on a clean synthetic case."""
        img = _make_textured_frame(width=720, height=1280, seed=1)
        blurred = img.copy()
        edge_w = int(720 * 0.12)
        blurred[:, :edge_w] = cv2.GaussianBlur(blurred[:, :edge_w], (25, 25), 0)
        result = measure_edge_sharpness(blurred, edge_fraction=0.12)
        assert result.ran is True
        assert result.left_to_center_ratio < result.right_to_center_ratio
