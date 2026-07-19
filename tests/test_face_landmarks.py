"""Tests for app/face_landmarks.py::mouth_aspect_ratio — pure geometric unit
tests, independent of app/active_challenge.py's SMILE-step wiring (that's
covered separately in tests/test_active_challenge.py::TestVerifyChallengeSmile).

These cases probe the CORE geometric behavior of the width/height formula
(open mouth -> low ratio, closed/flat mouth -> high ratio) with known,
hand-computed expected values — not a stand-in for the real-photo index
verification already recorded in face_landmarks.py's docstring (that
verification used a real inference pass on a real photo, see
docs handoff / face_landmarks.py comments; these tests only prove the
ARITHMETIC is correct, they don't re-verify the index mapping)."""
import numpy as np
import pytest

from app.face_landmarks import mouth_aspect_ratio


def _mouth_geometry(width: float, half_gap: float) -> np.ndarray:
    """(68, 3) synthetic array with only the 6 indices mouth_aspect_ratio()
    reads populated: 48/54 corners (width apart on the x-axis), 50/52 upper
    lip and 58/56 lower lip (each `half_gap` above/below the midline) — same
    construction as tests/test_active_challenge.py::_mouth_landmarks, kept
    as a local helper here so this file has no cross-file test dependency."""
    lmk = np.zeros((68, 3), dtype=np.float32)
    lmk[48] = (0.0, 0.0, 0.0)
    lmk[54] = (width, 0.0, 0.0)
    lmk[50] = (width * 0.25, -half_gap, 0.0)
    lmk[58] = (width * 0.25, half_gap, 0.0)
    lmk[52] = (width * 0.75, -half_gap, 0.0)
    lmk[56] = (width * 0.75, half_gap, 0.0)
    return lmk


class TestMouthAspectRatio:
    def test_closed_flat_mouth_gives_high_ratio(self):
        """Closed/relaxed mouth: wide corner-to-corner span (20), thin
        vertical gap (2*0.5=1) -> width/height = 20/1 = 20.0. A closed mouth
        should read UNAMBIGUOUSLY high on this width/height formula."""
        lmk = _mouth_geometry(width=20.0, half_gap=0.5)
        ratio = mouth_aspect_ratio(lmk)
        assert ratio == pytest.approx(20.0, rel=1e-4)

    def test_open_mouth_gives_low_ratio(self):
        """Open mouth (e.g. yawn): same width (20), but wide vertical gap
        (2*15=30) -> width/height = 20/30 = 0.667. An open mouth should
        read UNAMBIGUOUSLY lower than the closed case above — the two
        cases must be clearly separated, same "clean synthetic separation"
        pattern already used for the EAR open/closed synthetic cases in
        tests/test_active_challenge.py::_eye_landmarks."""
        lmk = _mouth_geometry(width=20.0, half_gap=15.0)
        ratio = mouth_aspect_ratio(lmk)
        assert ratio == pytest.approx(20.0 / 30.0, rel=1e-4)

    def test_open_is_strictly_lower_than_closed(self):
        closed = mouth_aspect_ratio(_mouth_geometry(width=20.0, half_gap=0.5))
        opened = mouth_aspect_ratio(_mouth_geometry(width=20.0, half_gap=15.0))
        assert opened < closed

    def test_realistic_neutral_mouth_matches_real_photo_baseline_order(self):
        """Geometry approximating the REAL neutral-mouth measurement
        recorded in face_landmarks.py's mouth-index verification (real
        photo: width=142.4px, height=55.6px, MAR=2.56) — same proportions,
        different scale, confirming the formula reproduces that same
        real-measured order of magnitude, not just the two extreme
        synthetic cases above."""
        lmk = _mouth_geometry(width=142.4, half_gap=27.8)  # height ~= 55.6
        ratio = mouth_aspect_ratio(lmk)
        assert ratio == pytest.approx(2.56, rel=0.02)

    def test_degenerate_zero_height_returns_zero_not_a_crash(self):
        """half_gap=0 -> height=0 -> the <1e-6 guard must return 0.0, same
        defensive pattern as _single_eye_ratio's horizontal<1e-6 guard —
        must not raise a ZeroDivisionError."""
        lmk = _mouth_geometry(width=20.0, half_gap=0.0)
        assert mouth_aspect_ratio(lmk) == 0.0
