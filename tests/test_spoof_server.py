"""Tests for POST /spoof-server endpoint with verdict field."""

import base64
import os
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Force dev mode (no SERVICE_TOKEN) for tests
os.environ.setdefault("SERVICE_TOKEN", "")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("RATE_LIMIT_BURST", "1000")
os.environ.setdefault("RATE_LIMIT_SUSTAINED", "1000.0")

# Prevent real model loading at import time
os.environ["ANTISPOOF_SKIP_MODELS"] = "1"


def _make_test_image(width: int = 200, height: int = 200) -> bytes:
    """Create a simple test image as JPEG bytes."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 3, (200, 180, 160), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_base64_image(width: int = 200, height: int = 200) -> str:
    """Create a base64-encoded test image."""
    return base64.b64encode(_make_test_image(width, height)).decode()


@pytest.fixture(autouse=True)
def _no_startup():
    """Prevent startup event from loading real models and reset rate limiter."""
    import app.main as m
    from app.main import app
    original_handlers = app.router.on_startup.copy()
    app.router.on_startup.clear()
    # Reset rate limiter between tests
    m._rate_limiter._windows.clear()
    m._rate_limiter._burst = 1000
    m._rate_limiter._sustained = 1000.0
    yield
    app.router.on_startup = original_handlers


@pytest.fixture
def client():
    """Create test client with mocked models."""
    import app.main as m

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [50, 50, 100, 100]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = ("real", 0.95, True, {
        "signal_scores": {"recapture": 0.1},
        "spoof_probability": 0.05,
        "nn_label": "real",
        "nn_score": 0.95,
        "combined_label": "real",
        "combined_score": 0.95,
    })

    m.detector = mock_detector
    m.engine = mock_engine
    m._models_loaded = True

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /spoof-server — backward compatibility with is_spoof/elapsed_time
# ---------------------------------------------------------------------------

class TestSpoofServerBackwardCompat:
    def test_existing_fields_still_present(self, client):
        """Existing clients reading only is_spoof/elapsed_time must not break."""
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        # Old fields MUST be present
        assert "elapsed_time" in data
        assert "is_spoof" in data
        assert isinstance(data["elapsed_time"], float)
        assert data["is_spoof"] in (0, 1)

    def test_is_spoof_zero_when_real(self, client):
        """is_real=true → is_spoof=0 (unchanged behavior)."""
        import app.main as m
        m.engine.predict.return_value = ("real", 0.95, True, {})

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_spoof"] == 0

    def test_is_spoof_one_when_spoof(self, client):
        """is_real=false → is_spoof=1 (unchanged behavior)."""
        import app.main as m
        m.engine.predict.return_value = ("spoof", 0.85, True, {
            "spoof_probability": 0.85,
            "nn_label": "spoof",
            "nn_score": 0.85,
        })

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_spoof"] == 1


# ---------------------------------------------------------------------------
# POST /spoof-server — new verdict field (additive)
# ---------------------------------------------------------------------------

class TestSpoofServerVerdictMapping:
    def test_verdict_live_when_real(self, client):
        """is_real=true, label='real' → verdict='live'."""
        import app.main as m
        m.engine.predict.return_value = ("real", 0.95, True, {
            "nn_label": "real",
            "nn_score": 0.95,
        })

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "live"
        assert data["is_spoof"] == 0  # backward compat
        assert data["reason"] is None  # live has no reason

    def test_verdict_spoof_when_spoof_detected(self, client):
        """is_real=false, label='spoof' → verdict='spoof', reason='PASSIVE_PAD_SPOOF'."""
        import app.main as m
        m.engine.predict.return_value = ("spoof", 0.85, True, {
            "nn_label": "spoof",
            "nn_score": 0.85,
        })

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["is_spoof"] == 1  # backward compat
        assert data["reason"] == "PASSIVE_PAD_SPOOF"

    def test_verdict_low_quality_when_no_face(self, client):
        """label='no_face' → verdict='low_quality', reason='NO_FACE'."""
        import app.main as m
        m.detector.detect.return_value = None

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "low_quality"
        assert data["is_spoof"] == 1  # no face counts as spoof
        assert data["reason"] == "NO_FACE"

    def test_verdict_low_quality_when_real_but_low_score(self, client):
        """🔴 REGRESS BLOCKER 2: label='real' + score<threshold → verdict='low_quality', NOT spoof."""
        import app.main as m
        # Real face but poor score (bad lighting/occlusion) — should be low_quality, NOT spoof
        m.engine.predict.return_value = ("real", 0.30, True, {
            "nn_label": "real",
            "nn_score": 0.30,
        })
        # Default LIVENESS_THRESHOLD is typically 0.5
        assert m.settings.LIVENESS_THRESHOLD > 0.30

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        # OLD BEHAVIOR (BUGGY): would have been verdict="spoof" — WRONG!
        # NEW BEHAVIOR (FIXED): verdict="low_quality" — correct, tell client to retry
        assert data["verdict"] == "low_quality", \
            f"Real face with score={0.30}<threshold → must be low_quality, not spoof"
        assert data["reason"] == "LOW_QUALITY"
        assert data["is_spoof"] == 1  # Still is_spoof=1 for backward compat (no face/bad quality)

    def test_verdict_spoof_when_document_photo(self, client):
        """label='document_photo' (geometry gate) → verdict='spoof', reason='DOCUMENT_PHOTO'."""
        import app.main as m
        # Geometry gate will detect large face in frame
        # bbox [50,50,100,100] on 200x200 frame
        # area_ratio = (100*100) / (200*200) = 0.25 < 0.27 threshold
        # So geometry gate doesn't fire with default fixture.
        # Simulate it by overriding detect to return a larger bbox
        m.detector.detect.return_value = [10, 10, 150, 150]  # area_ratio ~0.56 > 0.27
        m.settings.GEOMETRY_CHECK_ENABLED = True

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["reason"] == "DOCUMENT_PHOTO"
        assert data["is_spoof"] == 1  # backward compat


# ---------------------------------------------------------------------------
# POST /spoof-server — invalid input handling
# ---------------------------------------------------------------------------

class TestSpoofServerErrorCases:
    def test_invalid_base64(self, client):
        """Invalid base64 → 400 error."""
        resp = client.post("/spoof-server", json={"photo": "not-base64!!!"})
        assert resp.status_code == 400

    def test_empty_request(self, client):
        """Missing photo field → 422 validation error."""
        resp = client.post("/spoof-server", json={})
        assert resp.status_code == 422

    def test_image_too_large(self, client):
        """Image exceeding 8MB → 400 error."""
        huge_b64 = "A" * (9 * 1024 * 1024)
        resp = client.post("/spoof-server", json={"photo": huge_b64})
        assert resp.status_code == 400

    def test_internal_error_returns_verdict(self, client):
        """🔴 REGRESS BLOCKER 1: models not loaded → graceful fallback with verdict."""
        import app.main as m

        # Simulate internal error: models not loaded
        m._models_loaded = False
        m.detector = None
        m.engine = None

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        # Graceful fallback: still returns 200 with safe verdict (even better than 500!)
        assert resp.status_code == 200
        data = resp.json()

        # CRITICAL: verdict must be present with INTERNAL_ERROR reason
        assert "verdict" in data, \
            "verdict field MUST be present — required by contract"
        assert data["verdict"] == "low_quality", \
            "Internal error → low_quality verdict (ask client to retry)"
        assert data["reason"] == "INTERNAL_ERROR"
        assert "is_spoof" in data
        assert "elapsed_time" in data

    def test_internal_error_fail_closed(self, client):
        """On internal error, fail-closed: is_spoof=1 (assume spoof/low quality)."""
        import app.main as m
        m._models_loaded = False
        m.detector = None

        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_spoof"] == 1, \
            "Fail-closed: on internal error, assume is_spoof=1 (safe default)"


# ---------------------------------------------------------------------------
# POST /spoof-server — response shape consistency with /pad/check
# ---------------------------------------------------------------------------

class TestSpoofServerResponseShape:
    def test_response_has_verdict_and_is_spoof(self, client):
        """Response should have BOTH verdict (new) and is_spoof (old) for compatibility."""
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        # New field
        assert "verdict" in data
        assert data["verdict"] in ("live", "spoof", "low_quality")
        # Old fields
        assert "is_spoof" in data
        assert "elapsed_time" in data

    def test_elapsed_time_is_reasonable(self, client):
        """elapsed_time should be a positive float (milliseconds range but stored as seconds)."""
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["elapsed_time"], float)
        assert 0.0 <= data["elapsed_time"] < 10.0  # Should be fast

    def test_verdict_consistent_with_is_spoof(self, client):
        """Verdict and is_spoof should be consistent.

        - is_spoof=0 ⟺ verdict='live' (ONLY live → is_spoof=0)
        - is_spoof=1 ⟹ verdict ∈ {spoof, low_quality} (all non-live → is_spoof=1)
        """
        import app.main as m

        # Test case 1: real face with high score → is_spoof=0, verdict=live
        m.engine.predict.return_value = ("real", 0.95, True, {})
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        data = resp.json()
        assert data["is_spoof"] == 0
        assert data["verdict"] == "live", \
            "is_spoof=0 ONLY when verdict=live"

        # Test case 2: actual spoof → is_spoof=1, verdict=spoof
        m.engine.predict.return_value = ("spoof", 0.85, True, {})
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        data = resp.json()
        assert data["is_spoof"] == 1
        assert data["verdict"] == "spoof", \
            "Spoof label → verdict=spoof"

        # Test case 3: low quality → is_spoof=1, verdict=low_quality
        m.detector.detect.return_value = None
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        data = resp.json()
        assert data["is_spoof"] == 1
        assert data["verdict"] == "low_quality", \
            "No face → verdict=low_quality"

    def test_reason_field_always_present(self, client):
        """reason field should always be present (even if null for live verdict)."""
        import app.main as m

        # Test case 1: live (reason=null)
        m.engine.predict.return_value = ("real", 0.95, True, {})
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        data = resp.json()
        assert "reason" in data
        assert data["reason"] is None

        # Test case 2: spoof (reason=PASSIVE_PAD_SPOOF)
        m.engine.predict.return_value = ("spoof", 0.85, True, {})
        resp = client.post("/spoof-server", json={"photo": _make_base64_image()})
        data = resp.json()
        assert "reason" in data
        assert data["reason"] == "PASSIVE_PAD_SPOOF"
