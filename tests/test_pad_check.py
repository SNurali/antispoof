"""Tests for POST /pad/check endpoint and related hardening."""

import base64
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Force dev mode (no SERVICE_TOKEN) for most tests
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


@pytest.fixture
def client_with_auth():
    """Create test client with SERVICE_TOKEN enabled."""
    import app.main as m
    old_token = m.settings.SERVICE_TOKEN
    m.settings.SERVICE_TOKEN = "TEST_TOKEN_PLACEHOLDER"
    m._save_frame_verdicts = {"spoof"}

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [50, 50, 100, 100]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = ("real", 0.95, True, {})
    m.detector = mock_detector
    m.engine = mock_engine
    m._models_loaded = True

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    m.settings.SERVICE_TOKEN = old_token


# ---------------------------------------------------------------------------
# POST /pad/check — happy path
# ---------------------------------------------------------------------------

class TestPadCheckHappyPath:
    def test_valid_image_returns_verdict(self, client):
        """Valid base64 image → contract-shaped response per FACEID_PHASE1_PAD_GATE §1."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-001",
            "transaction_type": "sale",
            "transaction_ref": "req1:balloon1",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] in ("live", "spoof", "low_quality")
        assert "reason" in data
        assert "score" in data
        assert "threshold" in data
        assert "face_detected" in data
        assert "model_version" in data
        assert "processing_ms" in data
        assert isinstance(data["save_frame"], bool)
        assert isinstance(data["signals"], dict)

    def test_verdict_live_when_confident(self, client):
        """High-confidence real face → verdict='live', reason=null."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-real",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "live"
        assert data["reason"] is None
        assert data["face_detected"] is True

    def test_verdict_spoof_when_spoof_detected(self, client):
        """Spoof detection → verdict='spoof', save_frame=True."""
        # Override engine mock for this test
        import app.main as m
        original = m.engine.predict
        m.engine.predict.return_value = ("spoof", 0.85, True, {
            "signal_scores": {"recapture": 0.7},
            "spoof_probability": 0.85,
            "nn_label": "spoof",
            "nn_score": 0.85,
            "combined_label": "spoof",
            "combined_score": 0.85,
        })
        m._save_frame_verdicts = {"spoof"}

        resp = client.post("/pad/check", json={
            "correlation_id": "test-spoof",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["reason"] == "PASSIVE_PAD_SPOOF"
        assert data["save_frame"] is True

        # Restore
        m.engine.predict = original


# ---------------------------------------------------------------------------
# POST /pad/check — auth
# ---------------------------------------------------------------------------

class TestPadCheckAuth:
    def test_401_without_token_when_required(self, client_with_auth):
        """Missing X-Service-Token → 401 when SERVICE_TOKEN is set."""
        resp = client_with_auth.post("/pad/check", json={
            "correlation_id": "test-auth",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 401

    def test_200_with_valid_token(self, client_with_auth):
        """Valid X-Service-Token → 200."""
        resp = client_with_auth.post(
            "/pad/check",
            json={
                "correlation_id": "test-auth-ok",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            },
            headers={"X-Service-Token": "TEST_TOKEN_PLACEHOLDER"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /pad/check — validation
# ---------------------------------------------------------------------------

class TestPadCheckValidation:
    def test_invalid_base64_returns_400(self, client):
        """Invalid base64 → 400."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-bad-b64",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": "!!!not-base64!!!",
        })
        assert resp.status_code == 400

    def test_missing_required_field_returns_422(self, client):
        """Missing face_photo → 422 validation error."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-missing",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
        })
        assert resp.status_code == 422

    def test_oversized_image_rejected(self, client):
        """Image > 8MB → 400."""
        # Create a base64 string that decodes to > 8MB
        big_data = b"\x00" * (8 * 1024 * 1024 + 1)
        big_b64 = base64.b64encode(big_data).decode()
        resp = client.post("/pad/check", json={
            "correlation_id": "test-oversize",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": big_b64,
        })
        assert resp.status_code == 400

    def test_transaction_type_other_than_sale_returns_422(self, client):
        """transaction_type is a closed enum — only 'sale' is confirmed in v1."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-txn-type",
            "transaction_type": "receive",
            "transaction_ref": "req:bal",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /pad/check — service-failure paths (TIMEOUT / INTERNAL_ERROR)
# ---------------------------------------------------------------------------

class TestPadCheckErrorPaths:
    def test_timeout_returns_low_quality_with_reason(self, client):
        """Inference exceeding the deadline → verdict=low_quality, reason=TIMEOUT (not a security verdict)."""
        import app.main as m
        original_timeout = m.INFERENCE_TIMEOUT_S
        original_predict = m.engine.predict
        m.INFERENCE_TIMEOUT_S = 0.05

        def _slow_predict(*_args, **_kwargs):
            time.sleep(0.3)
            return ("real", 0.95, True, {})

        m.engine.predict = _slow_predict
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-timeout",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.INFERENCE_TIMEOUT_S = original_timeout
            m.engine.predict = original_predict

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "low_quality"
        assert data["reason"] == "TIMEOUT"

    def test_internal_error_returns_low_quality_with_reason(self, client):
        """Unhandled exception during inference → verdict=low_quality, reason=INTERNAL_ERROR, no traceback leak."""
        import app.main as m
        from app.main import app
        from fastapi.testclient import TestClient

        original_predict = m.engine.predict

        def _raising_predict(*_args, **_kwargs):
            raise RuntimeError("model exploded")

        m.engine.predict = _raising_predict
        try:
            # raise_server_exceptions=False: Starlette's ServerErrorMiddleware re-raises
            # the original exception for the ASGI server after invoking our handler —
            # we want the actual 500 JSON response the client would see, not the raw traceback.
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.post("/pad/check", json={
                    "correlation_id": "test-internal-error",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
        finally:
            m.engine.predict = original_predict

        assert resp.status_code == 500
        data = resp.json()
        assert data["verdict"] == "low_quality"
        assert data["reason"] == "INTERNAL_ERROR"
        assert "model exploded" not in resp.text


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_200_when_loaded(self, client):
        """Models loaded → 200 + healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

    def test_health_503_when_not_loaded(self):
        """Models NOT loaded → 503."""
        import app.main as m
        m._models_loaded = False
        m.engine = None
        m.detector = None

        from app.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 503
            assert resp.json()["status"] == "not_ready"


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_rate_limit_after_burst(self, client):
        """Requests beyond burst limit → 429."""
        import app.main as m
        m._rate_limiter._burst = 5
        m._rate_limiter._windows.clear()
        responses = []
        for _ in range(10):
            resp = client.get("/health")
            responses.append(resp.status_code)
        # Restore high limit for other tests
        m._rate_limiter._burst = 1000
        assert 429 in responses, f"Expected 429 in {responses}"

    def test_sustained_limit_blocks_average_rate_even_under_burst(self):
        """Sustained (average req/s over the window) must cap requests even when
        burst never trips — the old `_sustained` field was dead code and would
        have let this pass unbounded."""
        from app.main import _RateLimiter

        limiter = _RateLimiter(burst=1000, sustained=1.0)  # 1 req/s avg -> 60 allowed per 60s window
        allowed = [limiter.allow("203.0.113.5") for _ in range(65)]

        assert False in allowed, "sustained limiter should reject once the average rate is exceeded"
        assert sum(allowed) <= 60

    def test_prune_removes_stale_ip_entries(self):
        """Stale/empty per-IP deques must be dropped eventually (memory-leak guard)."""
        from app.main import _RateLimiter

        limiter = _RateLimiter(burst=1000, sustained=1000.0)
        limiter.allow("198.51.100.1")
        assert "198.51.100.1" in limiter._windows

        far_future_cutoff = time.monotonic() + 1000.0
        limiter._prune_stale(far_future_cutoff)
        assert "198.51.100.1" not in limiter._windows


# ---------------------------------------------------------------------------
# Regression: no frame persistence anywhere (not on disk, not in audit log)
# ---------------------------------------------------------------------------

class TestNoFrameStorage:
    def test_pad_check_leaves_no_frame_artifact_on_disk_or_audit_log(self, client):
        """The service must never write the raw frame to disk or leak it into the audit log."""
        import app.main as m

        service_root = Path(m.__file__).resolve().parent.parent  # antispoof/
        excluded = {".venv", ".git", ".pytest_cache", "__pycache__", "models", "certs"}

        def _snapshot() -> set:
            files = set()
            for p in service_root.rglob("*"):
                if p.is_file() and not any(part in excluded for part in p.parts):
                    files.add(str(p.relative_to(service_root)))
            return files

        before = _snapshot()

        audit_records: list = []
        capture_handler = logging.Handler()
        capture_handler.emit = lambda record: audit_records.append(record.getMessage())
        m.audit_log.addHandler(capture_handler)

        b64_photo = _make_base64_image()
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-no-frame-storage",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": b64_photo,
            })
        finally:
            m.audit_log.removeHandler(capture_handler)

        assert resp.status_code == 200

        after = _snapshot()
        new_files = after - before
        assert not new_files, f"pad/check must not write new files to disk, found: {new_files}"

        assert audit_records, "expected an audit log entry to be written"
        assert all(b64_photo not in rec for rec in audit_records), "audit log must not contain the raw base64 frame"
        assert b64_photo not in resp.text, "response must not echo back the raw frame"
