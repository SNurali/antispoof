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
# POST /pad/check — Layer 0a deterministic face-to-frame geometry gate
# ---------------------------------------------------------------------------

class TestPadCheckGeometryLayer:
    def test_large_face_ratio_short_circuits_to_document_photo(self, client):
        """A bbox filling most of the frame (document-photo profile, per the
        incident_urgut spoof calibration: ratio 0.472) => verdict=spoof,
        reason=DOCUMENT_PHOTO, WITHOUT ever calling passive-PAD."""
        import app.main as m

        # 200x200 frame (from _make_test_image), bbox covering 70% width/height
        # => area ratio = 0.49, above the default 0.27 threshold (re-calibrated
        # 2026-07-16, see app/config.py).
        m.detector.detect.return_value = [10, 10, 140, 140]
        try:
            with patch.object(m.engine, "predict") as mock_predict:
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-geo-reject",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
            mock_predict.assert_not_called()
        finally:
            m.detector.detect.return_value = [50, 50, 100, 100]

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["reason"] == "DOCUMENT_PHOTO"
        assert data["save_frame"] is True
        assert data["signals"]["geometry_check"]["face_area_ratio"] == pytest.approx(0.49)

    def test_normal_selfie_face_ratio_falls_through_to_passive_pad(self, client):
        """Default fixture bbox [50,50,100,100] on a 200x200 frame => area
        ratio 0.25 (below the 0.27 area threshold) and width ratio 0.5
        (below the 0.55 width threshold) — must NOT be rejected by the
        geometry layer; passive-PAD's verdict (mocked 'real') is what's
        returned."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-geo-pass",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": _make_base64_image(),
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"

    def test_disabled_flag_skips_geometry_layer_even_with_large_face(self, client):
        """GEOMETRY_CHECK_ENABLED=False => large-face frames must NOT be
        rejected by this layer; passive-PAD alone decides."""
        import app.main as m

        m.settings.GEOMETRY_CHECK_ENABLED = False
        m.detector.detect.return_value = [10, 10, 140, 140]  # ratio 0.49, would reject if enabled
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-geo-disabled",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.settings.GEOMETRY_CHECK_ENABLED = True
            m.detector.detect.return_value = [50, 50, 100, 100]

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"  # falls through to mocked passive-PAD

    def test_malformed_bbox_fails_safe_to_passive_pad(self, client):
        """A detector bbox the geometry check can't process must not crash
        the request — falls through to passive-PAD unchanged."""
        import app.main as m

        m.detector.detect.return_value = [0, 0, 0, 0]  # triggers INVALID_DIMENSIONS
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-geo-malformed",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.detector.detect.return_value = [50, 50, 100, 100]

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"


# ---------------------------------------------------------------------------
# POST /pad/check — Layer 0c deterministic frame-sharpness gate (RZA, 2026-07-21)
# ---------------------------------------------------------------------------

def _make_blurry_base64_image(width: int = 200, height: int = 200) -> str:
    """Flat-color image — Laplacian variance well below MIN_FACE_SHARPNESS_224
    (60.0), stands in for a smeared/motion-blurred attack frame."""
    return _make_base64_image(width, height)  # the default fixture is already flat/low-detail


def _make_sharp_base64_image(width: int = 200, height: int = 200) -> str:
    """Textured image — Laplacian variance well above MIN_FACE_SHARPNESS_224,
    stands in for a real in-focus face crop."""
    rng = np.random.default_rng(42)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 3, (200, 180, 160), -1)
    noise = rng.integers(-30, 30, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


class TestPadCheckSharpnessLayer:
    def test_disabled_by_default_never_blocks_blurry_frame(self, client):
        """FRAME_SHARPNESS_CHECK_ENABLED defaults to False (see app/config.py
        for why: n=1 subject calibration, existing test fixtures reuse a
        flat/low-detail synthetic image) — a blurry frame must still fall
        through to passive-PAD unchanged."""
        resp = client.post("/pad/check", json={
            "correlation_id": "test-sharp-disabled",
            "transaction_type": "sale",
            "transaction_ref": "req:bal",
            "face_photo": _make_blurry_base64_image(),
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"  # mocked passive-PAD, gate never ran

    def test_enabled_blurry_frame_short_circuits_to_low_quality(self, client):
        """FRAME_SHARPNESS_CHECK_ENABLED=True + a below-threshold frame =>
        verdict=low_quality, reason=BLURRY, WITHOUT ever calling passive-PAD
        (engine.predict) — mirrors TestPadCheckGeometryLayer's pattern."""
        import app.main as m

        m.settings.FRAME_SHARPNESS_CHECK_ENABLED = True
        try:
            with patch.object(m.engine, "predict") as mock_predict:
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-sharp-blocked",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_blurry_base64_image(),
                })
            mock_predict.assert_not_called()
        finally:
            m.settings.FRAME_SHARPNESS_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "low_quality"
        assert data["reason"] == "BLURRY"
        assert data["save_frame"] is False
        assert "sharpness_check" in data["signals"]

    def test_enabled_sharp_frame_falls_through_to_passive_pad(self, client):
        """A textured (non-blurry) frame must NOT be rejected by this layer
        even when enabled — passive-PAD's mocked verdict is what's returned."""
        import app.main as m

        m.settings.FRAME_SHARPNESS_CHECK_ENABLED = True
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-sharp-pass",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_sharp_base64_image(),
            })
        finally:
            m.settings.FRAME_SHARPNESS_CHECK_ENABLED = False

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"


# ---------------------------------------------------------------------------
# POST /pad/check — Layer 0d face-angle gate (RZA, 2026-07-21)
# ---------------------------------------------------------------------------

class TestPadCheckPoseLayer:
    def test_disabled_by_default_is_noop_even_with_landmark_detector_present(self, client):
        """POSE_CHECK_ENABLED defaults to False — even if a landmark_detector
        happened to be loaded, this layer must not run."""
        import app.main as m

        mock_landmark = MagicMock()
        mock_landmark.analyze.return_value = MagicMock(pose_yaw=80.0, pose_pitch=0.0)
        m.landmark_detector = mock_landmark
        m._liveness_models_loaded = True
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-pose-disabled",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.landmark_detector = None
            m._liveness_models_loaded = False

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"
        mock_landmark.analyze.assert_not_called()

    def test_enabled_but_landmark_detector_missing_fails_open(self, client):
        """POSE_CHECK_ENABLED=True but landmark_detector is None (
        LIVENESS_ENDPOINTS_ENABLED never turned on) — silent no-op per
        app/pose_check.py's documented limitation, falls through unchanged."""
        import app.main as m

        m.settings.POSE_CHECK_ENABLED = True
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-pose-no-detector",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.settings.POSE_CHECK_ENABLED = False

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"

    def test_enabled_off_angle_short_circuits_to_low_quality(self, client):
        """POSE_CHECK_ENABLED=True + landmark_detector reporting an off-angle
        face => verdict=low_quality, reason=OFF_ANGLE, WITHOUT ever calling
        passive-PAD (engine.predict)."""
        import app.main as m

        mock_landmark = MagicMock()
        mock_landmark.analyze.return_value = MagicMock(pose_yaw=55.0, pose_pitch=0.0)
        m.landmark_detector = mock_landmark
        m._liveness_models_loaded = True
        m.settings.POSE_CHECK_ENABLED = True
        try:
            with patch.object(m.engine, "predict") as mock_predict:
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-pose-blocked",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
            mock_predict.assert_not_called()
        finally:
            m.settings.POSE_CHECK_ENABLED = False
            m.landmark_detector = None
            m._liveness_models_loaded = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "low_quality"
        assert data["reason"] == "OFF_ANGLE"
        assert data["save_frame"] is False
        assert data["signals"]["pose_check"]["pose_yaw"] == pytest.approx(55.0)

    def test_enabled_frontal_falls_through_to_passive_pad(self, client):
        """A frontal (in-threshold) pose must NOT be rejected by this layer
        even when enabled — passive-PAD's mocked verdict is what's returned."""
        import app.main as m

        mock_landmark = MagicMock()
        mock_landmark.analyze.return_value = MagicMock(pose_yaw=2.0, pose_pitch=-1.0)
        m.landmark_detector = mock_landmark
        m._liveness_models_loaded = True
        m.settings.POSE_CHECK_ENABLED = True
        try:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-pose-pass",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        finally:
            m.settings.POSE_CHECK_ENABLED = False
            m.landmark_detector = None
            m._liveness_models_loaded = False

        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"


# ---------------------------------------------------------------------------
# POST /pad/check — Layer 0b document/passport-photo pre-filter (minicpm-v)
# ---------------------------------------------------------------------------

class TestPadCheckDocumentLayer:
    def test_disabled_by_default_never_calls_document_checker(self, client):
        """DOCUMENT_CHECK_ENABLED defaults to False — the layer must not run
        at all (and passive-PAD behavior must be untouched) unless explicitly
        turned on."""
        import app.main as m
        assert m.settings.DOCUMENT_CHECK_ENABLED is False

        with patch.object(m.document_checker, "check") as mock_check:
            resp = client.post("/pad/check", json={
                "correlation_id": "test-doc-disabled",
                "transaction_type": "sale",
                "transaction_ref": "req:bal",
                "face_photo": _make_base64_image(),
            })
        assert resp.status_code == 200
        mock_check.assert_not_called()

    def test_enabled_high_confidence_document_short_circuits_to_spoof(self, client):
        """Layer 0 confidently flags a document photo => verdict=spoof,
        reason=DOCUMENT_PHOTO, WITHOUT ever calling passive-PAD (engine.predict)."""
        import app.main as m
        from app.document_check import DocumentCheckResult

        m.settings.DOCUMENT_CHECK_ENABLED = True
        m.settings.DOCUMENT_REJECT_THRESHOLD = 0.70
        try:
            with patch.object(
                m.document_checker, "check",
                return_value=DocumentCheckResult(
                    ran=True, is_document=True, confidence=0.92,
                    raw_label="STUDIO_BACKGROUND", raw_response="LABEL=STUDIO_BACKGROUND CONFIDENCE=92",
                ),
            ), patch.object(m.engine, "predict") as mock_predict:
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-doc-reject",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
            mock_predict.assert_not_called()
        finally:
            m.settings.DOCUMENT_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["reason"] == "DOCUMENT_PHOTO"
        assert data["save_frame"] is True
        assert data["signals"]["document_check"]["label"] == "STUDIO_BACKGROUND"

    def test_enabled_below_threshold_falls_through_to_passive_pad(self, client):
        """Confidence below DOCUMENT_REJECT_THRESHOLD => must NOT short-circuit;
        passive-PAD still runs and its verdict is what's returned."""
        import app.main as m
        from app.document_check import DocumentCheckResult

        m.settings.DOCUMENT_CHECK_ENABLED = True
        m.settings.DOCUMENT_REJECT_THRESHOLD = 0.70
        try:
            with patch.object(
                m.document_checker, "check",
                return_value=DocumentCheckResult(
                    ran=True, is_document=True, confidence=0.40,
                    raw_label="STUDIO_BACKGROUND", raw_response="LABEL=STUDIO_BACKGROUND CONFIDENCE=40",
                ),
            ):
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-doc-below-threshold",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
        finally:
            m.settings.DOCUMENT_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        # Falls through to the mocked passive-PAD engine ("real", 0.95 from `client` fixture)
        assert data["verdict"] == "live"

    def test_enabled_ollama_unavailable_fails_open_to_passive_pad(self, client):
        """Layer 0 fails (ran=False) => must fall through to passive-PAD
        unchanged, never surface an error to the caller."""
        import app.main as m
        from app.document_check import DocumentCheckResult

        m.settings.DOCUMENT_CHECK_ENABLED = True
        try:
            with patch.object(
                m.document_checker, "check",
                return_value=DocumentCheckResult(ran=False, error="OLLAMA_UNAVAILABLE: connection refused"),
            ):
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-doc-failopen",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
        finally:
            m.settings.DOCUMENT_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "live"  # passive-PAD's mocked verdict, untouched

    def test_enabled_outer_timeout_fails_open_to_passive_pad(self, client):
        """Even if the checker call itself hangs past the outer asyncio timeout,
        /pad/check must still resolve via passive-PAD, not hang or error.
        (The underlying thread is not killed — Python threads can't be — but
        the request path must not wait for it; it abandons and moves on.)"""
        import app.main as m
        import time as _time

        m.settings.DOCUMENT_CHECK_ENABLED = True
        m.settings.DOCUMENT_CHECK_TIMEOUT_S = 0.05  # outer wait = 1.05s
        try:
            def _slow_check(*_a, **_kw):
                _time.sleep(2.0)
                from app.document_check import DocumentCheckResult
                return DocumentCheckResult(ran=True, is_document=True, confidence=0.99)

            with patch.object(m.document_checker, "check", side_effect=_slow_check):
                t0 = _time.monotonic()
                resp = client.post("/pad/check", json={
                    "correlation_id": "test-doc-outer-timeout",
                    "transaction_type": "sale",
                    "transaction_ref": "req:bal",
                    "face_photo": _make_base64_image(),
                })
                elapsed = _time.monotonic() - t0
        finally:
            m.settings.DOCUMENT_CHECK_ENABLED = False
            m.settings.DOCUMENT_CHECK_TIMEOUT_S = 20.0

        assert resp.status_code == 200
        assert elapsed < 2.0, "request must not block for the full slow-checker duration"
        data = resp.json()
        assert data["verdict"] == "live"


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
