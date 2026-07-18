"""Tests for the anti-replay timestamp-window guard (X-Request-Timestamp),
layered on top of mTLS (deploy/mtls/, BUSTA RHYMES) + the existing
X-Service-Token/IP-allowlist (both untouched by this). Applies to the three
money-path endpoints only: /pad/check, /liveness/challenge, /liveness/verdict.

See app/main.py::_verify_replay_protection and app/config.py's
REPLAY_PROTECTION_ENABLED / REPLAY_TOLERANCE_S for the design rationale.
"""
import base64
import os
import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

os.environ.setdefault("SERVICE_TOKEN", "")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("RATE_LIMIT_BURST", "1000")
os.environ.setdefault("RATE_LIMIT_SUSTAINED", "1000.0")
os.environ["ANTISPOOF_SKIP_MODELS"] = "1"


def _make_base64_image(width: int = 200, height: int = 200) -> str:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 3, (200, 180, 160), -1)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


def _pad_check_body() -> dict:
    return {
        "correlation_id": "replay-test",
        "transaction_type": "sale",
        "transaction_ref": "req:bal",
        "face_photo": _make_base64_image(),
    }


def _challenge_body() -> dict:
    return {"correlation_id": "replay-c1", "transaction_type": "sale", "transaction_ref": "r:b"}


def _frame(seq: int) -> dict:
    return {"seq": seq, "base64": _make_base64_image(), "captured_at": None}


def _mock_frame_face(yaw: float = 0.0):
    """Bbox 100x100 in a 200x200 frame => face_area_ratio=0.25, below the
    default FACE_RATIO_REJECT=0.27 geometry-gate threshold (see
    tests/test_liveness_endpoints.py for the same convention)."""
    from app.face_landmarks import FrameFace
    return FrameFace(
        bbox_xyxy=(50.0, 50.0, 150.0, 150.0),
        kps=np.array([[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]], dtype=np.float32),
        pose_pitch=0.0, pose_yaw=yaw, pose_roll=0.0, det_score=0.9, n_faces_detected=1,
    )


@pytest.fixture(autouse=True)
def _no_startup():
    """Prevent startup event from loading real models and reset rate limiter,
    same pattern as tests/test_pad_check.py and tests/test_liveness_endpoints.py."""
    import app.main as m
    from app.main import app
    original_handlers = app.router.on_startup.copy()
    app.router.on_startup.clear()
    m._rate_limiter._windows.clear()
    m._rate_limiter._burst = 1000
    m._rate_limiter._sustained = 1000.0
    yield
    app.router.on_startup = original_handlers


@pytest.fixture
def pad_check_client():
    """/pad/check with mocked detector/engine, replay protection OFF by
    default — each test flips settings.REPLAY_PROTECTION_ENABLED itself."""
    import app.main as m

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [50, 50, 100, 100]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = ("real", 0.95, True, {})
    m.detector = mock_detector
    m.engine = mock_engine
    m._models_loaded = True

    old_enabled = m.settings.REPLAY_PROTECTION_ENABLED
    old_tolerance = m.settings.REPLAY_TOLERANCE_S

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

    m.settings.REPLAY_PROTECTION_ENABLED = old_enabled
    m.settings.REPLAY_TOLERANCE_S = old_tolerance


@pytest.fixture
def liveness_client():
    """/liveness/challenge + /liveness/verdict with LIVENESS_ENDPOINTS_ENABLED=True
    and mocked landmark_detector/adaface_embedder (same pattern as
    tests/test_liveness_endpoints.py::client_enabled)."""
    import app.main as m

    m.settings.LIVENESS_ENDPOINTS_ENABLED = True
    m._liveness_models_loaded = True

    mock_landmark_detector = MagicMock()
    mock_landmark_detector.analyze.return_value = _mock_frame_face(yaw=0.0)
    mock_adaface = MagicMock()
    mock_adaface.embed_aligned.return_value = np.array([1.0, 0.0], dtype=np.float32)
    m.landmark_detector = mock_landmark_detector
    m.adaface_embedder = mock_adaface

    m.detector = MagicMock()
    m.detector.detect.return_value = [50, 50, 100, 100]
    m.engine = MagicMock()
    m.engine.predict.return_value = ("real", 0.95, True, {})
    m._models_loaded = True

    old_enabled = m.settings.REPLAY_PROTECTION_ENABLED
    old_tolerance = m.settings.REPLAY_TOLERANCE_S

    from app.main import app
    from fastapi.testclient import TestClient

    import app.face_landmarks as fl
    orig_align = fl.LandmarkDetector.align_112
    fl.LandmarkDetector.align_112 = staticmethod(lambda img, kps: img)

    with TestClient(app) as c:
        yield c, m

    fl.LandmarkDetector.align_112 = staticmethod(orig_align)
    m.settings.LIVENESS_ENDPOINTS_ENABLED = False
    m._liveness_models_loaded = False
    m.settings.REPLAY_PROTECTION_ENABLED = old_enabled
    m.settings.REPLAY_TOLERANCE_S = old_tolerance


# ---------------------------------------------------------------------------
# POST /pad/check
# ---------------------------------------------------------------------------

class TestReplayProtectionPadCheck:
    def test_valid_timestamp_passes_when_enabled(self, pad_check_client):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": str(time.time())},
        )
        assert resp.status_code == 200

    def test_old_timestamp_rejected(self, pad_check_client):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        old_ts = time.time() - 200  # > 120s in the past
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": str(old_ts)},
        )
        assert resp.status_code == 401

    def test_future_timestamp_rejected(self, pad_check_client):
        """Clock-skew in the OTHER direction (request appears to be from the
        future) must also be rejected, not just stale requests."""
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        future_ts = time.time() + 200  # > 120s ahead
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": str(future_ts)},
        )
        assert resp.status_code == 401

    def test_missing_header_rejected_when_enabled(self, pad_check_client):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.post("/pad/check", json=_pad_check_body())
        assert resp.status_code == 401

    def test_non_numeric_header_rejected(self, pad_check_client):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": "not-a-number"},
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("nan_value", ["nan", "NaN", "+nan", "-nan", "NAN"])
    def test_nan_timestamp_rejected(self, pad_check_client, nan_value):
        """2PAC (2026-07-18): float("nan") parses successfully and every
        comparison against nan is False per IEEE754 — the naive `>` window
        check would silently never raise for a nan timestamp, a full bypass
        of this guard. math.isfinite() must catch it explicitly."""
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": nan_value},
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize("inf_value", ["inf", "-inf", "Infinity"])
    def test_infinite_timestamp_rejected(self, pad_check_client, inf_value):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": inf_value},
        )
        assert resp.status_code == 401

    def test_boundary_just_inside_tolerance_passes(self, pad_check_client):
        """Check uses strict `>` — exactly at (tolerance - 1)s the window is
        still inclusive, request must pass."""
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        ts = time.time() - 119  # 1s inside the 120s tolerance
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": str(ts)},
        )
        assert resp.status_code == 200

    def test_boundary_just_outside_tolerance_rejected(self, pad_check_client):
        """Exactly at (tolerance + 1)s the request must be rejected."""
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        ts = time.time() - 121  # 1s outside the 120s tolerance
        resp = pad_check_client.post(
            "/pad/check", json=_pad_check_body(),
            headers={"X-Request-Timestamp": str(ts)},
        )
        assert resp.status_code == 401

    def test_disabled_by_default_missing_header_still_passes(self, pad_check_client):
        """Backward compatibility: REPLAY_PROTECTION_ENABLED defaults False —
        a partner that has not rolled out the header yet must be unaffected."""
        import app.main as m
        assert m.settings.REPLAY_PROTECTION_ENABLED is False
        resp = pad_check_client.post("/pad/check", json=_pad_check_body())
        assert resp.status_code == 200

    def test_disabled_explicitly_missing_header_still_passes(self, pad_check_client):
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = False
        resp = pad_check_client.post("/pad/check", json=_pad_check_body())
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /liveness/challenge
# ---------------------------------------------------------------------------

class TestReplayProtectionLivenessChallenge:
    def test_valid_timestamp_passes_when_enabled(self, liveness_client):
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/challenge", json=_challenge_body(),
            headers={"X-Request-Timestamp": str(time.time())},
        )
        assert resp.status_code == 200

    def test_old_timestamp_rejected(self, liveness_client):
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        resp = client.post(
            "/liveness/challenge", json=_challenge_body(),
            headers={"X-Request-Timestamp": str(time.time() - 200)},
        )
        assert resp.status_code == 401

    def test_future_timestamp_rejected(self, liveness_client):
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        resp = client.post(
            "/liveness/challenge", json=_challenge_body(),
            headers={"X-Request-Timestamp": str(time.time() + 200)},
        )
        assert resp.status_code == 401

    def test_missing_header_rejected_when_enabled(self, liveness_client):
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post("/liveness/challenge", json=_challenge_body())
        assert resp.status_code == 401

    def test_non_numeric_header_rejected(self, liveness_client):
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/challenge", json=_challenge_body(),
            headers={"X-Request-Timestamp": "banana"},
        )
        assert resp.status_code == 401

    def test_nan_timestamp_rejected(self, liveness_client):
        """2PAC (2026-07-18) NaN-bypass regression — see the pad/check
        variant of this test for the full IEEE754 explanation."""
        client, m = liveness_client
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/challenge", json=_challenge_body(),
            headers={"X-Request-Timestamp": "nan"},
        )
        assert resp.status_code == 401

    def test_disabled_missing_header_still_passes(self, liveness_client):
        client, m = liveness_client
        assert m.settings.REPLAY_PROTECTION_ENABLED is False
        resp = client.post("/liveness/challenge", json=_challenge_body())
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /liveness/verdict
# ---------------------------------------------------------------------------

class TestReplayProtectionLivenessVerdict:
    def _verdict_body(self, client, m) -> dict:
        """Mint a real session via /liveness/challenge (replay protection OFF
        at mint time) so the verdict call has a session_id that will pass
        session lookup once past the replay check under test."""
        m.settings.REPLAY_PROTECTION_ENABLED = False
        ch = client.post("/liveness/challenge", json=_challenge_body()).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        return {
            "correlation_id": "replay-c1", "session_id": ch["session_id"],
            "transaction_type": "sale", "transaction_ref": "r:b", "frames": frames,
        }

    def test_valid_timestamp_reaches_live(self, liveness_client):
        client, m = liveness_client
        yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]
        body = self._verdict_body(client, m)

        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/verdict", json=body,
            headers={"X-Request-Timestamp": str(time.time())},
        )
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"

    def test_old_timestamp_rejected(self, liveness_client):
        client, m = liveness_client
        body = self._verdict_body(client, m)
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        resp = client.post(
            "/liveness/verdict", json=body,
            headers={"X-Request-Timestamp": str(time.time() - 200)},
        )
        assert resp.status_code == 401

    def test_future_timestamp_rejected(self, liveness_client):
        client, m = liveness_client
        body = self._verdict_body(client, m)
        m.settings.REPLAY_PROTECTION_ENABLED = True
        m.settings.REPLAY_TOLERANCE_S = 120
        resp = client.post(
            "/liveness/verdict", json=body,
            headers={"X-Request-Timestamp": str(time.time() + 200)},
        )
        assert resp.status_code == 401

    def test_missing_header_rejected_when_enabled(self, liveness_client):
        client, m = liveness_client
        body = self._verdict_body(client, m)
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post("/liveness/verdict", json=body)
        assert resp.status_code == 401

    def test_non_numeric_header_rejected(self, liveness_client):
        client, m = liveness_client
        body = self._verdict_body(client, m)
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/verdict", json=body,
            headers={"X-Request-Timestamp": "not-numeric"},
        )
        assert resp.status_code == 401

    def test_nan_timestamp_rejected(self, liveness_client):
        """2PAC (2026-07-18) NaN-bypass regression — see the pad/check
        variant of this test for the full IEEE754 explanation."""
        client, m = liveness_client
        body = self._verdict_body(client, m)
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = client.post(
            "/liveness/verdict", json=body,
            headers={"X-Request-Timestamp": "nan"},
        )
        assert resp.status_code == 401

    def test_disabled_missing_header_still_passes(self, liveness_client):
        client, m = liveness_client
        yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]
        body = self._verdict_body(client, m)
        assert m.settings.REPLAY_PROTECTION_ENABLED is False
        resp = client.post("/liveness/verdict", json=body)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Health/service endpoints must be untouched by this feature
# ---------------------------------------------------------------------------

class TestVerifyReplayProtectionUnit:
    """Direct unit coverage of app.main._verify_replay_protection — the NaN
    bypass 2PAC found (2026-07-18) was proven by calling this function
    directly, not just through the HTTP layer; keep that direct repro."""

    @pytest.mark.parametrize("value", ["nan", "NaN", "+nan", "-nan", "NAN", "inf", "-inf", "Infinity"])
    def test_non_finite_values_raise_401(self, value):
        import app.main as m
        from fastapi import HTTPException

        old_enabled = m.settings.REPLAY_PROTECTION_ENABLED
        m.settings.REPLAY_PROTECTION_ENABLED = True
        try:
            with pytest.raises(HTTPException) as exc_info:
                m._verify_replay_protection(value)
            assert exc_info.value.status_code == 401
        finally:
            m.settings.REPLAY_PROTECTION_ENABLED = old_enabled

    def test_valid_current_timestamp_does_not_raise(self):
        import app.main as m

        old_enabled = m.settings.REPLAY_PROTECTION_ENABLED
        m.settings.REPLAY_PROTECTION_ENABLED = True
        try:
            m._verify_replay_protection(str(time.time()))  # must not raise
        finally:
            m.settings.REPLAY_PROTECTION_ENABLED = old_enabled


class TestReplayProtectionScope:
    def test_health_endpoint_not_gated_by_replay_protection(self, pad_check_client):
        """/health is not a money-path endpoint — must never require
        X-Request-Timestamp, even with the flag enabled."""
        import app.main as m
        m.settings.REPLAY_PROTECTION_ENABLED = True
        resp = pad_check_client.get("/health")
        assert resp.status_code == 200
