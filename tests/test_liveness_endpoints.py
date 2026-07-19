"""Tests for POST /liveness/challenge and POST /liveness/verdict — endpoint
wiring, auth, session lifecycle, and fusion-order behavior. Landmark
detector / AdaFace embedder are MOCKED (same pattern as test_pad_check.py's
detector/engine mocks) so these tests do not require the real 260MB ONNX
weight file or insightface's buffalo_l bundle to be present."""
import base64
import importlib
import logging
import os
from datetime import datetime, timedelta, timezone
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


def _frame(seq: int) -> dict:
    return {"seq": seq, "base64": _make_base64_image(), "captured_at": None}


def _yaw_for_step(step: str) -> float:
    return 25.0 if step == "TURN_LEFT" else -25.0


def _yaws_matching_step_order(steps: list[str]) -> list[float]:
    """Builds a [frontal, evidence(steps[0]), frontal, evidence(steps[1])]
    yaw sequence matching WHATEVER order /liveness/challenge actually
    returned. Phase 3.1 (order-by-evidence, CHALLENGE_ENTROPY_SPRINT_v1.md
    §6.1) requires each requested step's evidence frame to have a strictly
    LATER seq than the previous step's — tests can no longer hardcode a
    fixed TURN_LEFT-then-TURN_RIGHT order now that `steps` order is
    genuinely randomized (secrets ГСЧ, Фаза 0/2), so callers must build the
    yaw sequence from the actual `challenge_spec.steps` returned."""
    assert len(steps) == 2, "helper assumes today's 2-step pool (TURN_LEFT,TURN_RIGHT)"
    return [0.0, _yaw_for_step(steps[0]), 0.0, _yaw_for_step(steps[1])]


@pytest.fixture(autouse=True)
def _no_startup():
    import app.main as m
    from app.main import app
    original_handlers = app.router.on_startup.copy()
    app.router.on_startup.clear()
    m._rate_limiter._windows.clear()
    m._rate_limiter._burst = 1000
    m._rate_limiter._sustained = 1000.0
    yield
    app.router.on_startup = original_handlers


def _mock_frame_face(yaw=0.0, n_faces=1, bbox_xyxy=(50.0, 50.0, 150.0, 150.0)):
    """Default bbox is 100x100 in a 200x200 frame (_make_base64_image's
    default size) => face_area_ratio=0.25, BELOW GEOMETRY_CHECK_ENABLED's
    default FACE_RATIO_REJECT=0.27 (app/config.py) — deliberately chosen so
    existing non-geometry tests don't spuriously trip the Layer 0a gate
    added 2026-07-17. Also satisfies frame_qc.MIN_FACE_EDGE_PX=60 (edge=100).
    Tests that specifically exercise the geometry gate pass a bigger,
    document-like bbox_xyxy explicitly (see TestLivenessGeometryGate)."""
    from app.face_landmarks import FrameFace
    return FrameFace(
        bbox_xyxy=bbox_xyxy,
        kps=np.array([[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]], dtype=np.float32),
        pose_pitch=0.0, pose_yaw=yaw, pose_roll=0.0, det_score=0.9, n_faces_detected=n_faces,
    )


@pytest.fixture
def client_disabled():
    """LIVENESS_ENDPOINTS_ENABLED=False (default) — endpoints must 503, not crash."""
    import app.main as m
    m.settings.LIVENESS_ENDPOINTS_ENABLED = False
    m._liveness_models_loaded = False
    m.detector = MagicMock()
    m.engine = MagicMock()
    m._models_loaded = True

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_enabled():
    """LIVENESS_ENDPOINTS_ENABLED=True with mocked landmark_detector/adaface_embedder."""
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


class TestDisabledByDefault:
    def test_challenge_returns_503_when_disabled(self, client_disabled):
        resp = client_disabled.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        })
        assert resp.status_code == 503

    def test_verdict_returns_503_when_disabled(self, client_disabled):
        resp = client_disabled.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": "s1", "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": [_frame(0)],
        })
        assert resp.status_code == 503


class TestChallengeEndpoint:
    def test_issues_session_with_steps_from_pool(self, client_enabled):
        client, m = client_enabled
        resp = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"]
        pool = set(s.strip() for s in m.settings.LIVENESS_CHALLENGE_STEPS_POOL.split(","))
        assert set(data["challenge_spec"]["steps"]).issubset(pool)
        assert data["challenge_spec"]["min_frames"] == m.settings.LIVENESS_MIN_FRAMES
        assert data["challenge_spec"]["max_frames"] == m.settings.LIVENESS_MAX_FRAMES
        assert "model_version" in data

    def test_step_count_within_configured_range(self, client_enabled):
        """Фаза 2 (§5.1): k теперь варьируется в [MIN, MAX], клампленный к
        размеру пула — с сегодняшним 2-элементным пулом это схлопывается в
        k=2 детерминированно (см. app/liveness_session.py::
        generate_challenge_spec docstring), но контракт сам по себе должен
        соблюдать диапазон, не полагаясь на сегодняшний размер пула."""
        client, m = client_enabled
        pool_size = len(set(s.strip() for s in m.settings.LIVENESS_CHALLENGE_STEPS_POOL.split(",")))
        expected_min = min(m.settings.LIVENESS_CHALLENGE_STEP_COUNT_MIN, pool_size)
        expected_max = min(m.settings.LIVENESS_CHALLENGE_STEP_COUNT_MAX, pool_size)
        for _ in range(10):
            data = client.post("/liveness/challenge", json={
                "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
            }).json()
            assert expected_min <= len(data["challenge_spec"]["steps"]) <= expected_max

    def test_step_windows_present_and_valid(self, client_enabled):
        """Фаза 2 (§5.3) / DoD п.2 плана: step_windows присутствует в ответе,
        покрывает КАЖДЫЙ выбранный шаг в том же порядке, и min<=max для
        каждого окна."""
        client, m = client_enabled
        data = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        steps = data["challenge_spec"]["steps"]
        windows = data["challenge_spec"]["step_windows"]
        assert [w["step"] for w in windows] == steps
        for w in windows:
            assert w["min_delay_ms"] <= w["max_delay_ms"]
            assert m.settings.LIVENESS_STEP_DELAY_MIN_MS <= w["min_delay_ms"]
            assert w["max_delay_ms"] <= m.settings.LIVENESS_STEP_DELAY_MAX_MS

    def test_transaction_type_not_validated(self, client_enabled):
        """Unlike /pad/check's Literal['sale'], this endpoint passes
        transaction_type through unchecked (explicit contract requirement)."""
        client, _ = client_enabled
        resp = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "receive", "transaction_ref": "r:b",
        })
        assert resp.status_code == 200


class TestVerdictEndpoint:
    def test_rejects_too_many_frames_with_422(self, client_enabled):
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        too_many = [_frame(i) for i in range(m.settings.LIVENESS_MAX_FRAMES + 1)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": too_many,
        })
        assert resp.status_code == 422

    def test_rejects_empty_frames_with_422(self, client_enabled):
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": [],
        })
        assert resp.status_code == 422

    def test_rejects_duplicate_seq_with_422(self, client_enabled):
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        frames = [_frame(0), _frame(0), _frame(1), _frame(2)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 422

    def test_unknown_session_id_returns_incomplete(self, client_enabled):
        client, m = client_enabled
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": "does-not-exist", "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "incomplete"
        assert data["reason"] == "SESSION_NOT_FOUND"

    def test_session_replay_rejected(self, client_enabled):
        client, m = client_enabled
        # force TURN_LEFT/TURN_RIGHT to actually be satisfiable: mock analyze
        # to alternate yaw across calls so active-challenge can pass.
        yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws] * 3

        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]

        first = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert first.status_code == 200

        second = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert second.status_code == 200
        data = second.json()
        assert data["verdict"] == "incomplete"
        assert data["reason"] == "SESSION_ALREADY_USED"

    def test_correlation_id_mismatch_rejected(self, client_enabled):
        """R2 binding is keyed on correlation_id (per backend confirmation,
        agent-mesh 2026-07-17) — a verdict request for a DIFFERENT
        correlation_id than the one the session was minted for must be
        rejected, even with a matching session_id.

        P0-5 (2026-07-18) integration test: mints a real session via
        POST /liveness/challenge with correlation_id=A, then calls
        POST /liveness/verdict with a deliberately different correlation_id=B
        — exercising the exact real-integration scenario the code's own
        'soft in this increment' comment (app/main.py::_run_liveness_verdict)
        and docs/LIVENESS_CONTRACT_v1.md §3 flagged as unverified. Both A and
        B must be present in `signals` for Laravel/audit diagnostics."""
        client, m = client_enabled
        correlation_id_a = "correlation-A-session-created-with"
        correlation_id_b = "correlation-B-sent-with-verdict"
        ch = client.post("/liveness/challenge", json={
            "correlation_id": correlation_id_a, "transaction_type": "sale", "transaction_ref": "req1:ball1",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": correlation_id_b, "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "req1:ball1", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "incomplete"
        assert data["reason"] == "SESSION_CORRELATION_MISMATCH"
        # Both values present for diagnostics — session-side vs request-side.
        assert data["signals"]["session_correlation_id"] == correlation_id_a
        assert data["signals"]["request_correlation_id"] == correlation_id_b
        # Response-level correlation_id echoes the REQUEST's value (not the
        # session's) — same echo convention as every other /liveness/verdict path.
        assert data["correlation_id"] == correlation_id_b

    def test_transaction_ref_mismatch_alone_is_not_rejected(self, client_enabled):
        """transaction_ref is passthrough-only for /liveness/* (the sale
        natural key can legitimately not be final yet when the challenge is
        issued) — a mismatch on transaction_ref ALONE, with a matching
        correlation_id, must NOT be treated as a binding violation. Uses a
        yaw sequence that also satisfies the active-challenge steps so the
        request can reach `live`, proving the mismatch truly did not block
        anything downstream."""
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "req1:ball1",
        }).json()
        # Yaw sequence built from the ACTUAL step order the challenge
        # returned — order-by-evidence (Phase 3.1) means a hardcoded
        # TURN_LEFT-then-TURN_RIGHT sequence would be flaky now that step
        # order is genuinely randomized (secrets ГСЧ).
        yaws = _yaws_matching_step_order(ch["challenge_spec"]["steps"])
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "A_DIFFERENT_SALE_REF", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reason"] != "SESSION_CORRELATION_MISMATCH"
        assert data["verdict"] == "live"

    def test_static_frames_fail_active_challenge_not_live(self, client_enabled):
        """Default mock returns yaw=0.0 for every frame — a real
        TURN_LEFT/TURN_RIGHT challenge must NOT pass on a static series."""
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] in ("spoof", "incomplete")
        assert data["verdict"] != "live"

    def test_full_happy_path_reaches_live(self, client_enabled):
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        yaws = _yaws_matching_step_order(ch["challenge_spec"]["steps"])
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "live"
        assert data["reason"] is None
        assert data["best_frame_seq"] in [f["seq"] for f in frames]
        assert data["frame_consistency_score"] == pytest.approx(1.0)

    def test_auth_required_when_token_set(self, client_enabled):
        client, m = client_enabled
        old_token = m.settings.SERVICE_TOKEN
        m.settings.SERVICE_TOKEN = "SECRET"
        try:
            resp = client.post("/liveness/challenge", json={
                "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
            })
            assert resp.status_code == 401
        finally:
            m.settings.SERVICE_TOKEN = old_token


class TestLivenessGeometryGate:
    """Layer 0a (app/geometry_check.py) reused inside /liveness/verdict,
    2026-07-17 — same calibrated logic/threshold already proven for
    /pad/check, not a new gate. `_frame()` always encodes a 200x200 JPEG
    (see _make_base64_image), so a bbox filling most of that frame reproduces
    the document-photo signature (see app/geometry_check.py for the real
    incident_urgut calibration numbers this threshold is based on)."""

    def _document_like_face(self, yaw=0.0):
        # bbox 180x180 in a 200x200 frame => face_area_ratio=0.81, far above
        # FACE_RATIO_REJECT=0.27 — same shape of signal as the real
        # passport-style spoof incidents documented in geometry_check.py.
        return _mock_frame_face(yaw=yaw, bbox_xyxy=(10.0, 10.0, 190.0, 190.0))

    def test_document_like_bbox_flagged_as_spoof(self, client_enabled):
        client, m = client_enabled
        m.landmark_detector.analyze.return_value = self._document_like_face()

        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "spoof"
        assert data["reason"] == "DOCUMENT_PHOTO"
        assert "layer0a_geometry_check" in data["signals"]

    def test_document_like_bbox_flagged_even_when_other_frames_would_pass(self, client_enabled):
        """A SINGLE flagged frame must fail the whole session — same
        conservative posture as Layer 1's 'any_frame_spoof' aggregation."""
        client, m = client_enabled
        yaws = [0.0, 25.0, 0.0, -25.0]
        faces = [_mock_frame_face(yaw=y) for y in yaws]
        faces[1] = self._document_like_face(yaw=25.0)  # one bad frame among good ones
        m.landmark_detector.analyze.side_effect = faces

        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "spoof"
        assert resp.json()["reason"] == "DOCUMENT_PHOTO"

    def test_geometry_gate_disabled_falls_through_to_normal_pipeline(self, client_enabled):
        """GEOMETRY_CHECK_ENABLED=False must restore the pre-2026-07-17
        behavior exactly — a document-like bbox alone must not block a
        session that otherwise satisfies the active challenge."""
        client, m = client_enabled
        m.settings.GEOMETRY_CHECK_ENABLED = False
        try:
            ch = client.post("/liveness/challenge", json={
                "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "r:b",
            }).json()
            yaws = _yaws_matching_step_order(ch["challenge_spec"]["steps"])
            m.landmark_detector.analyze.side_effect = [self._document_like_face(yaw=y) for y in yaws]

            frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": ch["session_id"], "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["reason"] != "DOCUMENT_PHOTO"
            assert data["verdict"] == "live"
        finally:
            m.settings.GEOMETRY_CHECK_ENABLED = True


class TestLivenessVerdictInternalError:
    """P0-1 (2026-07-18): an unhandled exception AFTER the request body has
    already been parsed must be caught locally in the route handler (not the
    generic app.exception_handler(Exception)) so the response carries the
    correct model_version, the documented -1.0 'not computed' sentinel for
    frame_consistency_score, and the request's own echo fields (session_id/
    correlation_id/transaction_type/transaction_ref) instead of null."""

    def test_internal_error_after_parsing_echoes_request_fields(self, client_enabled):
        from fastapi.testclient import TestClient
        client, m = client_enabled

        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c-internal-error", "transaction_type": "sale", "transaction_ref": "req9:ball9",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]

        m.landmark_detector.analyze.side_effect = RuntimeError("model exploded")

        # raise_server_exceptions=False: we want the actual JSON response our
        # handler builds, not Starlette re-raising the original exception.
        with TestClient(m.app, raise_server_exceptions=False) as c:
            resp = c.post("/liveness/verdict", json={
                "correlation_id": "c-internal-error", "session_id": ch["session_id"],
                "transaction_type": "sale", "transaction_ref": "req9:ball9", "frames": frames,
            })

        assert resp.status_code == 500
        data = resp.json()
        assert data["verdict"] == "incomplete"
        assert data["reason"] == "INTERNAL_ERROR"
        assert data["model_version"] == m.LIVENESS_MODEL_VERSION
        assert data["model_version"] != m.MODEL_VERSION
        assert data["frame_consistency_score"] == -1.0
        assert data["session_id"] == ch["session_id"]
        assert data["correlation_id"] == "c-internal-error"
        assert data["transaction_type"] == "sale"
        assert data["transaction_ref"] == "req9:ball9"
        assert "model exploded" not in resp.text


class TestLivenessBlinkEndToEnd:
    """BLINK detection (app/active_challenge.py) is implemented but not in
    the default LIVENESS_CHALLENGE_STEPS_POOL — this proves the wiring works
    end-to-end through the real HTTP contract when a session is explicitly
    given a BLINK step (bypassing the generator, as a future caller with a
    calibrated threshold would be able to do without code changes)."""

    def test_blink_step_reaches_live_when_ear_dips(self, client_enabled):
        from app.face_landmarks import FrameFace
        client, m = client_enabled

        def _face_with_ear(mode: str) -> FrameFace:
            from tests.test_active_challenge import _eye_landmarks
            base = _mock_frame_face(yaw=0.0)
            return FrameFace(
                bbox_xyxy=base.bbox_xyxy, kps=base.kps, pose_pitch=0.0, pose_yaw=0.0,
                pose_roll=0.0, det_score=0.9, n_faces_detected=1,
                landmark_68=_eye_landmarks(mode),
            )

        m.landmark_detector.analyze.side_effect = [
            _face_with_ear("open"), _face_with_ear("closed"), _face_with_ear("open"), _face_with_ear("open"),
        ]

        # Bypass the generator (pool excludes BLINK by design) — directly
        # mint a session with a BLINK step, same as session_store.create()
        # would be called with by a future caller once calibrated.
        session = m.session_store.create(
            steps=["BLINK"], ttl_s=90.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="r:b",
        )
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "live"
        assert data["signals"]["layer2_active_challenge"]["passed"] is True


def _captured_at_frames(count: int, offsets_s: list[float]) -> list[dict]:
    now = datetime.now(timezone.utc)
    assert len(offsets_s) == count
    return [
        {"seq": i, "base64": _make_base64_image(), "captured_at": (now + timedelta(seconds=offsets_s[i])).isoformat()}
        for i in range(count)
    ]


class TestCapturedAtValidation:
    """Фаза 3.2 (CHALLENGE_ENTROPY_SPRINT_v1.md §6.2) — мягкий rollout:
    LIVENESS_CAPTURED_AT_VALIDATION_ENABLED=False (дефолт спринта) логирует
    аномалию, не режет вердикт; =True режет с reason=CAPTURED_AT_INVALID."""

    def _mint_session(self, m):
        return m.session_store.create(
            steps=["TURN_LEFT", "TURN_RIGHT"], ttl_s=90.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="r:b",
        )

    def test_soft_mode_logs_anomaly_without_failing_verdict(self, client_enabled):
        client, m = client_enabled
        assert m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED is False  # sprint default

        yaws = [0.0, 25.0, 0.0, -25.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

        session = self._mint_session(m)
        # seq1's captured_at is EARLIER than seq0's — NOT_MONOTONIC anomaly,
        # would be a hard fail if the flag were on, but it is not.
        frames = _captured_at_frames(4, [1.0, 0.0, 2.0, 3.0])
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"

    def test_hard_mode_fails_verdict_with_captured_at_invalid(self, client_enabled):
        client, m = client_enabled
        m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = True
        try:
            session = self._mint_session(m)
            frames = _captured_at_frames(4, [1.0, 0.0, 2.0, 3.0])  # same NOT_MONOTONIC anomaly as above
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "spoof"
            assert data["reason"] == "CAPTURED_AT_INVALID"
        finally:
            m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = False

    def test_missing_captured_at_is_not_an_anomaly(self, client_enabled):
        """captured_at absent on (some/all) frames must NOT be treated as an
        anomaly in this increment — partner has not confirmed sending it
        stably yet (§6.2)."""
        client, m = client_enabled
        m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = True
        try:
            yaws = [0.0, 25.0, 0.0, -25.0]
            m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]
            session = self._mint_session(m)
            frames = [_frame(i) for i in range(4)]  # captured_at=None on every frame
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            assert resp.json()["reason"] != "CAPTURED_AT_INVALID"
        finally:
            m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = False

    def test_naive_captured_at_treated_as_unparseable_soft_mode(self, client_enabled):
        """HIGH finding (MF DOOM code review, 2026-07-20): a naive ISO string
        (no 'Z'/offset) on EVERY frame must be reported as the UNPARSEABLE
        anomaly (soft: logged only), NOT silently accepted as if it were a
        valid, in-window UTC timestamp — see app/main.py::_parse_captured_at
        and docs/LIVENESS_CONTRACT_v1.md §7.2."""
        client, m = client_enabled
        assert m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED is False  # sprint default
        yaws = [0.0, 25.0, 0.0, -25.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]
        session = self._mint_session(m)
        naive_now = datetime.now()  # deliberately NO tzinfo
        frames = [
            {"seq": i, "base64": _make_base64_image(), "captured_at": (naive_now + timedelta(seconds=i)).isoformat()}
            for i in range(4)
        ]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"  # soft mode: anomaly logged, verdict not cut

    def test_naive_captured_at_fails_verdict_in_hard_mode(self, client_enabled):
        """Same naive-string input as above, but with the flag on — must be
        rejected exactly like a malformed/unparseable string, NOT silently
        accepted as an in-window UTC timestamp just because it happens to
        parse via datetime.fromisoformat."""
        client, m = client_enabled
        m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = True
        try:
            session = self._mint_session(m)
            naive_now = datetime.now()
            frames = [
                {"seq": i, "base64": _make_base64_image(), "captured_at": (naive_now + timedelta(seconds=i)).isoformat()}
                for i in range(4)
            ]
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "spoof"
            assert data["reason"] == "CAPTURED_AT_INVALID"
        finally:
            m.settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED = False


class TestParseCapturedAt:
    """Direct unit coverage of app/main.py::_parse_captured_at — HIGH finding
    (MF DOOM code review, 2026-07-20). See docs/LIVENESS_CONTRACT_v1.md §7.2
    for the contract-level requirement (`captured_at` MUST carry an
    offset)."""

    def test_naive_string_is_rejected(self):
        import app.main as m
        assert m._parse_captured_at("2026-07-17T14:32:00.100") is None

    def test_aware_utc_z_suffix_parses(self):
        import app.main as m
        expected = datetime(2026, 7, 17, 14, 32, 0, 100000, tzinfo=timezone.utc).timestamp()
        assert m._parse_captured_at("2026-07-17T14:32:00.100Z") == pytest.approx(expected)

    def test_aware_plus_offset_normalizes_to_same_epoch_as_utc(self):
        """An aware timestamp carrying a NON-UTC offset (+05:00, this
        service's own server timezone on egaz-02.uz) must normalize to the
        SAME unix epoch as the equivalent UTC 'Z' instant — proving the
        offset is actually respected, not dropped/reinterpreted."""
        import app.main as m
        ts_utc = m._parse_captured_at("2026-07-17T14:32:00Z")
        ts_plus5 = m._parse_captured_at("2026-07-17T19:32:00+05:00")
        assert ts_plus5 == pytest.approx(ts_utc)

    def test_malformed_string_is_rejected(self):
        import app.main as m
        assert m._parse_captured_at("not-a-timestamp") is None

    def test_none_and_empty_string_are_rejected(self):
        import app.main as m
        assert m._parse_captured_at(None) is None
        assert m._parse_captured_at("") is None


class TestTimingWindowValidation:
    """Фаза 3.3 (CHALLENGE_ENTROPY_SPRINT_v1.md §6.3) — тот же мягкий
    паттерн, но для step_windows (Фаза 2). Окна намеренно недостижимые
    (100-200с), реальный интервал между кадрами теста — секунды, поэтому
    нарушение гарантировано без завязки на таймингах исполнения теста."""

    _UNREACHABLE_WINDOWS = [
        {"step": "TURN_LEFT", "min_delay_ms": 100_000, "max_delay_ms": 200_000},
        {"step": "TURN_RIGHT", "min_delay_ms": 100_000, "max_delay_ms": 200_000},
    ]

    def _mint_session(self, m, step_windows):
        return m.session_store.create(
            steps=["TURN_LEFT", "TURN_RIGHT"], ttl_s=90.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="r:b", step_windows=step_windows,
        )

    def test_soft_mode_logs_anomaly_without_failing_verdict(self, client_enabled):
        client, m = client_enabled
        assert m.settings.LIVENESS_TIMING_VALIDATION_ENABLED is False  # sprint default

        yaws = [0.0, 25.0, 0.0, -25.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

        session = self._mint_session(m, self._UNREACHABLE_WINDOWS)
        frames = _captured_at_frames(4, [0.0, 1.0, 2.0, 3.0])
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
            "transaction_ref": "r:b", "frames": frames,
        })
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "live"

    def test_hard_mode_fails_verdict_with_timing_window_violated(self, client_enabled):
        client, m = client_enabled
        m.settings.LIVENESS_TIMING_VALIDATION_ENABLED = True
        try:
            yaws = [0.0, 25.0, 0.0, -25.0]
            m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

            session = self._mint_session(m, self._UNREACHABLE_WINDOWS)
            frames = _captured_at_frames(4, [0.0, 1.0, 2.0, 3.0])
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "spoof"
            assert data["reason"] == "TIMING_WINDOW_VIOLATED"
        finally:
            m.settings.LIVENESS_TIMING_VALIDATION_ENABLED = False

    def test_reachable_window_passes_in_hard_mode(self, client_enabled):
        """Regression guard: a window the client DOES honor must not be
        flagged even with the flag on."""
        client, m = client_enabled
        m.settings.LIVENESS_TIMING_VALIDATION_ENABLED = True
        try:
            yaws = [0.0, 25.0, 0.0, -25.0]
            m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

            windows = [
                {"step": "TURN_LEFT", "min_delay_ms": 0, "max_delay_ms": 10_000},
                {"step": "TURN_RIGHT", "min_delay_ms": 0, "max_delay_ms": 10_000},
            ]
            session = self._mint_session(m, windows)
            frames = _captured_at_frames(4, [0.0, 1.0, 2.0, 3.0])
            resp = client.post("/liveness/verdict", json={
                "correlation_id": "c1", "session_id": session.session_id, "transaction_type": "sale",
                "transaction_ref": "r:b", "frames": frames,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["verdict"] == "live"
        finally:
            m.settings.LIVENESS_TIMING_VALIDATION_ENABLED = False


class TestQualityCertifiedStartupGuard:
    """MEDIUM finding (2PAC code review, 2026-07-20): the `LIVENESS_
    QUALITY_CERTIFIED` guard (app/main.py, right after `LIVENESS_MODEL_
    VERSION`) is module-level startup code, not a function — the only honest
    way to exercise it is a real `importlib.reload(app.main)` with env vars
    set BEFORE the reload (mutating `m.settings` after import does not
    re-run the guard). Each test restores the environment and reloads again
    in a `finally` so later tests in this module see the module in its
    normal (non-certified) state."""

    def test_certified_model_version_mismatch_logs_warning(self, monkeypatch, caplog):
        import app.main as m

        monkeypatch.setenv("LIVENESS_QUALITY_CERTIFIED", "true")
        monkeypatch.setenv("LIVENESS_TARGET_APCER", "0.01")
        monkeypatch.setenv("LIVENESS_TARGET_BPCER", "0.01")
        monkeypatch.setenv("LIVENESS_CERTIFIED_MODEL_VERSION", "stale-certified-build-mismatch")
        try:
            with caplog.at_level(logging.WARNING, logger="app.main"):
                importlib.reload(m)
            assert any(
                "LIVENESS_CERTIFIED_MODEL_VERSION" in rec.message and "does not match" in rec.message
                for rec in caplog.records
            )
        finally:
            monkeypatch.undo()
            importlib.reload(m)

    def test_certified_missing_targets_logs_warning(self, monkeypatch, caplog):
        import app.main as m

        monkeypatch.setenv("LIVENESS_QUALITY_CERTIFIED", "true")
        monkeypatch.delenv("LIVENESS_TARGET_APCER", raising=False)
        monkeypatch.delenv("LIVENESS_TARGET_BPCER", raising=False)
        try:
            with caplog.at_level(logging.WARNING, logger="app.main"):
                importlib.reload(m)
            assert any(
                "LIVENESS_TARGET_APCER" in rec.message and "not" in rec.message
                for rec in caplog.records
            )
        finally:
            monkeypatch.undo()
            importlib.reload(m)

    def test_certified_matching_version_and_targets_no_warning(self, monkeypatch, caplog):
        import app.main as m

        monkeypatch.setenv("LIVENESS_QUALITY_CERTIFIED", "true")
        monkeypatch.setenv("LIVENESS_TARGET_APCER", "0.01")
        monkeypatch.setenv("LIVENESS_TARGET_BPCER", "0.01")
        monkeypatch.setenv("LIVENESS_CERTIFIED_MODEL_VERSION", m.LIVENESS_MODEL_VERSION)
        try:
            with caplog.at_level(logging.WARNING, logger="app.main"):
                importlib.reload(m)
            assert not any(
                "LIVENESS_QUALITY_CERTIFIED" in rec.message for rec in caplog.records
            )
        finally:
            monkeypatch.undo()
            importlib.reload(m)
