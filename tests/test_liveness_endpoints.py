"""Tests for POST /liveness/challenge and POST /liveness/verdict — endpoint
wiring, auth, session lifecycle, and fusion-order behavior. Landmark
detector / AdaFace embedder are MOCKED (same pattern as test_pad_check.py's
detector/engine mocks) so these tests do not require the real 260MB ONNX
weight file or insightface's buffalo_l bundle to be present."""
import base64
import os
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
        rejected, even with a matching session_id."""
        client, m = client_enabled
        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "req1:ball1",
        }).json()
        frames = [_frame(i) for i in range(m.settings.LIVENESS_MIN_FRAMES)]
        resp = client.post("/liveness/verdict", json={
            "correlation_id": "DIFFERENT_CORRELATION_ID", "session_id": ch["session_id"], "transaction_type": "sale",
            "transaction_ref": "req1:ball1", "frames": frames,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"] == "incomplete"
        assert data["reason"] == "SESSION_CORRELATION_MISMATCH"

    def test_transaction_ref_mismatch_alone_is_not_rejected(self, client_enabled):
        """transaction_ref is passthrough-only for /liveness/* (the sale
        natural key can legitimately not be final yet when the challenge is
        issued) — a mismatch on transaction_ref ALONE, with a matching
        correlation_id, must NOT be treated as a binding violation. Uses a
        yaw sequence that also satisfies the active-challenge steps so the
        request can reach `live`, proving the mismatch truly did not block
        anything downstream."""
        client, m = client_enabled
        yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

        ch = client.post("/liveness/challenge", json={
            "correlation_id": "c1", "transaction_type": "sale", "transaction_ref": "req1:ball1",
        }).json()
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
        yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
        m.landmark_detector.analyze.side_effect = [_mock_frame_face(yaw=y) for y in yaws]

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
            yaws = [0.0, 25.0, 0.0, -25.0, 0.0, 0.0]
            m.landmark_detector.analyze.side_effect = [self._document_like_face(yaw=y) for y in yaws]

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
            assert data["reason"] != "DOCUMENT_PHOTO"
            assert data["verdict"] == "live"
        finally:
            m.settings.GEOMETRY_CHECK_ENABLED = True


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
