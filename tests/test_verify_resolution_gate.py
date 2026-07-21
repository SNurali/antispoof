"""Tests for the Layer 0e resolution/weight gate (app/resolution_check.py)
wired into POST /verify and POST /verify_batch — see app/main.py::
_run_single (shared by /verify and /spoof-server) and
app/main.py::verify_batch's own inlined per-frame loop.

/pad/check's own resolution-gate coverage lives in tests/test_pad_check.py
(TestPadCheckResolutionLayer) — this file only covers the two multipart
legacy endpoints that reuse different code paths.
"""

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


def _make_test_image_bytes(width: int = 200, height: int = 200) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 3, (200, 180, 160), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_client_shaped_image_bytes() -> bytes:
    """~960x1280 (3:4) — the client's expected real capture shape, see
    app/resolution_check.py's module docstring."""
    rng = np.random.default_rng(11)
    img = np.zeros((1280, 960, 3), dtype=np.uint8)
    cv2.circle(img, (480, 640), 300, (200, 180, 160), -1)
    noise = rng.integers(-30, 30, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


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


@pytest.fixture
def client():
    import app.main as m

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [50, 50, 100, 100]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = ("real", 0.95, True, {"nn_label": "real", "nn_score": 0.95})
    mock_engine.predict_batch.return_value = [("real", 0.95, True, {"nn_label": "real", "nn_score": 0.95})]
    # verify_batch's own inlined path reads `engine._models[0][3]` as a scale
    # hint for `_crop_face` when truthy — an empty list here (falsy) makes it
    # take the `else None` branch instead of indexing into a MagicMock.
    mock_engine._models = []

    m.detector = mock_detector
    m.engine = mock_engine
    m._models_loaded = True

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


class TestVerifyResolutionLayer:
    def test_disabled_by_default_small_frame_passes_through(self, client):
        resp = client.post(
            "/verify",
            files={"image": ("photo.jpg", _make_test_image_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["is_real"] is True

    def test_enabled_small_frame_rejected_as_low_quality(self, client):
        import app.main as m
        m.settings.RESOLUTION_CHECK_ENABLED = True
        try:
            resp = client.post(
                "/verify",
                files={"image": ("photo.jpg", _make_test_image_bytes(), "image/jpeg")},
            )
        finally:
            m.settings.RESOLUTION_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["is_real"] is False
        assert data["label"] == "low_quality"
        assert data["face_detected"] is False
        assert "resolution_check" in data["signals"]

    def test_enabled_client_shaped_frame_passes_through(self, client):
        import app.main as m
        m.settings.RESOLUTION_CHECK_ENABLED = True
        try:
            resp = client.post(
                "/verify",
                files={"image": ("photo.jpg", _make_client_shaped_image_bytes(), "image/jpeg")},
            )
        finally:
            m.settings.RESOLUTION_CHECK_ENABLED = False

        assert resp.status_code == 200
        assert resp.json()["is_real"] is True


class TestVerifyBatchResolutionLayer:
    def test_disabled_by_default_small_frame_passes_through(self, client):
        resp = client.post(
            "/verify_batch",
            files=[("images", ("photo.jpg", _make_test_image_bytes(), "image/jpeg"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["is_real"] is True

    def test_enabled_small_frame_rejected_without_face_detection(self, client):
        """Bbox-independent gate — must reject the small frame in the batch
        WITHOUT ever calling detector.detect() for it (see app/main.py::
        verify_batch's first loop, which now runs the resolution gate before
        detection)."""
        import app.main as m
        m.settings.RESOLUTION_CHECK_ENABLED = True
        try:
            resp = client.post(
                "/verify_batch",
                files=[("images", ("photo.jpg", _make_test_image_bytes(), "image/jpeg"))],
            )
        finally:
            m.settings.RESOLUTION_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["is_real"] is False
        assert data["results"][0]["label"] == "low_quality"
        m.detector.detect.assert_not_called()

    def test_enabled_mixed_batch_only_small_frame_rejected(self, client):
        """A batch with one small and one client-shaped frame — only the
        small one is rejected by this layer, the other reaches passive-PAD."""
        import app.main as m
        m.settings.RESOLUTION_CHECK_ENABLED = True
        try:
            resp = client.post(
                "/verify_batch",
                files=[
                    ("images", ("small.jpg", _make_test_image_bytes(), "image/jpeg")),
                    ("images", ("big.jpg", _make_client_shaped_image_bytes(), "image/jpeg")),
                ],
            )
        finally:
            m.settings.RESOLUTION_CHECK_ENABLED = False

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["label"] == "low_quality"
        assert data["results"][1]["is_real"] is True
