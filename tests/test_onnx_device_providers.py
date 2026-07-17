"""Tests for the DEVICE=auto|cpu|cuda wiring added to the onnxruntime/
insightface consumers of the active-liveness identity pipeline
(app/adaface.py::AdaFaceEmbedder, app/face_landmarks.py::LandmarkDetector,
app/config.py::onnx_providers — 2026-07-17 GPU dual-mode work).

Covers exactly the two things that matter for a CPU-only prod host
(egaz-02.uz today) NOT to break:
  1. providers are chosen correctly from the resolved DEVICE value
     (onnx_providers()).
  2. a CUDA init attempt that fails at RUNTIME (onnxruntime-gpu installed but
     no working CUDA/cuDNN — the exact failure mode observed on this repo's
     own dev box without LD_LIBRARY_PATH set, see requirements.txt) falls
     back to CPU-only instead of raising, on BOTH consumers.

No real ONNX weights or insightface buffalo_l download needed — onnxruntime.
InferenceSession / insightface.app.FaceAnalysis are mocked, same pattern
tests/test_liveness_endpoints.py already uses for landmark_detector/
adaface_embedder at the endpoint layer.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DEVICE", "cpu")

from app.config import onnx_providers


class TestOnnxProviders:
    def test_cpu_device_always_cpu_only(self):
        assert onnx_providers("cpu") == ["CPUExecutionProvider"]

    def test_cuda_device_with_cuda_provider_available(self):
        with patch("onnxruntime.get_available_providers", return_value=[
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]):
            assert onnx_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_cuda_device_without_cuda_provider_falls_back_to_cpu(self):
        """The onnxruntime-gpu package is not installed — plain CPU wheel
        only reports CPUExecutionProvider."""
        with patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]):
            assert onnx_providers("cuda") == ["CPUExecutionProvider"]

    def test_cuda_device_onnxruntime_not_importable_falls_back_to_cpu(self):
        """Defensive path: onnxruntime itself can't be imported at all."""
        import sys

        real_ort = sys.modules.get("onnxruntime")
        sys.modules["onnxruntime"] = None  # forces ImportError on `import onnxruntime`
        try:
            assert onnx_providers("cuda") == ["CPUExecutionProvider"]
        finally:
            if real_ort is not None:
                sys.modules["onnxruntime"] = real_ort
            else:
                del sys.modules["onnxruntime"]


class TestAdaFaceEmbedderDevice:
    def _fake_session(self, providers):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.get_providers.return_value = providers
        return session

    def test_cpu_requests_cpu_provider_only(self, tmp_path):
        onnx_path = tmp_path / "fake.onnx"
        onnx_path.write_bytes(b"stub")
        cpu_session = self._fake_session(["CPUExecutionProvider"])

        with patch("onnxruntime.InferenceSession", return_value=cpu_session) as ctor:
            from app.adaface import AdaFaceEmbedder
            embedder = AdaFaceEmbedder(onnx_path, device="cpu")

        assert ctor.call_count == 1
        assert ctor.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
        assert embedder._session is cpu_session

    def test_cuda_available_requests_cuda_provider_first(self, tmp_path):
        onnx_path = tmp_path / "fake.onnx"
        onnx_path.write_bytes(b"stub")
        gpu_session = self._fake_session(["CUDAExecutionProvider", "CPUExecutionProvider"])

        with patch("onnxruntime.get_available_providers", return_value=[
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]), patch("onnxruntime.InferenceSession", return_value=gpu_session) as ctor:
            from app.adaface import AdaFaceEmbedder
            embedder = AdaFaceEmbedder(onnx_path, device="cuda")

        assert ctor.call_args.kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert embedder._session is gpu_session

    def test_cuda_runtime_init_failure_falls_back_to_cpu_without_raising(self, tmp_path):
        """The observed real failure mode: onnxruntime-gpu is installed
        (so onnx_providers() offers CUDAExecutionProvider) but the CUDA/cuDNN
        shared libs are missing at runtime — session creation with that
        provider list must not crash the whole service."""
        onnx_path = tmp_path / "fake.onnx"
        onnx_path.write_bytes(b"stub")
        cpu_session = self._fake_session(["CPUExecutionProvider"])
        calls = []

        def fake_ctor(*args, **kwargs):
            calls.append(kwargs.get("providers"))
            if kwargs.get("providers", [None])[0] == "CUDAExecutionProvider":
                raise RuntimeError("libcublasLt.so.12: cannot open shared object file")
            return cpu_session

        with patch("onnxruntime.get_available_providers", return_value=[
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]), patch("onnxruntime.InferenceSession", side_effect=fake_ctor):
            from app.adaface import AdaFaceEmbedder
            embedder = AdaFaceEmbedder(onnx_path, device="cuda")

        assert len(calls) == 2
        assert calls[0] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert calls[1] == ["CPUExecutionProvider"]
        assert embedder._session is cpu_session

    def test_missing_weights_still_raises_filenotfound(self, tmp_path):
        missing = tmp_path / "does_not_exist.onnx"
        from app.adaface import AdaFaceEmbedder

        with pytest.raises(FileNotFoundError):
            AdaFaceEmbedder(missing, device="cpu")


class TestLandmarkDetectorDevice:
    def test_cpu_uses_ctx_id_minus_one(self):
        instance = MagicMock()

        with patch("insightface.app.FaceAnalysis", return_value=instance) as ctor:
            from app.face_landmarks import LandmarkDetector
            LandmarkDetector(det_size=320, device="cpu")

        assert ctor.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
        assert instance.prepare.call_args.kwargs["ctx_id"] == -1

    def test_cuda_available_uses_ctx_id_zero(self):
        instance = MagicMock()

        with patch("onnxruntime.get_available_providers", return_value=[
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]), patch("insightface.app.FaceAnalysis", return_value=instance) as ctor:
            from app.face_landmarks import LandmarkDetector
            LandmarkDetector(det_size=320, device="cuda")

        assert ctor.call_args.kwargs["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert instance.prepare.call_args.kwargs["ctx_id"] == 0

    def test_cuda_runtime_init_failure_falls_back_to_cpu_without_raising(self):
        """Same real failure mode as AdaFaceEmbedder: providers look
        available per onnxruntime.get_available_providers(), but the actual
        session/model construction fails at runtime (missing CUDA/cuDNN
        libs) — must fall back, not crash /liveness/challenge startup."""
        cpu_instance = MagicMock()
        calls = []

        def fake_ctor(*args, **kwargs):
            calls.append(kwargs.get("providers"))
            if kwargs.get("providers", [None])[0] == "CUDAExecutionProvider":
                raise RuntimeError("cuDNN 9.* required, found none")
            return cpu_instance

        with patch("onnxruntime.get_available_providers", return_value=[
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]), patch("insightface.app.FaceAnalysis", side_effect=fake_ctor):
            from app.face_landmarks import LandmarkDetector
            LandmarkDetector(det_size=320, device="cuda")

        assert len(calls) == 2
        assert calls[0] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert calls[1] == ["CPUExecutionProvider"]
        cpu_instance.prepare.assert_called_once()
        assert cpu_instance.prepare.call_args.kwargs["ctx_id"] == -1
