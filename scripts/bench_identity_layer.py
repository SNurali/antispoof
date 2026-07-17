#!/usr/bin/env python3
"""Measure Layer 0/2/3 (active-liveness identity) latency: CPU vs GPU.

This is the ONNX/insightface identity pipeline that app/config.py's
ADAFACE_ONNX_PATH docstring measured at 342-524ms/frame CPU-only on the
i5-11400 prod-like box — this script benchmarks the SAME two model calls
(SCRFD+landmark_3d_68 detection via app/face_landmarks.py::LandmarkDetector,
AdaFace IR-101 embedding via app/adaface.py::AdaFaceEmbedder) on whatever
hardware it is run on, CPU-only vs CUDA (if onnxruntime-gpu + a CUDA-capable
GPU are both actually available — see requirements.txt GPU section for the
install steps and REQUIRED LD_LIBRARY_PATH, without which CUDAExecutionProvider
silently fails to load and this script's "GPU" column will just report the same
CPU numbers).

Does NOT touch app/liveness.py's torch-based MiniFASNet ensemble — that is
scripts/bench_cpu.py's job (Layer 1 passive-PAD, already dual-mode via
app/config.py::resolve_device before this increment).

Usage:
    python scripts/bench_identity_layer.py [image_path]

Default image: docs/photo_2026-07-06_10-40-40.jpg (a real bonafide selfie
already used by scripts/bench_cpu.py).
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.config import Settings, onnx_providers

DEFAULT_IMAGE = "/home/mrnurali/E-GAZ/docs/photo_2026-07-06_10-40-40.jpg"


def _bench_scrfd(image_bgr: np.ndarray, device: str, warmup: int = 3, n: int = 20):
    from app.face_landmarks import LandmarkDetector

    settings = Settings()
    det = LandmarkDetector(det_size=settings.LIVENESS_DET_SIZE, device=device)
    active = det._app.det_model.session.get_providers()

    for _ in range(warmup):
        det.analyze(image_bgr)

    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        face = det.analyze(image_bgr)
        samples.append((time.perf_counter() - t0) * 1000)

    return active, samples, face


def _bench_adaface(image_bgr: np.ndarray, kps: np.ndarray, device: str, warmup: int = 3, n: int = 20):
    from app.adaface import AdaFaceEmbedder
    from app.face_landmarks import LandmarkDetector

    settings = Settings()
    embedder = AdaFaceEmbedder(settings.ADAFACE_ONNX_PATH, device=device)
    active = embedder._session.get_providers()

    aligned = LandmarkDetector.align_112(image_bgr, kps)

    for _ in range(warmup):
        embedder.embed_aligned(aligned)

    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        embedder.embed_aligned(aligned)
        samples.append((time.perf_counter() - t0) * 1000)

    return active, samples


def _report(label: str, active_providers: list[str], samples: list[float]) -> None:
    mean = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
    print(
        f"  {label:28s} active_providers={active_providers}\n"
        f"    mean={mean:7.2f}ms  p50={p50:7.2f}ms  p95={p95:7.2f}ms  n={len(samples)}"
    )


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not decode: {image_path}")
        sys.exit(1)
    print(f"Image: {image_path}  shape={img.shape}")

    available = []
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception as exc:
        print(f"onnxruntime not importable: {exc}")
        sys.exit(1)
    print(f"onnxruntime.get_available_providers(): {available}")
    gpu_possible = "CUDAExecutionProvider" in available
    print(f"CUDAExecutionProvider available: {gpu_possible}\n")

    devices = ["cpu", "cuda"] if gpu_possible else ["cpu"]
    if not gpu_possible:
        print("(No CUDAExecutionProvider — see requirements.txt GPU section. "
              "Reporting CPU-only.)\n")

    print("=== SCRFD + landmark_3d_68 (app/face_landmarks.py::LandmarkDetector) ===")
    for device in devices:
        active, samples, face = _bench_scrfd(img, device)
        _report(f"device={device}", active, samples)
        if face is None:
            print("    WARNING: no face detected in this image")

    print("\n=== AdaFace IR-101 (app/adaface.py::AdaFaceEmbedder) ===")
    # Reuse a real detection for realistic aligned-crop input.
    from app.face_landmarks import LandmarkDetector as _LD
    settings = Settings()
    ld = _LD(det_size=settings.LIVENESS_DET_SIZE, device="cpu")
    face = ld.analyze(img)
    if face is None:
        print("No face detected — cannot benchmark AdaFace on aligned crop.")
        sys.exit(1)

    for device in devices:
        active, samples = _bench_adaface(img, face.kps, device)
        _report(f"device={device}", active, samples)

    print(
        "\nNote: 'device=cuda' here means DEVICE=cuda was REQUESTED — check "
        "'active_providers' in each block above to confirm CUDAExecutionProvider "
        "actually initialized (a version mismatch falls back to CPU silently, "
        "see app/adaface.py / app/face_landmarks.py docstrings)."
    )


if __name__ == "__main__":
    main()
