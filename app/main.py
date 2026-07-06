"""FastAPI application — face liveness anti-spoofing service."""

import base64
import io
import time
import logging
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

import ipaddress

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, resolve_device
from app.face_detect import FaceDetector
from app.liveness import LivenessEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("antispoof")

settings = Settings()
DEVICE = resolve_device(settings.DEVICE)

# Globals — initialized on startup
detector: Optional[FaceDetector] = None
engine: Optional[LivenessEngine] = None
gpu_name: str = "N/A"

app = FastAPI(title="Anti-Spoofing Liveness Service", version="1.0.0")

# IP allowlist — only local network + localhost
ALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # localhost
    ipaddress.ip_network("192.168.0.0/24"),    # local LAN
    ipaddress.ip_network("10.0.0.0/8"),        # VPC private range
    ipaddress.ip_network("172.16.0.0/12"),     # Docker/VPC private range
]

# ---------------------------------------------------------------------------
# Rate limiter — sliding window, per-IP (token bucket with burst + sustained)
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, burst: int, sustained: float) -> None:
        self._burst = burst
        self._sustained = sustained
        self._windows: dict[str, deque] = {}  # ip -> deque of timestamps

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._windows.setdefault(key, deque())
        # Purge entries older than 1 second
        cutoff = now - 1.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._burst:
            return False
        window.append(now)
        return True


_rate_limiter = _RateLimiter(burst=settings.RATE_LIMIT_BURST, sustained=settings.RATE_LIMIT_SUSTAINED)


@app.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    """IP allowlist + rate limiting + shared-secret auth for /pad/check."""
    client_ip = ipaddress.ip_address(request.client.host if request.client else "0.0.0.0")
    if not any(client_ip in net for net in ALLOWED_NETWORKS):
        return JSONResponse(status_code=403, content={"detail": "Access denied"})

    # Rate limit
    if not _rate_limiter.allow(str(client_ip)):
        log.warning("Rate limited: %s", client_ip)
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    return await call_next(request)


# Serve the test UI
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(content=index_html.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Place index.html in app/static/</h1>", status_code=404)


@app.on_event("startup")
def _load_models() -> None:
    global detector, engine, gpu_name
    log.info("Loading models on device=%s ...", DEVICE)

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info("GPU: %s", gpu_name)
    else:
        log.warning("Running on CPU — inference will be slower")

    detector = FaceDetector(settings.MODEL_DIR)
    engine = LivenessEngine(settings.MODEL_DIR, DEVICE)
    log.info("Models loaded. Ready.")


def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _run_single(image_bgr: np.ndarray) -> dict:
    """Detect face + predict liveness for a single image."""
    bbox = detector.detect(image_bgr)
    if bbox is None:
        return {
            "is_real": False,
            "label": "no_face",
            "score": 0.0,
            "threshold": settings.LIVENESS_THRESHOLD,
            "face_detected": False,
        }

    label, score, face_detected, signal_info = engine.predict(image_bgr, bbox)
    is_real = label == "real" and score >= settings.LIVENESS_THRESHOLD
    return {
        "is_real": is_real,
        "label": label,
        "score": round(score, 4),
        "threshold": settings.LIVENESS_THRESHOLD,
        "face_detected": face_detected,
        "signals": signal_info,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "device": DEVICE,
        "gpu": gpu_name,
        "models_loaded": engine is not None,
    }


@app.post("/verify")
async def verify(image: UploadFile = File(...)) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    img = _read_image(data)
    t0 = time.perf_counter()
    result = _run_single(img)
    result["processing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


@app.post("/verify_batch")
async def verify_batch(images: list[UploadFile] = File(...)) -> dict:
    if len(images) > settings.MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Max batch size is {settings.MAX_BATCH}, got {len(images)}",
        )

    t0 = time.perf_counter()

    # Phase 1: decode images and detect faces (CPU)
    decoded: list[tuple[np.ndarray, Optional[list[int]], str]] = []
    for upload in images:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "not an image"))
            continue
        data = await upload.read()
        if not data:
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "empty file"))
            continue
        img = _read_image(data)
        bbox = detector.detect(img)
        decoded.append((img, bbox, ""))

    # Phase 2: crop faces and batch through GPU in ONE forward pass
    crops: list[np.ndarray] = []
    crop_face_px: list[int] = []
    crop_indices: list[int] = []  # map crop index → decoded index
    results: list[dict] = [{}] * len(decoded)

    for i, (img, bbox, err) in enumerate(decoded):
        if err:
            results[i] = {"is_real": False, "label": "no_face", "score": 0.0,
                           "threshold": settings.LIVENESS_THRESHOLD,
                           "face_detected": False, "error": err}
        elif bbox is None:
            results[i] = {"is_real": False, "label": "no_face", "score": 0.0,
                           "threshold": settings.LIVENESS_THRESHOLD,
                           "face_detected": False}
        else:
            # Pre-crop on CPU for the batch at native resolution (224px) so the
            # recapture signal keeps full facial detail; NN resizes down itself.
            from app.liveness import _crop_face
            scale = engine._models[0][3] if engine._models else None
            crop = _crop_face(img, bbox, scale, 224, 224)
            crops.append(crop)
            crop_face_px.append(max(0, bbox[2]) * max(0, bbox[3]))
            crop_indices.append(i)

    # Single GPU batch inference
    if crops:
        batch_results = engine.predict_batch(crops, crop_face_px)
        for j, idx in enumerate(crop_indices):
            label, score, detected, signal_info = batch_results[j]
            is_real = label == "real" and score >= settings.LIVENESS_THRESHOLD
            results[idx] = {
                "is_real": is_real,
                "label": label,
                "score": round(score, 4),
                "threshold": settings.LIVENESS_THRESHOLD,
                "face_detected": detected,
                "signals": signal_info,
            }

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"results": results, "total_ms": total_ms, "count": len(results)}


# ---------------------------------------------------------------------------
# POST /spoof-server  —  формат для интеграции с внешним сервером
# ---------------------------------------------------------------------------
class SpoofRequest(BaseModel):
    photo: str  # base64-encoded image


@app.post("/spoof-server")
async def spoof_server(req: SpoofRequest) -> dict:
    """Проверка liveness по base64. Формат совместим с внешним сервером."""
    t0 = time.perf_counter()

    try:
        img_bytes = base64.b64decode(req.photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    img = _read_image(img_bytes)
    result = _run_single(img)

    elapsed = round(time.perf_counter() - t0, 3)
    is_spoof = 0 if result["is_real"] else 1

    return {"elapsed_time": elapsed, "is_spoof": is_spoof}


# ---------------------------------------------------------------------------
# POST /pad/check — PAD-gate integration contract (FACEID_PHASE1_PAD_GATE.md)
# Called by Laravel AFTER Adliya match, BEFORE business logic commit.
# Auth: X-Service-Token shared secret.
# ---------------------------------------------------------------------------

class PadCheckRequest(BaseModel):
    """PAD-gate request from Laravel (BACKEND_REQUIREMENTS_2026-07-06 п.8)."""
    correlation_id: str = Field(..., description="UUID from Laravel for end-to-end log tracing")
    transaction_type: str = Field("sale", description="Only 'sale' in v1; 'receive' in v2")
    transaction_ref: str = Field(..., description="id_request:id_ballon (natural key)")
    face_photo: str = Field(..., description="Base64 JPEG/PNG — same frame sent to Adliya")


class PadCheckResponse(BaseModel):
    """PAD-gate response."""
    verdict: str = Field(..., description="real | spoof | low_quality | no_face")
    score: float = Field(..., description="Confidence score [0..1]")
    save_frame: bool = Field(False, description="True if Laravel should encrypt+store this frame")
    signals: dict = Field(default_factory=dict, description="Multi-signal breakdown for audit")
    processing_ms: float = Field(..., description="Server-side processing time")


_save_frame_verdicts = set(v.strip() for v in settings.SAVE_FRAME_VERDICTS.split(","))


def _verify_service_token(x_service_token: Optional[str]) -> None:
    """Validate X-Service-Token if SERVICE_TOKEN is configured."""
    if not settings.SERVICE_TOKEN:
        return  # auth disabled (dev mode)
    if not x_service_token or x_service_token != settings.SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Service-Token")


@app.post("/pad/check", response_model=PadCheckResponse)
async def pad_check(
    req: PadCheckRequest,
    x_service_token: Optional[str] = Header(None, alias="X-Service-Token"),
) -> PadCheckResponse:
    """PAD-gate: classify a face frame as real/spoof/low_quality/no_face.

    Contract per FACEID_PHASE1_PAD_GATE.md + BACKEND_REQUIREMENTS_2026-07-06.
    Called by Laravel after Adliya match. Verdict drives:
    - real → business logic continues
    - spoof → FACE_LIVENESS_FAILED error, frame saved if save_frame=true
    - low_quality → UX "retake photo" (not an incident)
    - no_face → UX "no face detected"
    """
    _verify_service_token(x_service_token)

    t0 = time.perf_counter()

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.face_photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in face_photo")

    img = _read_image(img_bytes)

    # Detect face
    bbox = detector.detect(img) if detector else None
    if bbox is None:
        return PadCheckResponse(
            verdict="no_face",
            score=0.0,
            save_frame=False,
            signals={"reason": "no_face_detected"},
            processing_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # Run liveness
    label, score, face_detected, signal_info = engine.predict(img, bbox)

    # Map to PAD-gate verdict
    if label == "spoof":
        verdict = "spoof"
    elif score < settings.LIVENESS_THRESHOLD:
        verdict = "low_quality"
    else:
        verdict = "real"

    save_frame = verdict in _save_frame_verdicts
    processing_ms = round((time.perf_counter() - t0) * 1000, 1)

    log.info(
        "PAD check: correlation=%s verdict=%s score=%.3f ms=%.1f txn=%s",
        req.correlation_id, verdict, score, processing_ms, req.transaction_ref,
    )

    return PadCheckResponse(
        verdict=verdict,
        score=round(score, 4),
        save_frame=save_frame,
        signals=signal_info,
        processing_ms=processing_ms,
    )
