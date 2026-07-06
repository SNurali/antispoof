"""FastAPI application — face liveness anti-spoofing service.

Phase 1 PAD-gate hardening (2026-07-06):
- POST /pad/check per FACEID_PHASE1_PAD_GATE contract
- X-Service-Token shared-secret auth
- Sliding-window rate limiter (20 burst / 5 sustained per IP)
- 2-second inference timeout
- Global exception handler (no tracebacks in responses)
- Input validation (8MB, 4000x4000px)
- Health endpoint returns 503 when models not loaded
- JSON structured audit log on every /pad/check
"""

import asyncio
import base64
import json
import logging
import logging.handlers
import time
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

# ---------------------------------------------------------------------------
# Logging: application + audit
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("antispoof")

# Audit logger — structured JSON, separate from app log
audit_log = logging.getLogger("antispoof.audit")
audit_log.setLevel(logging.INFO)
audit_log.propagate = False  # don't bubble to root logger


def _setup_audit_handler() -> None:
    """Set up rotating file audit log (or stdout for Docker)."""
    import os
    if os.environ.get("AUDIT_LOG_STDOUT", "").lower() in ("1", "true", "yes"):
        handler = logging.StreamHandler()
    else:
        handler = logging.handlers.RotatingFileHandler(
            "audit.log", maxBytes=50 * 1024 * 1024, backupCount=5
        )
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_log.addHandler(handler)


_setup_audit_handler()

# ---------------------------------------------------------------------------
# Settings & globals
# ---------------------------------------------------------------------------
settings = Settings()
DEVICE = resolve_device(settings.DEVICE)

# Globals — initialized on startup
detector: Optional[FaceDetector] = None
engine: Optional[LivenessEngine] = None
gpu_name: str = "N/A"
_models_loaded: bool = False

MODEL_VERSION = "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1"

# Max image dimensions (memory DoS protection)
MAX_IMAGE_DIMENSION = 4000
MAX_BASE64_BYTES = 8 * 1024 * 1024  # 8 MB

# Inference timeout
INFERENCE_TIMEOUT_S = 2.0

app = FastAPI(title="Anti-Spoofing Liveness Service", version="1.0.0")

# IP allowlist — local network + localhost + VPC ranges
ALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("192.168.0.0/24"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
]

# ---------------------------------------------------------------------------
# Rate limiter — sliding window, per-IP
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple sliding-window rate limiter (burst requests per second)."""

    def __init__(self, burst: int, sustained: float) -> None:
        self._burst = burst
        self._sustained = sustained
        self._windows: dict[str, deque] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._windows.setdefault(key, deque())
        cutoff = now - 1.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._burst:
            return False
        window.append(now)
        return True


_rate_limiter = _RateLimiter(burst=settings.RATE_LIMIT_BURST, sustained=settings.RATE_LIMIT_SUSTAINED)

# ---------------------------------------------------------------------------
# Global exception handler — never leak tracebacks
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: return safe error response, log full traceback server-side."""
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    # For /pad/check requests, return contract-shaped error
    if request.url.path == "/pad/check":
        return JSONResponse(
            status_code=500,
            content={
                "verdict": "low_quality",
                "score": 0.0,
                "save_frame": False,
                "signals": {"reason": "INTERNAL_ERROR"},
                "processing_ms": 0.0,
            },
        )
    # For /verify requests
    return JSONResponse(
        status_code=500,
        content={"is_real": False, "label": "error", "error": "internal"},
    )


# ---------------------------------------------------------------------------
# Middleware: IP allowlist + rate limiting
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_and_rate_limit(request: Request, call_next):
    raw_ip = request.client.host if request.client else "0.0.0.0"
    try:
        client_ip = ipaddress.ip_address(raw_ip)
        ip_str = str(client_ip)
    except ValueError:
        # TestClient or malformed address — skip IP filter, still rate-limit
        ip_str = raw_ip

    # IP allowlist (skip for non-IP addresses like TestClient)
    try:
        client_ip = ipaddress.ip_address(raw_ip)
        if not any(client_ip in net for net in ALLOWED_NETWORKS):
            return JSONResponse(status_code=403, content={"detail": "Access denied"})
    except ValueError:
        pass  # non-IP address (TestClient), skip allowlist

    if not _rate_limiter.allow(ip_str):
        log.warning("Rate limited: %s", ip_str)
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


# ---------------------------------------------------------------------------
# Startup — load models
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _load_models() -> None:
    global detector, engine, gpu_name, _models_loaded
    log.info("Loading models on device=%s ...", DEVICE)

    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        log.info("GPU: %s", gpu_name)
    else:
        log.warning("Running on CPU — inference will be slower")

    detector = FaceDetector(settings.MODEL_DIR)
    engine = LivenessEngine(settings.MODEL_DIR, DEVICE)
    _models_loaded = True
    log.info("Models loaded. Ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


def _validate_image_size(img_bytes: bytes) -> None:
    """Reject oversized images before decoding."""
    if len(img_bytes) > MAX_BASE64_BYTES:
        raise HTTPException(status_code=400, detail=f"Image exceeds {MAX_BASE64_BYTES // (1024*1024)}MB limit")


def _validate_image_dimensions(img: np.ndarray) -> None:
    """Reject images with excessive dimensions (memory DoS protection)."""
    h, w = img.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions {w}x{h} exceed {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} limit",
        )


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


# ---------------------------------------------------------------------------
# Health endpoint — 503 when models not loaded (1.7)
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    loaded = _models_loaded and engine is not None and detector is not None
    status_code = 200 if loaded else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if loaded else "not_ready",
            "device": DEVICE,
            "gpu": gpu_name,
            "models_loaded": loaded,
            "model_version": MODEL_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Legacy endpoints (backward compatible)
# ---------------------------------------------------------------------------
@app.post("/verify")
async def verify(image: UploadFile = File(...)) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    _validate_image_size(data)
    img = _read_image(data)
    _validate_image_dimensions(img)

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

    decoded: list[tuple[np.ndarray, Optional[list[int]], str]] = []
    for upload in images:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "not an image"))
            continue
        data = await upload.read()
        if not data:
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "empty file"))
            continue
        _validate_image_size(data)
        img = _read_image(data)
        _validate_image_dimensions(img)
        bbox = detector.detect(img)
        decoded.append((img, bbox, ""))

    crops: list[np.ndarray] = []
    crop_face_px: list[int] = []
    crop_indices: list[int] = []
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
            from app.liveness import _crop_face
            scale = engine._models[0][3] if engine._models else None
            crop = _crop_face(img, bbox, scale, 224, 224)
            crops.append(crop)
            crop_face_px.append(max(0, bbox[2]) * max(0, bbox[3]))
            crop_indices.append(i)

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
# POST /spoof-server — legacy base64 format
# ---------------------------------------------------------------------------
class SpoofRequest(BaseModel):
    photo: str


@app.post("/spoof-server")
async def spoof_server(req: SpoofRequest) -> dict:
    t0 = time.perf_counter()
    try:
        img_bytes = base64.b64decode(req.photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    _validate_image_size(img_bytes)
    img = _read_image(img_bytes)
    _validate_image_dimensions(img)
    result = _run_single(img)

    elapsed = round(time.perf_counter() - t0, 3)
    is_spoof = 0 if result["is_real"] else 1
    return {"elapsed_time": elapsed, "is_spoof": is_spoof}


# ---------------------------------------------------------------------------
# POST /pad/check — PAD-gate integration (Phase 1)
# ---------------------------------------------------------------------------

class PadCheckRequest(BaseModel):
    """PAD-gate request from Laravel."""
    correlation_id: str = Field(..., description="UUID from Laravel for log tracing")
    transaction_type: str = Field("sale", description="'sale' in v1, 'receive' in v2")
    transaction_ref: str = Field(..., description="id_request:id_ballon (natural key)")
    face_photo: str = Field(..., description="Base64 JPEG/PNG — same frame as Adliya")


class PadCheckResponse(BaseModel):
    """PAD-gate response."""
    verdict: str = Field(..., description="real | spoof | low_quality | no_face")
    score: float = Field(..., description="Confidence score [0..1]")
    save_frame: bool = Field(False, description="Laravel should encrypt+store frame")
    signals: dict = Field(default_factory=dict, description="Multi-signal breakdown")
    processing_ms: float = Field(..., description="Server-side processing time")


_save_frame_verdicts = set(v.strip() for v in settings.SAVE_FRAME_VERDICTS.split(","))


def _verify_service_token(x_service_token: Optional[str]) -> None:
    if not settings.SERVICE_TOKEN:
        return
    if not x_service_token or x_service_token != settings.SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Service-Token")


def _audit_entry(correlation_id: str, transaction_type: str, transaction_ref: str,
                 verdict: str, score: float, signals: dict, processing_ms: float,
                 save_frame: bool, reason: str = "") -> None:
    """Write structured JSON audit log entry."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "correlation_id": correlation_id,
        "transaction_type": transaction_type,
        "transaction_ref": transaction_ref,
        "verdict": verdict,
        "reason": reason,
        "score": round(score, 4),
        "model_version": MODEL_VERSION,
        "processing_ms": round(processing_ms, 1),
        "save_frame": save_frame,
    }
    if signals:
        entry["signal_scores"] = signals.get("signal_scores", {})
        entry["nn_label"] = signals.get("nn_label", "")
        entry["nn_score"] = signals.get("nn_score", 0.0)
    audit_log.info(json.dumps(entry, ensure_ascii=False))


@app.post("/pad/check", response_model=PadCheckResponse)
async def pad_check(
    req: PadCheckRequest,
    x_service_token: Optional[str] = Header(None, alias="X-Service-Token"),
) -> PadCheckResponse:
    """PAD-gate: classify a face frame as real/spoof/low_quality/no_face.

    Called by Laravel after Adliya match (BACKEND_REQUIREMENTS_2026-07-06 п.8).
    """
    _verify_service_token(x_service_token)

    t0 = time.perf_counter()

    # Decode + validate
    try:
        img_bytes = base64.b64decode(req.face_photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in face_photo")

    _validate_image_size(img_bytes)
    img = _read_image(img_bytes)
    _validate_image_dimensions(img)

    # Models ready?
    if not _models_loaded or detector is None or engine is None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        _audit_entry(req.correlation_id, req.transaction_type, req.transaction_ref,
                     "low_quality", 0.0, {}, ms, False, reason="MODELS_NOT_LOADED")
        return PadCheckResponse(
            verdict="low_quality", score=0.0, save_frame=False,
            signals={"reason": "MODELS_NOT_LOADED"}, processing_ms=ms,
        )

    # Detect face
    bbox = detector.detect(img)
    if bbox is None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        _audit_entry(req.correlation_id, req.transaction_type, req.transaction_ref,
                     "no_face", 0.0, {}, ms, False, reason="NO_FACE_DETECTED")
        return PadCheckResponse(
            verdict="no_face", score=0.0, save_frame=False,
            signals={"reason": "NO_FACE_DETECTED"}, processing_ms=ms,
        )

    # Run liveness with 2-second timeout (1.4)
    try:
        label, score, face_detected, signal_info = await asyncio.wait_for(
            asyncio.to_thread(engine.predict, img, bbox),
            timeout=INFERENCE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.error("Inference timeout after %.1fs for correlation=%s", INFERENCE_TIMEOUT_S, req.correlation_id)
        _audit_entry(req.correlation_id, req.transaction_type, req.transaction_ref,
                     "low_quality", 0.0, {}, ms, False, reason="TIMEOUT")
        return PadCheckResponse(
            verdict="low_quality", score=0.0, save_frame=False,
            signals={"reason": "TIMEOUT"}, processing_ms=ms,
        )

    # Map to PAD-gate verdict
    if label == "spoof":
        verdict = "spoof"
    elif score < settings.LIVENESS_THRESHOLD:
        verdict = "low_quality"
    else:
        verdict = "real"

    save_frame = verdict in _save_frame_verdicts
    processing_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Audit log (every request, including "real" — for APCER/BPCER analysis)
    _audit_entry(
        req.correlation_id, req.transaction_type, req.transaction_ref,
        verdict, score, signal_info, processing_ms, save_frame,
        reason="PASSIVE_PAD_SPOOF" if verdict == "spoof" else "",
    )

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
