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
from typing import Literal, Optional

from pydantic import BaseModel, Field

import ipaddress

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, resolve_device
from app.document_check import DocumentPhotoChecker
from app.face_detect import FaceDetector
from app.geometry_check import GeometryCheckResult, check_face_geometry
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

if not settings.SERVICE_TOKEN:
    log.warning(
        "SERVICE_TOKEN is empty — /pad/check auth is DISABLED (dev mode). "
        "Do NOT run in production without a SERVICE_TOKEN set."
    )

# Globals — initialized on startup
detector: Optional[FaceDetector] = None
engine: Optional[LivenessEngine] = None
gpu_name: str = "N/A"
_models_loaded: bool = False

# Layer 0 document-photo checker — cheap to construct (holds no model
# weights, just HTTP config), always built regardless of DOCUMENT_CHECK_ENABLED
# so flipping the flag at runtime doesn't require a restart.
document_checker: DocumentPhotoChecker = DocumentPhotoChecker(
    model=settings.DOCUMENT_CHECK_MODEL,
    ollama_url=settings.DOCUMENT_CHECK_OLLAMA_URL,
    timeout_s=settings.DOCUMENT_CHECK_TIMEOUT_S,
)

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
    """Two-level sliding-window rate limiter, per-IP.

    - Burst: at most `burst` requests within the last 1 second.
    - Sustained: at most `sustained` requests/sec on AVERAGE over a
      `SUSTAINED_WINDOW_S`-second window (e.g. 5 req/s over 60s == 300/60s).
    Both checks share one deque of monotonic timestamps per key.
    """

    SUSTAINED_WINDOW_S = 60.0
    _PRUNE_EVERY = 500  # opportunistically drop stale/empty per-IP deques

    def __init__(self, burst: int, sustained: float) -> None:
        self._burst = burst
        self._sustained = sustained
        self._windows: dict[str, deque] = {}
        self._calls_since_prune = 0

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._windows.setdefault(key, deque())

        sustained_cutoff = now - self.SUSTAINED_WINDOW_S
        while window and window[0] < sustained_cutoff:
            window.popleft()

        burst_cutoff = now - 1.0
        burst_count = sum(1 for t in window if t >= burst_cutoff)
        sustained_limit = max(1, int(self._sustained * self.SUSTAINED_WINDOW_S))

        allowed = burst_count < self._burst and len(window) < sustained_limit
        if allowed:
            window.append(now)

        self._calls_since_prune += 1
        if self._calls_since_prune >= self._PRUNE_EVERY:
            self._prune_stale(sustained_cutoff)
            self._calls_since_prune = 0

        return allowed

    def _prune_stale(self, cutoff: float) -> None:
        """Drop per-IP deques that are empty or fully stale (memory-leak guard)."""
        stale_keys = [k for k, w in self._windows.items() if not w or w[-1] < cutoff]
        for k in stale_keys:
            del self._windows[k]


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
                "reason": "INTERNAL_ERROR",
                "score": 0.0,
                "threshold": settings.LIVENESS_THRESHOLD,
                "face_detected": False,
                "save_frame": False,
                "signals": {},
                "model_version": MODEL_VERSION,
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


# CORS — OFF by default (MF DOOM review, 2026-07-16: unconditional
# allow_origins=["*"] is unnecessary attack surface — the real production
# caller is server-side Laravel, not a browser). The middleware is only
# attached when CORS_ALLOW_ORIGINS is non-empty, which is the case for local
# manual browser testing via testpage/index.html (set the env var, do not
# hardcode "*" here).
if settings.CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


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


def _run_geometry_gate(bbox: list[int], image_bgr: np.ndarray) -> Optional[GeometryCheckResult]:
    """Layer 0a — shared face-to-frame geometry gate.

    Reuses the bbox already computed by `FaceDetector.detect()` for this
    request — no extra model, no network call. Shared by ALL detect
    endpoints (/verify, /verify_batch, /spoof-server, /pad/check) so the
    logic and calibration live in exactly one place (app/geometry_check.py).

    Returns the `GeometryCheckResult` when the gate fired (caller should
    short-circuit to its own document/spoof verdict in its native response
    shape), or `None` when the gate is disabled, did not run (bad input —
    should not happen with a real bbox), or did not flag the frame (caller
    falls through to passive-PAD unchanged). Never raises.

    2026-07-16 2PAC review: `width_threshold` is deliberately NOT passed
    here. `face_width_ratio` is ~1.09*sqrt(face_area_ratio) on every sample
    measured so far (see app/geometry_check.py docstring) — gating on it too
    does not catch any attack area_ratio misses, it only tightens the
    effective margin against real bonafide near a close-up camera (pure FRR
    cost, no FAR benefit). `face_width_ratio` is still computed and reported
    as a diagnostic-only field (see `_geometry_signals`) for future
    independent calibration; it is not part of the reject decision.
    """
    if not settings.GEOMETRY_CHECK_ENABLED:
        return None
    geo_result = check_face_geometry(bbox, image_bgr.shape[:2], settings.FACE_RATIO_REJECT)
    if geo_result.ran and geo_result.is_document:
        return geo_result
    return None


def _geometry_signals(geo_result: GeometryCheckResult) -> dict:
    """Build the `signals` sub-dict shape used across endpoints for a geometry-gate hit."""
    return {
        "geometry_check": {
            "face_area_ratio": geo_result.face_area_ratio,
            "face_width_ratio": geo_result.face_width_ratio,
            "frame_aspect_ratio": geo_result.frame_aspect_ratio,
        }
    }


def _run_single(image_bgr: np.ndarray) -> dict:
    """Detect face + (Layer 0a geometry gate) + predict liveness for a single image."""
    bbox = detector.detect(image_bgr)
    if bbox is None:
        return {
            "is_real": False,
            "label": "no_face",
            "score": 0.0,
            "threshold": settings.LIVENESS_THRESHOLD,
            "face_detected": False,
        }

    geo_result = _run_geometry_gate(bbox, image_bgr)
    if geo_result is not None:
        return {
            "is_real": False,
            "label": "document_photo",
            "score": round(geo_result.face_area_ratio, 4),
            "threshold": settings.FACE_RATIO_REJECT,
            "face_detected": True,
            "signals": _geometry_signals(geo_result),
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
            # Layer 0a geometry gate — per-frame, BEFORE this frame is added
            # to the liveness batch. A hit short-circuits this frame only;
            # other frames in the batch are unaffected.
            geo_result = _run_geometry_gate(bbox, img)
            if geo_result is not None:
                results[i] = {
                    "is_real": False,
                    "label": "document_photo",
                    "score": round(geo_result.face_area_ratio, 4),
                    "threshold": settings.FACE_RATIO_REJECT,
                    "face_detected": True,
                    "signals": _geometry_signals(geo_result),
                }
                continue

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
    response: dict = {"elapsed_time": elapsed, "is_spoof": is_spoof}
    # Additive-only field — existing consumers reading elapsed_time/is_spoof
    # are unaffected; present only when Layer 0a geometry gate fired
    # (result["label"] == "document_photo", see _run_single).
    if result.get("label") == "document_photo":
        response["reason"] = "DOCUMENT_PHOTO"
    return response


# ---------------------------------------------------------------------------
# POST /pad/check — PAD-gate integration (Phase 1)
# ---------------------------------------------------------------------------

class PadCheckRequest(BaseModel):
    """PAD-gate request from Laravel."""
    correlation_id: str = Field(..., description="UUID from Laravel for log tracing")
    transaction_type: Literal["sale"] = Field(
        "sale", description="Only 'sale' is confirmed in v1 (see FACEID_PHASE1_PAD_GATE §1)"
    )
    transaction_ref: str = Field(..., description="id_request:id_ballon (natural key)")
    face_photo: str = Field(..., description="Base64 JPEG/PNG — same frame as Adliya")


class PadCheckResponse(BaseModel):
    """PAD-gate response — contract per FACEID_PHASE1_PAD_GATE.md §1."""
    verdict: Literal["live", "spoof", "low_quality"] = Field(...)
    reason: Optional[str] = Field(
        None,
        description="PASSIVE_PAD_SPOOF | DOCUMENT_PHOTO | NO_FACE | LOW_QUALITY | TIMEOUT | INTERNAL_ERROR | null",
    )
    score: float = Field(..., description="Confidence score [0..1]")
    threshold: float = Field(..., description="Liveness threshold used for this decision")
    face_detected: bool = Field(..., description="Whether a face was found in the frame")
    signals: dict = Field(default_factory=dict, description="Multi-signal breakdown")
    save_frame: bool = Field(False, description="Service already decided to persist this frame")
    model_version: str = Field(..., description="Model/ensemble version used for this verdict")
    processing_ms: float = Field(..., description="Server-side processing time")


_save_frame_verdicts = set(v.strip() for v in settings.SAVE_FRAME_VERDICTS.split(","))


def _verify_service_token(x_service_token: Optional[str]) -> None:
    if not settings.SERVICE_TOKEN:
        return
    if not x_service_token or x_service_token != settings.SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Service-Token")


def _audit_entry(correlation_id: str, transaction_type: str, transaction_ref: str,
                 verdict: str, score: float, signals: dict, processing_ms: float,
                 save_frame: bool, reason: Optional[str] = None) -> None:
    """Write structured JSON audit log entry. Metadata only — never the frame itself."""
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
    """PAD-gate: classify a face frame as live/spoof/low_quality.

    Called by Laravel after Adliya match (BACKEND_REQUIREMENTS_2026-07-06 п.8).
    Contract: FACEID_PHASE1_PAD_GATE.md §1.
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

    # Models ready? (service-side failure, not a bad frame — same family as TIMEOUT/INTERNAL_ERROR)
    if not _models_loaded or detector is None or engine is None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        _audit_entry(req.correlation_id, req.transaction_type, req.transaction_ref,
                     "low_quality", 0.0, {}, ms, False, reason="INTERNAL_ERROR")
        return PadCheckResponse(
            verdict="low_quality", reason="INTERNAL_ERROR", score=0.0,
            threshold=settings.LIVENESS_THRESHOLD, face_detected=False,
            save_frame=False, signals={}, model_version=MODEL_VERSION, processing_ms=ms,
        )

    # Detect face
    bbox = detector.detect(img)
    if bbox is None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        _audit_entry(req.correlation_id, req.transaction_type, req.transaction_ref,
                     "low_quality", 0.0, {}, ms, False, reason="NO_FACE")
        return PadCheckResponse(
            verdict="low_quality", reason="NO_FACE", score=0.0,
            threshold=settings.LIVENESS_THRESHOLD, face_detected=False,
            save_frame=False, signals={}, model_version=MODEL_VERSION, processing_ms=ms,
        )

    # Layer 0a — deterministic face-to-frame geometry gate (runs BEFORE
    # passive-PAD and before the minicpm-v Layer 0b below). Shared helper
    # (_run_geometry_gate) reuses the SAME bbox just computed above — no
    # extra model, no network call, microseconds. See app/geometry_check.py
    # for calibration numbers and known limitations (n=1 spoof sample,
    # evadable by a smarter attacker who does not fill the frame). Fail-safe:
    # any bad input (should not happen given a real bbox) falls straight
    # through to passive-PAD.
    geo_result = _run_geometry_gate(bbox, img)
    if geo_result is not None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        geo_signals = _geometry_signals(geo_result)
        _audit_entry(
            req.correlation_id, req.transaction_type, req.transaction_ref,
            "spoof", geo_result.face_area_ratio, geo_signals, ms, True, reason="DOCUMENT_PHOTO",
        )
        log.info(
            "PAD check: correlation=%s verdict=spoof reason=DOCUMENT_PHOTO "
            "face_area_ratio=%.3f ms=%.1f txn=%s",
            req.correlation_id, geo_result.face_area_ratio, ms, req.transaction_ref,
        )
        return PadCheckResponse(
            verdict="spoof", reason="DOCUMENT_PHOTO", score=round(geo_result.face_area_ratio, 4),
            threshold=settings.FACE_RATIO_REJECT, face_detected=True,
            save_frame=True, signals=geo_signals, model_version=MODEL_VERSION, processing_ms=ms,
        )
    # Below threshold, not ran (disabled or bad input) — continue unchanged.

    # Layer 0b — document/passport-photo pre-filter via minicpm-v (runs BEFORE
    # passive-PAD). DEFAULT DISABLED (see app/document_check.py for why).
    # Ortho signal: catches studio/ID-style composition (plain backdrop,
    # matted cutout, passport pose) that the texture/frequency-based
    # passive-PAD signals do not target. Fail-open unconditionally: any
    # non-"ran" result (disabled, Ollama down, timeout, unparseable) falls
    # straight through to passive-PAD below, unchanged.
    if settings.DOCUMENT_CHECK_ENABLED:
        try:
            doc_result = await asyncio.wait_for(
                asyncio.to_thread(document_checker.check, img),
                timeout=settings.DOCUMENT_CHECK_TIMEOUT_S + 1.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "document_check: outer timeout for correlation=%s — failing open to passive-PAD",
                req.correlation_id,
            )
            doc_result = None

        if doc_result is not None and doc_result.ran and doc_result.is_document \
                and doc_result.confidence >= settings.DOCUMENT_REJECT_THRESHOLD:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            doc_signals = {
                "document_check": {
                    "label": doc_result.raw_label,
                    "confidence": round(doc_result.confidence, 4),
                }
            }
            _audit_entry(
                req.correlation_id, req.transaction_type, req.transaction_ref,
                "spoof", doc_result.confidence, doc_signals, ms, True, reason="DOCUMENT_PHOTO",
            )
            log.info(
                "PAD check: correlation=%s verdict=spoof reason=DOCUMENT_PHOTO confidence=%.3f ms=%.1f txn=%s",
                req.correlation_id, doc_result.confidence, ms, req.transaction_ref,
            )
            return PadCheckResponse(
                verdict="spoof", reason="DOCUMENT_PHOTO", score=round(doc_result.confidence, 4),
                threshold=settings.DOCUMENT_REJECT_THRESHOLD, face_detected=True,
                save_frame=True, signals=doc_signals, model_version=MODEL_VERSION, processing_ms=ms,
            )
        # Below threshold, not ran, or fail-open — continue to passive-PAD unchanged.

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
            verdict="low_quality", reason="TIMEOUT", score=0.0,
            threshold=settings.LIVENESS_THRESHOLD, face_detected=True,
            save_frame=False, signals={}, model_version=MODEL_VERSION, processing_ms=ms,
        )

    # Map to PAD-gate verdict (§1) — verdict/threshold logic unchanged, only naming aligned to spec
    if label == "spoof":
        verdict = "spoof"
        reason: Optional[str] = "PASSIVE_PAD_SPOOF"
    elif score < settings.LIVENESS_THRESHOLD:
        verdict = "low_quality"
        reason = "LOW_QUALITY"
    else:
        verdict = "live"
        reason = None

    save_frame = verdict in _save_frame_verdicts
    processing_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Audit log (every request, including "live" — for APCER/BPCER analysis)
    _audit_entry(
        req.correlation_id, req.transaction_type, req.transaction_ref,
        verdict, score, signal_info, processing_ms, save_frame, reason=reason,
    )

    log.info(
        "PAD check: correlation=%s verdict=%s score=%.3f ms=%.1f txn=%s",
        req.correlation_id, verdict, score, processing_ms, req.transaction_ref,
    )

    return PadCheckResponse(
        verdict=verdict,
        reason=reason,
        score=round(score, 4),
        threshold=settings.LIVENESS_THRESHOLD,
        face_detected=face_detected,
        save_frame=save_frame,
        signals=signal_info,
        model_version=MODEL_VERSION,
        processing_ms=processing_ms,
    )
