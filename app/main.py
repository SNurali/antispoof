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

from app.active_challenge import verify_challenge
from app.config import Settings, resolve_device
from app.document_check import DocumentPhotoChecker
from app.face_detect import FaceDetector
from app.frame_qc import assess_frame
from app.geometry_check import GeometryCheckResult, check_face_geometry
from app.identity_consistency import compute_identity_consistency
from app.liveness import LivenessEngine
from app.liveness_session import SessionStore, generate_challenge_spec

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

# Active-liveness globals (Layer 0/2/3) — only constructed when
# settings.LIVENESS_ENDPOINTS_ENABLED is True (see _load_liveness_models).
# Typed Optional[object] rather than the real classes here so this module
# can be imported without insightface/onnxruntime installed at all when the
# flag is off — the real types live in app/face_landmarks.py and
# app/adaface.py, imported lazily inside _load_liveness_models().
landmark_detector = None
adaface_embedder = None
session_store = SessionStore()  # cheap (pure Python), always constructed
_liveness_models_loaded: bool = False

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
    # For /liveness/verdict — fail-CLOSED per FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md
    # R1(в): an unhandled exception here must never look like "live".
    if request.url.path == "/liveness/verdict":
        return JSONResponse(
            status_code=500,
            content={
                "verdict": "incomplete",
                "reason": "INTERNAL_ERROR",
                "model_version": MODEL_VERSION,
                "frame_consistency_score": 0.0,
                "best_frame_seq": None,
                "session_id": None,
                "correlation_id": None,
                "transaction_type": None,
                "transaction_ref": None,
                "processing_ms": 0.0,
                "signals": {},
            },
        )
    # For /spoof-server requests, return contract-shaped error (same verdict/reason as /pad/check)
    if request.url.path == "/spoof-server":
        return JSONResponse(
            status_code=500,
            content={
                "elapsed_time": 0.0,
                "is_spoof": 1,  # Fail-closed: assume spoof on internal error
                "verdict": "low_quality",
                "reason": "INTERNAL_ERROR",
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

    _load_liveness_models()


def _load_liveness_models() -> None:
    """Layer 0/2/3 models for POST /liveness/challenge + /liveness/verdict.

    Deliberately separate from _load_models() above and gated on
    settings.LIVENESS_ENDPOINTS_ENABLED — importing insightface/onnxruntime
    and loading the 260MB AdaFace ONNX weight file must never happen (and
    must never be able to break startup) on a deploy that has not rolled
    this feature out. Any failure here is logged and leaves
    _liveness_models_loaded False — /liveness/challenge and
    /liveness/verdict then respond 503, they do not crash the whole app.
    """
    global landmark_detector, adaface_embedder, _liveness_models_loaded

    if not settings.LIVENESS_ENDPOINTS_ENABLED:
        log.info("LIVENESS_ENDPOINTS_ENABLED=False — /liveness/challenge and /liveness/verdict return 503.")
        return

    try:
        from app.adaface import AdaFaceEmbedder
        from app.face_landmarks import LandmarkDetector

        landmark_detector = LandmarkDetector(det_size=settings.LIVENESS_DET_SIZE, device=DEVICE)
        adaface_embedder = AdaFaceEmbedder(settings.ADAFACE_ONNX_PATH, device=DEVICE)
        _liveness_models_loaded = True
        log.info("Active-liveness models loaded (Layer 0/2/3 ready).")
    except Exception:
        log.exception(
            "Failed to load active-liveness models — /liveness/* will return 503. "
            "Existing /verify, /verify_batch, /spoof-server, /pad/check are unaffected."
        )
        _liveness_models_loaded = False


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
            # Additive fields — existing consumers reading only the fields
            # above are unaffected. /health stays 200 based on the Phase 1
            # models above even if active-liveness failed to load; Laravel's
            # fail-closed policy for /liveness/* is enforced by THOSE
            # endpoints returning 503 individually, not by /health.
            "liveness_endpoints_enabled": settings.LIVENESS_ENDPOINTS_ENABLED,
            "liveness_models_loaded": _liveness_models_loaded,
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


class SpoofServerResponse(BaseModel):
    """Legacy /spoof-server response — verdict field is new (2026-07-16), is_spoof/elapsed_time legacy."""
    elapsed_time: float = Field(..., description="Server-side processing time in seconds")
    is_spoof: int = Field(..., description="0=real, 1=spoof (backward compat)")
    verdict: Literal["live", "spoof", "low_quality"] = Field(..., description="Verdict per FACEID_PHASE1_PAD_GATE")
    reason: Optional[str] = Field(
        None,
        description="PASSIVE_PAD_SPOOF | DOCUMENT_PHOTO | NO_FACE | LOW_QUALITY | null",
    )


@app.post("/spoof-server", response_model=SpoofServerResponse)
async def spoof_server(req: SpoofRequest) -> SpoofServerResponse:
    t0 = time.perf_counter()
    try:
        img_bytes = base64.b64decode(req.photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    _validate_image_size(img_bytes)
    img = _read_image(img_bytes)
    _validate_image_dimensions(img)

    # Models ready? (service-side failure — fail-closed)
    if not _models_loaded or detector is None or engine is None:
        elapsed = round(time.perf_counter() - t0, 3)
        return SpoofServerResponse(
            elapsed_time=elapsed,
            is_spoof=1,  # Fail-closed
            verdict="low_quality",
            reason="INTERNAL_ERROR",
        )

    result = _run_single(img)

    elapsed = round(time.perf_counter() - t0, 3)
    is_spoof = 0 if result["is_real"] else 1

    # Map to verdict enum per FACEID_PHASE1_PAD_GATE contract (§1).
    # IDENTICAL logic to /pad/check (app/main.py ~744-752) to prevent divergence.
    # This is an ADDITIVE field — existing consumers reading only
    # elapsed_time/is_spoof are unaffected (same values as before).
    label = result.get("label", "unknown")
    score = result.get("score", 0.0)

    # Explicit three-branch verdict mapping (NOT via is_real boolean).
    # This prevents the regression where label="real" + score<threshold → spoof.
    if label == "document_photo":
        verdict = "spoof"
        reason: Optional[str] = "DOCUMENT_PHOTO"
    elif label == "no_face":
        verdict = "low_quality"
        reason = "NO_FACE"
    elif label == "spoof":
        verdict = "spoof"
        reason = "PASSIVE_PAD_SPOOF"
    elif score < settings.LIVENESS_THRESHOLD:
        # Real face but low score (bad lighting, occlusion, etc.) → low quality
        verdict = "low_quality"
        reason = "LOW_QUALITY"
    else:
        # label="real" and score >= threshold → live
        verdict = "live"
        reason = None

    return SpoofServerResponse(
        elapsed_time=elapsed,
        is_spoof=is_spoof,
        verdict=verdict,
        reason=reason,
    )


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


# ---------------------------------------------------------------------------
# POST /liveness/challenge + POST /liveness/verdict — active liveness
# (Phase 2, FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md §1.2 / FACEID_LIVENESS_ML_CORE_v1.md)
#
# Internal ML-perimeter endpoints — Laravel calls these when it handles the
# PUBLIC POST /liveness/start / POST /liveness/verify, same pattern already
# used for /pad/check (127.0.0.1-only, X-Service-Token, not client-facing).
# Does NOT touch /verify, /verify_batch, /spoof-server, /pad/check.
# ---------------------------------------------------------------------------

LIVENESS_MODEL_VERSION = (
    f"{MODEL_VERSION}"
    "+scrfd-buffalo_l-det+landmark_3d_68"
    "+adaface-ir101-webface12m-onnx"
    "+active_challenge-turn_only_v1"
)


class LivenessChallengeRequest(BaseModel):
    """`transaction_type` is passthrough ONLY here — unlike PadCheckRequest's
    Literal["sale"], this service does not validate it (per explicit
    instruction: identity/transaction semantics are Laravel's domain)."""
    correlation_id: str = Field(
        ..., description="UUID minted by Laravel in POST /liveness/start, echoed unchanged through "
        "/liveness/challenge -> /liveness/verdict. This is the R2 SESSION-BINDING key (see "
        "LivenessVerdictRequest.correlation_id) as well as a log-tracing id.",
    )
    transaction_type: str = Field(..., description="Passthrough, not validated by this service")
    transaction_ref: str = Field(
        ..., description="Natural key, e.g. id_request:id_ballon — PASSTHROUGH ONLY for /liveness/*. "
        "May legitimately not be final at challenge time; NOT used for session binding (correlation_id is).",
    )


class ChallengeSpec(BaseModel):
    steps: list[str] = Field(..., description="Randomized subset+order, e.g. ['TURN_RIGHT','TURN_LEFT']")
    min_frames: int
    max_frames: int


class LivenessChallengeResponse(BaseModel):
    session_id: str
    challenge_spec: ChallengeSpec
    t_instruction_shown: float = Field(..., description="Unix timestamp, echoed back for window-timing checks")
    expires_at: float = Field(..., description="Unix timestamp — session_id invalid after this")
    model_version: str


class LivenessFrame(BaseModel):
    seq: int = Field(..., ge=0)
    base64: str
    captured_at: Optional[str] = None


class LivenessVerdictRequest(BaseModel):
    correlation_id: str = Field(
        ..., description="Must match the correlation_id the session_id was minted under — the R2 "
        "binding check (see app/main.py::_run_liveness_verdict) compares THIS, not transaction_ref.",
    )
    session_id: str
    transaction_type: str = Field(..., description="Passthrough, not validated by this service")
    transaction_ref: str = Field(..., description="Passthrough only — not part of the R2 binding check.")
    frames: list[LivenessFrame]


class LivenessVerdictResponse(BaseModel):
    """`reason` is the DETAILED internal cause — per contract this response
    is consumed by Laravel only, never relayed verbatim to the end client
    (Laravel must map it to a generic client-facing message/code so an
    attacker cannot calibrate an evasion attempt against per-signal
    feedback)."""
    verdict: Literal["live", "spoof", "incomplete", "low_quality"]
    reason: Optional[str] = None
    model_version: str
    frame_consistency_score: float = Field(
        ..., description="Layer 3 min pairwise cosine similarity across key frames; -1.0 if not computed"
    )
    best_frame_seq: Optional[int] = Field(
        None,
        description="seq of the frame Laravel should forward downstream (e.g. to Adliya) — "
        "ALWAYS one of the frames in the request, never re-sampled. Only set when verdict=live.",
    )
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    transaction_type: Optional[str] = None
    transaction_ref: Optional[str] = None
    processing_ms: float
    signals: dict = Field(default_factory=dict, description="Internal-only layer breakdown, not for client display")


def _liveness_audit_entry(
    correlation_id: Optional[str], session_id: Optional[str], transaction_type: Optional[str],
    transaction_ref: Optional[str], verdict: str, reason: Optional[str],
    frame_consistency_score: float, processing_ms: float, n_frames_received: int, n_frames_valid: int,
) -> None:
    """Metadata-only audit entry — NEVER the frame bytes themselves, same
    privacy posture as _audit_entry() for /pad/check."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": "liveness_verdict",
        "correlation_id": correlation_id,
        "session_id": session_id,
        "transaction_type": transaction_type,
        "transaction_ref": transaction_ref,
        "verdict": verdict,
        "reason": reason,
        "frame_consistency_score": frame_consistency_score,
        "n_frames_received": n_frames_received,
        "n_frames_valid": n_frames_valid,
        "model_version": LIVENESS_MODEL_VERSION,
        "processing_ms": round(processing_ms, 1),
    }
    audit_log.info(json.dumps(entry, ensure_ascii=False))


def _liveness_not_ready_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Active-liveness endpoints are disabled or models failed to load"},
    )


@app.post("/liveness/challenge", response_model=LivenessChallengeResponse)
async def liveness_challenge(
    req: LivenessChallengeRequest,
    x_service_token: Optional[str] = Header(None, alias="X-Service-Token"),
):
    """Generates a randomized challenge_spec and mints session_id. Laravel
    relays session_id + challenge_spec to the client unchanged when it
    handles the public POST /liveness/start. THIS service is the sole
    source of truth for what challenge_spec a given session_id means —
    POST /liveness/verdict looks it up by session_id, it does not trust a
    client- or Laravel-supplied spec."""
    _verify_service_token(x_service_token)

    if not settings.LIVENESS_ENDPOINTS_ENABLED or not _liveness_models_loaded:
        return _liveness_not_ready_response()

    pool = [s.strip() for s in settings.LIVENESS_CHALLENGE_STEPS_POOL.split(",") if s.strip()]
    steps = generate_challenge_spec(pool, settings.LIVENESS_CHALLENGE_STEP_COUNT)

    session = session_store.create(
        steps=steps,
        ttl_s=settings.LIVENESS_SESSION_TTL_S,
        correlation_id=req.correlation_id,
        transaction_type=req.transaction_type,
        transaction_ref=req.transaction_ref,
    )

    log.info(
        "liveness_challenge: correlation=%s session=%s steps=%s txn=%s",
        req.correlation_id, session.session_id, steps, req.transaction_ref,
    )

    return LivenessChallengeResponse(
        session_id=session.session_id,
        challenge_spec=ChallengeSpec(
            steps=steps,
            min_frames=settings.LIVENESS_MIN_FRAMES,
            max_frames=settings.LIVENESS_MAX_FRAMES,
        ),
        t_instruction_shown=session.t_instruction_shown,
        expires_at=session.expires_at,
        model_version=LIVENESS_MODEL_VERSION,
    )


def _run_liveness_verdict(req: LivenessVerdictRequest) -> LivenessVerdictResponse:
    """Synchronous body of POST /liveness/verdict — run via asyncio.wait_for
    with LIVENESS_INFERENCE_TIMEOUT_S from the route handler below."""
    t0 = time.perf_counter()

    def _respond(verdict: str, reason: Optional[str], *, frame_consistency_score: float = -1.0,
                 best_frame_seq: Optional[int] = None, signals: Optional[dict] = None) -> LivenessVerdictResponse:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        out_signals = dict(signals or {})
        n_valid = out_signals.pop("_n_valid", 0)  # audit-only counter, not part of the outward signals shape
        _liveness_audit_entry(
            req.correlation_id, req.session_id, req.transaction_type, req.transaction_ref,
            verdict, reason, frame_consistency_score, ms,
            n_frames_received=len(req.frames), n_frames_valid=n_valid,
        )
        return LivenessVerdictResponse(
            verdict=verdict, reason=reason, model_version=LIVENESS_MODEL_VERSION,
            frame_consistency_score=frame_consistency_score, best_frame_seq=best_frame_seq,
            session_id=req.session_id, correlation_id=req.correlation_id,
            transaction_type=req.transaction_type, transaction_ref=req.transaction_ref,
            processing_ms=ms, signals=out_signals,
        )

    # --- session lookup (R2: challenge_spec is ALWAYS the one this service
    # generated for this session_id, never client/Laravel-supplied) ---
    session, err = session_store.consume(req.session_id)
    if err is not None:
        return _respond("incomplete", err)

    # --- R2 binding check: the frames graded here must belong to the SAME
    # /liveness/start session the challenge was issued for. Per backend
    # confirmation (agent-mesh 2026-07-17, umid-agent msg 1784265688632-0):
    # `correlation_id` is minted by Laravel in POST /liveness/start and
    # echoed unchanged through /liveness/challenge -> /liveness/verdict —
    # THAT is the binding key, not `transaction_ref`. `transaction_ref`
    # (the id_request:id_ballon natural sale key) can legitimately not be
    # final yet when the challenge is issued (a sale reference can be
    # assigned/confirmed after the liveness check runs) and is kept on the
    # request purely as passthrough for audit/logging, same as
    # `transaction_type` — it is intentionally NOT compared here anymore.
    # Soft in this increment — logged and rejected, but not yet proven
    # against a real Laravel integration test, see final report.
    if session.correlation_id != req.correlation_id:
        return _respond(
            "incomplete", "SESSION_CORRELATION_MISMATCH",
            signals={"session_correlation_id": session.correlation_id, "request_correlation_id": req.correlation_id},
        )

    # --- per-frame decode + Layer 0 QC + Layer 0a geometry gate ---
    from app.face_landmarks import LandmarkDetector  # local import, models already loaded at startup

    per_frame_signals: dict = {}
    valid_frames: list[tuple[int, np.ndarray, "FrameFace", np.ndarray]] = []
    geometry_hits: dict = {}
    for frame in sorted(req.frames, key=lambda f: f.seq):
        try:
            img_bytes = base64.b64decode(frame.base64)
            _validate_image_size(img_bytes)
            img = _read_image(img_bytes)
            _validate_image_dimensions(img)
        except HTTPException:
            per_frame_signals[frame.seq] = {"valid": False, "reason": "DECODE_ERROR"}
            continue

        face = landmark_detector.analyze(img)
        aligned = LandmarkDetector.align_112(img, face.kps) if face is not None else None
        qc = assess_frame(img, face, aligned_112=aligned)
        frame_signal = {"valid": qc.valid, "reason": qc.reason, "metrics": qc.metrics}

        # Layer 0a — SAME deterministic face-to-frame geometry gate already
        # used by /pad/check and friends (app/geometry_check.py), reused
        # here unmodified via the shared _run_geometry_gate/_geometry_signals
        # helpers — no new logic, no new calibration. Runs on the RAW frame
        # (not the QC-gated crop) because a document/passport photo held up
        # to the camera is often sharp/well-lit enough to otherwise pass
        # Layer 0 QC cleanly; this must not depend on a frame first being
        # judged "valid" by frame_qc. `face.bbox_xyxy` is SCRFD's box
        # (x1,y1,x2,y2) — converted to the [x,y,w,h] shape
        # check_face_geometry() expects (same shape FaceDetector.detect()
        # returns for the RetinaFace-based endpoints).
        if face is not None:
            x1, y1, x2, y2 = face.bbox_xyxy
            bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            geo_result = _run_geometry_gate(bbox_xywh, img)
            if geo_result is not None:
                frame_signal["geometry_check"] = _geometry_signals(geo_result)["geometry_check"]
                geometry_hits[frame.seq] = geo_result

        per_frame_signals[frame.seq] = frame_signal
        if qc.valid:
            valid_frames.append((frame.seq, img, face, aligned))

    # Any frame flagged as a document/ID photo fails the WHOLE session,
    # same priority as /pad/check (before passive-PAD, before the
    # MIN_FRAMES completeness check — a sharp, well-lit document photo can
    # otherwise sail through Layer 0 QC and would wrongly read as
    # "incomplete" instead of "spoof" if checked after the MIN_FRAMES gate).
    if geometry_hits:
        base_signals = {"layer0_frame_qc": per_frame_signals, "_n_valid": len(valid_frames)}
        base_signals["layer0a_geometry_check"] = {
            str(seq): _geometry_signals(geo)["geometry_check"] for seq, geo in geometry_hits.items()
        }
        # frame_consistency_score stays -1.0 (its documented "not computed"
        # value) — Layer 3 never ran, and that field is specifically Layer
        # 3's cosine similarity, not a generic score slot; the geometry
        # ratio lives in signals.layer0a_geometry_check instead.
        return _respond("spoof", "DOCUMENT_PHOTO", signals=base_signals)

    n_valid = len(valid_frames)
    base_signals = {"layer0_frame_qc": per_frame_signals, "_n_valid": n_valid}

    if n_valid < settings.LIVENESS_MIN_FRAMES:
        return _respond("incomplete", "LOW_QUALITY_FRAMES", signals=base_signals)

    # --- Layer 2: active challenge (OBLIGATORY gate — runs BEFORE Layer 3/1,
    # per FACEID_LIVENESS_ML_CORE_v1.md §3: a good passive/identity score
    # must never compensate for a failed active challenge) ---
    active_result = verify_challenge(
        session.steps, [(seq, face) for seq, _, face, _ in valid_frames], settings,
    )
    base_signals["layer2_active_challenge"] = {
        "passed": active_result.passed, "reason": active_result.reason, "detail": active_result.detail,
        "requested_steps": session.steps,
    }
    if not active_result.passed:
        if active_result.reason == "UNSUPPORTED_STEP":
            return _respond("incomplete", "ACTIVE_CHALLENGE_NOT_IMPLEMENTED", signals=base_signals)
        if active_result.reason == "NO_FRONTAL_REFERENCE":
            return _respond("incomplete", "NO_FRONTAL_REFERENCE", signals=base_signals)
        return _respond("spoof", "CHALLENGE_FAILED", signals=base_signals)

    # --- Layer 3: cross-frame identity consistency (PRIORITY #1 of this
    # increment — see app/identity_consistency.py) ---
    identity_result = compute_identity_consistency(
        adaface_embedder, [(seq, aligned) for seq, _, _, aligned in valid_frames], settings.IDENTITY_MIN,
    )
    base_signals["layer3_identity_consistency"] = {
        "passed": identity_result.passed, "min_similarity": identity_result.min_similarity,
        "reference_seq": identity_result.reference_seq, "pairwise": identity_result.pairwise,
        "threshold": settings.IDENTITY_MIN,
    }
    if not identity_result.passed:
        return _respond(
            "spoof", "IDENTITY_SWAP_MID_SESSION",
            frame_consistency_score=identity_result.min_similarity, signals=base_signals,
        )

    # --- Layer 1: passive PAD, REUSED unmodified (defense-in-depth) ---
    # Simplification vs FACEID_LIVENESS_ML_CORE_v1.md §2.1's "75th
    # percentile of combined_score" recommendation: that recommendation
    # itself is flagged there as an unconfirmed working hypothesis, AND
    # combined_score's numeric meaning is label-conditional in the current
    # `_fuse()` implementation (real-confidence vs spoof-confidence are not
    # on a shared scale), so a naive percentile over raw scores would not
    # mean what the recommendation intends without deeper rework. This
    # increment uses "any valid key frame labeled spoof by the existing
    # engine.predict() -> aggregate spoof" instead — MORE conservative
    # (biases toward rejecting live traffic, i.e. higher BPCER not higher
    # APCER), consistent with "не занижай FAR". Revisit once real
    # multi-frame session data exists to calibrate the intended aggregation.
    layer1_frames = []
    passive_spoof = False
    for seq, img, _, _ in valid_frames:
        bbox = detector.detect(img)  # RetinaFace bbox, [x,y,w,h] — SEPARATE detector from Layer 0/2/3's SCRFD
        if bbox is None:
            layer1_frames.append({"seq": seq, "label": "no_face"})
            continue
        label, score, _, _sig = engine.predict(img, bbox)
        layer1_frames.append({"seq": seq, "label": label, "score": round(score, 4)})
        if label == "spoof":
            passive_spoof = True
    base_signals["layer1_passive_pad"] = {"frames": layer1_frames, "aggregate": "any_frame_spoof"}

    if passive_spoof:
        return _respond(
            "spoof", "PASSIVE_PAD_SPOOF",
            frame_consistency_score=identity_result.min_similarity, signals=base_signals,
        )

    # --- live: pick best_frame_seq from the SAME valid_frames set (R2 — no
    # re-sampling). Prefer frontal + sharpest among valid frames. ---
    def _quality_key(item):
        seq, _, face, _ = item
        frontal_bonus = 1.0 if abs(face.pose_yaw) <= settings.LIVENESS_YAW_FRONTAL_MAX_DEG else 0.0
        sharp = per_frame_signals.get(seq, {}).get("metrics", {}).get("sharpness", 0.0)
        return (frontal_bonus, sharp)

    best_seq = max(valid_frames, key=_quality_key)[0]

    return _respond(
        "live", None, frame_consistency_score=identity_result.min_similarity,
        best_frame_seq=best_seq, signals=base_signals,
    )


@app.post("/liveness/verdict", response_model=LivenessVerdictResponse)
async def liveness_verdict(
    req: LivenessVerdictRequest,
    x_service_token: Optional[str] = Header(None, alias="X-Service-Token"),
):
    _verify_service_token(x_service_token)

    if not settings.LIVENESS_ENDPOINTS_ENABLED or not _liveness_models_loaded:
        return _liveness_not_ready_response()

    if not req.frames or len(req.frames) > settings.LIVENESS_MAX_FRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"frames must be 1..{settings.LIVENESS_MAX_FRAMES}, got {len(req.frames)}",
        )
    seqs = [f.seq for f in req.frames]
    if len(seqs) != len(set(seqs)):
        raise HTTPException(status_code=422, detail="duplicate frame seq values")

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_liveness_verdict, req),
            timeout=settings.LIVENESS_INFERENCE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        log.error(
            "liveness_verdict: inference timeout after %.1fs correlation=%s session=%s",
            settings.LIVENESS_INFERENCE_TIMEOUT_S, req.correlation_id, req.session_id,
        )
        _liveness_audit_entry(
            req.correlation_id, req.session_id, req.transaction_type, req.transaction_ref,
            "incomplete", "TIMEOUT", -1.0, settings.LIVENESS_INFERENCE_TIMEOUT_S * 1000,
            n_frames_received=len(req.frames), n_frames_valid=0,
        )
        return LivenessVerdictResponse(
            verdict="incomplete", reason="TIMEOUT", model_version=LIVENESS_MODEL_VERSION,
            frame_consistency_score=-1.0, best_frame_seq=None,
            session_id=req.session_id, correlation_id=req.correlation_id,
            transaction_type=req.transaction_type, transaction_ref=req.transaction_ref,
            processing_ms=round(settings.LIVENESS_INFERENCE_TIMEOUT_S * 1000, 1), signals={},
        )
