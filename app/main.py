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
import hmac
import json
import logging
import logging.handlers
import math
import os
import time
import uuid
from collections import deque
from datetime import datetime
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
from app.aspect_ratio_check import AspectRatioCheckResult, check_aspect_ratio
from app.blur_check import SharpnessCheckResult, check_face_sharpness
from app.config import Settings, resolve_device
from app.dedup_store import build_dedup_store, compute_phash
from app.document_check import DocumentPhotoChecker
from app.edge_sharpness_check import measure_edge_sharpness
from app.face_detect import FaceDetector
from app.frame_qc import assess_frame
from app.geometry_check import GeometryCheckResult, check_face_geometry
from app.identity_consistency import compute_identity_consistency
from app.pose_check import PoseCheckResult, check_face_pose
from app.resolution_check import ResolutionCheckResult, check_image_resolution
from app.liveness import LivenessEngine, pad_check_reason
from app.liveness_session import (
    StepWindowDict,
    build_session_store,
    generate_challenge_spec,
    generate_step_windows,
)

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
    # 2PAC review (2026-07-18): normalize before comparing — a bare `==`
    # against the literal "prod" silently treats any typo/casing variant
    # ("Prod", "PRODUCTION", "production", a stray leading/trailing space
    # from a copy-pasted systemd Environment= line) as "dev", i.e. the exact
    # failure mode this guard exists to prevent.
    if settings.ENVIRONMENT.strip().lower() == "prod":
        # P0-3 (2026-07-18): a silent dev-mode warning is not enough in prod —
        # refuse to start rather than serve /pad/check + /liveness/* with auth
        # OFF. Requires ENVIRONMENT=prod to be set in antispoof.service on
        # egaz-02.uz in lockstep with this change (50 CENT, deploy-side).
        raise RuntimeError(
            "SERVICE_TOKEN is empty while ENVIRONMENT=prod — refusing to start "
            "with X-Service-Token auth disabled in production. Set SERVICE_TOKEN "
            "(shared secret with Laravel) before starting this service."
        )
    log.warning(
        "SERVICE_TOKEN is empty — /pad/check auth is DISABLED (dev mode). "
        "Do NOT run in production without a SERVICE_TOKEN set."
    )

# P0-3 (2026-07-18), updated for the Redis session store: the in-memory
# SessionStore backend (app/liveness_session.py) is NOT shared across
# worker processes (see its module docstring) — a session minted by one
# worker would be invisible to /liveness/verdict handled by a different
# worker (silent SESSION_NOT_FOUND, not a crash), so multi-worker + the
# memory backend must be caught at startup rather than rediscovered in
# prod. The redis backend (SESSION_STORE_BACKEND=redis) IS shared across
# workers, so it is exempt from this guard. WEB_CONCURRENCY is the
# conventional env var for worker count (gunicorn/Heroku-style multi-worker
# deploys); ctl.sh/antispoof.service do not set it today (single
# `uvicorn ...` process, no --workers), but nothing stops a future deploy
# change from doing so without touching this file.
#
# ⚠️ KNOWN GAP (2PAC review, 2026-07-18): this ONLY sees WEB_CONCURRENCY. If
# someone runs `uvicorn app.main:app --workers 4` directly (no
# WEB_CONCURRENCY env var set), uvicorn's CLI forks workers AFTER this
# module has already been imported once in the parent process, and this
# code cannot reliably read the CLI's own --workers value from in here
# (uvicorn does not re-exec with an equivalent env var, and sys.argv in a
# forked worker does not have to reflect it either). MUST-CHECK for 50 CENT
# before any deploy config change: if a deploy ever launches uvicorn with
# `--workers N` (N>1) instead of WEB_CONCURRENCY, this guard will NOT catch
# it — either keep launching with a single worker + WEB_CONCURRENCY (or no
# worker flag at all, today's setup), or set WEB_CONCURRENCY=N to match
# --workers N so this guard actually sees it.
try:
    _web_concurrency = int((os.environ.get("WEB_CONCURRENCY") or "1").strip())
except ValueError as _exc:
    raise RuntimeError(
        f"WEB_CONCURRENCY must be an integer, got {os.environ.get('WEB_CONCURRENCY')!r}"
    ) from _exc
if _web_concurrency > 1 and settings.SESSION_STORE_BACKEND.strip().lower() != "redis":
    raise RuntimeError(
        f"WEB_CONCURRENCY={_web_concurrency} (>1) but SESSION_STORE_BACKEND="
        f"{settings.SESSION_STORE_BACKEND!r} (not 'redis') — the in-memory "
        "session store is not shared across worker processes. Set "
        "SESSION_STORE_BACKEND=redis (and REDIS_URL) before running with "
        "more than one worker, or run with a single worker "
        "(WEB_CONCURRENCY=1, no --workers flag)."
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
# Backend chosen via SESSION_STORE_BACKEND (memory|redis, app/config.py) —
# see build_session_store() docstring for the no-silent-fallback contract.
# Always constructed regardless of LIVENESS_ENDPOINTS_ENABLED, same as
# before, so flipping that flag at runtime doesn't need a restart.
session_store = build_session_store(settings)
_liveness_models_loaded: bool = False

# Frame-reuse dedup + inspector/abonent fraud-alert store (app/dedup_store.py)
# — always constructed regardless of DEDUP_ENABLED/DEDUP_EMBEDDING_ALERT_ENABLED
# /FRAUD_INSPECTOR_ALERT_ENABLED, same "flip the flag without a restart"
# pattern as session_store above, and so the SQLite file already exists
# before the first request in any test (see build_dedup_store docstring).
dedup_store = build_dedup_store(settings)

# Layer 0 document-photo checker — cheap to construct (holds no model
# weights, just HTTP config), always built regardless of DOCUMENT_CHECK_ENABLED
# so flipping the flag at runtime doesn't require a restart.
document_checker: DocumentPhotoChecker = DocumentPhotoChecker(
    model=settings.DOCUMENT_CHECK_MODEL,
    ollama_url=settings.DOCUMENT_CHECK_OLLAMA_URL,
    timeout_s=settings.DOCUMENT_CHECK_TIMEOUT_S,
)

MODEL_VERSION = "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1+print-pattern-v1"

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


def _effective_client_ip(request: Request) -> str:
    """Resolve the address the allowlist + rate limiter should judge.

    Deployment topology (BUSTA RHYMES, deploy/mtls/nginx-antispoof-mtls.conf):
    nginx terminates TLS+mTLS on the public interface and reverse-proxies to
    uvicorn on 127.0.0.1 only (firewalled off from the outside). Once that
    lands, request.client.host for EVERY external caller becomes nginx's own
    loopback address — the allowlist below would silently stop filtering
    anyone at all, since 127.0.0.0/8 is itself an allowed network.

    settings.TRUST_PROXY_HEADERS (default False, see app/config.py) opts in
    to reading X-Forwarded-For instead, but ONLY when the physical TCP peer
    that reached uvicorn really is loopback (request.client.host is
    127.0.0.1/::1) — the one legitimate case in this topology, since uvicorn
    is loopback-only and the firewall blocks direct external access to its
    port. If a request instead arrives from a NON-loopback peer while
    TRUST_PROXY_HEADERS is on, something bypassed nginx (or nginx is
    misconfigured) and reached uvicorn directly — X-Forwarded-For is then
    attacker-controlled with no proxy in between to have appended a real
    address, so it must be ignored and request.client.host used as-is.

    deploy/mtls/nginx-antispoof-mtls.conf sets
    `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`. nginx's
    $proxy_add_x_forwarded_for APPENDS $remote_addr (the address nginx
    itself saw on the TCP connection) to whatever X-Forwarded-For value the
    client already sent — it never replaces the header. With exactly one
    trusted hop (nginx is the public edge here; nothing else sits in front
    of it), the LAST entry in the resulting header is always the one nginx
    itself wrote from the real TCP peer; every entry before it is
    attacker-supplied input nginx merely passed through unchanged. Trusting
    the FIRST entry instead would let anyone bypass the allowlist by simply
    prepending a fake "trusted" IP to their own X-Forwarded-For request
    header before it ever reaches nginx.
    """
    raw_ip = request.client.host if request.client else "0.0.0.0"

    if not settings.TRUST_PROXY_HEADERS:
        return raw_ip

    try:
        parsed = ipaddress.ip_address(raw_ip)
    except ValueError:
        return raw_ip  # non-IP peer (e.g. TestClient) — nothing to trust-upgrade

    if not parsed.is_loopback:
        return raw_ip  # direct connection bypassing nginx — never trust X-Forwarded-For

    xff = request.headers.get("x-forwarded-for", "")
    hops = [hop.strip() for hop in xff.split(",") if hop.strip()]
    if not hops:
        return raw_ip

    real_client = hops[-1]
    try:
        ipaddress.ip_address(real_client)
    except ValueError:
        return raw_ip  # malformed header — fail back to the (loopback) peer address

    return real_client

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
    # NOTE (P0-1, 2026-07-18): this generic handler is a fallback for
    # exceptions that happen BEFORE the request body is parsed (e.g.
    # malformed JSON) — session_id/correlation_id/transaction_type/
    # transaction_ref are genuinely unavailable here, hence null. Once the
    # body IS parsed, the route handler's own local try/except
    # (app/main.py::liveness_verdict) takes over and echoes those fields —
    # this branch should rarely fire in practice.
    if request.url.path == "/liveness/verdict":
        return JSONResponse(
            status_code=500,
            content={
                "verdict": "incomplete",
                "reason": "INTERNAL_ERROR",
                "model_version": LIVENESS_MODEL_VERSION,
                "frame_consistency_score": -1.0,
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
    raw_ip = _effective_client_ip(request)
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
    engine = LivenessEngine(
        settings.MODEL_DIR, DEVICE,
        print_pattern_override_enabled=settings.PRINT_PATTERN_OVERRIDE_ENABLED,
    )
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


def _run_resolution_gate(image_bgr: np.ndarray, byte_size: int) -> Optional[ResolutionCheckResult]:
    """Layer 0e — shared image resolution/weight gate (RZA, 2026-07-21).

    Unlike every other Layer 0 gate in this file, this one needs NO detected
    face/bbox at all — pure arithmetic on the decoded image's width/height
    and the raw upload's byte size — so callers should run it as early as
    possible (right after decode + `_validate_image_dimensions`), before
    face detection, to reject a too-small/re-compressed frame at the lowest
    possible cost. See app/resolution_check.py for the full rationale
    (a 199-file Telegram-preview calibration dataset that was unusable
    because every file was a re-encoded thumbnail) and the numbers behind
    every threshold.

    Returns the `ResolutionCheckResult` when the gate fired (caller should
    short-circuit to its own low_quality response), or `None` when disabled,
    did not run (bad input — should not happen with a real decoded image),
    or did not flag the frame (caller falls through unchanged). Never
    raises.
    """
    if not settings.RESOLUTION_CHECK_ENABLED:
        return None
    h, w = image_bgr.shape[:2]
    result = check_image_resolution(
        w, h, byte_size,
        settings.MIN_IMAGE_MIN_SIDE_PX, settings.MIN_IMAGE_MEGAPIXELS, settings.MIN_IMAGE_BYTES,
    )
    if result.ran and result.is_low_resolution:
        return result
    return None


def _resolution_signals(result: ResolutionCheckResult) -> dict:
    """Build the `signals` sub-dict shape used across endpoints for a resolution-gate hit."""
    return {
        "resolution_check": {
            "width": result.width,
            "height": result.height,
            "min_side": result.min_side,
            "megapixels": result.megapixels,
            "byte_size": result.byte_size,
            "fired": result.reason,
            "min_side_threshold": settings.MIN_IMAGE_MIN_SIDE_PX,
            "megapixels_threshold": settings.MIN_IMAGE_MEGAPIXELS,
            "bytes_threshold": settings.MIN_IMAGE_BYTES,
        }
    }


def _run_aspect_ratio_gate(image_bgr: np.ndarray) -> Optional[AspectRatioCheckResult]:
    """Layer 0g — shared camera-aspect-ratio gate (RZA, 2026-07-21).

    Bbox-independent (no face detection needed, same shape as
    _run_resolution_gate) — see app/aspect_ratio_check.py for the full
    rationale (a real confirmed-fraud sample and 174/199 of
    faces-dataset/'s files sit at a 9:16 "screen" ratio a phone camera
    still-photo never produces) and the numbers behind every threshold.

    Returns the `AspectRatioCheckResult` when the gate fired (caller should
    short-circuit to its own low_quality response), or `None` when
    disabled, did not run (bad input), or did not flag the frame (caller
    falls through unchanged). Never raises.
    """
    if not settings.ASPECT_RATIO_CHECK_ENABLED:
        return None
    h, w = image_bgr.shape[:2]
    result = check_aspect_ratio(w, h, settings.ASPECT_RATIO_MIN, settings.ASPECT_RATIO_MAX)
    if result.ran and result.is_non_camera_geometry:
        return result
    return None


def _aspect_ratio_signals(result: AspectRatioCheckResult) -> dict:
    """Build the `signals` sub-dict shape used across endpoints for an aspect-ratio-gate hit."""
    return {
        "aspect_ratio_check": {
            "width": result.width,
            "height": result.height,
            "ratio": result.ratio,
            "min_ratio": settings.ASPECT_RATIO_MIN,
            "max_ratio": settings.ASPECT_RATIO_MAX,
        }
    }


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


def _run_sharpness_gate(bbox: list[int], image_bgr: np.ndarray) -> Optional[SharpnessCheckResult]:
    """Layer 0c — shared frame-sharpness gate (RZA, 2026-07-21).

    Reuses the SAME bbox already computed for passive-PAD/geometry — no
    extra model. See app/blur_check.py for the full rationale (blur is one
    half of the reported angle+blur bypass) and calibration numbers.

    Returns the `SharpnessCheckResult` when the gate fired (caller should
    short-circuit to its own low_quality/BLURRY response), or `None` when
    disabled, did not run (bad input), or did not flag the frame (caller
    falls through unchanged). Never raises.
    """
    if not settings.FRAME_SHARPNESS_CHECK_ENABLED:
        return None
    result = check_face_sharpness(bbox, image_bgr, settings.MIN_FACE_SHARPNESS_224)
    if result.ran and result.is_blurry:
        return result
    return None


def _sharpness_signals(result: SharpnessCheckResult) -> dict:
    """Build the `signals` sub-dict shape used across endpoints for a sharpness-gate hit."""
    return {"sharpness_check": {"sharpness": result.sharpness, "threshold": settings.MIN_FACE_SHARPNESS_224}}


def _run_pose_gate(image_bgr: np.ndarray) -> Optional[PoseCheckResult]:
    """Layer 0d — face-angle gate (RZA, 2026-07-21). See app/pose_check.py
    for the full rationale and calibration numbers.

    Fails open (returns None) unless BOTH `settings.POSE_CHECK_ENABLED` AND
    the landmark_detector singleton actually loaded (requires
    `LIVENESS_ENDPOINTS_ENABLED=True` at startup, app/main.py::
    _load_liveness_models) — see app/pose_check.py's "DEFAULT DISABLED"
    limitation for why this is a silent no-op, not a security control, when
    either precondition is missing. Any exception from the detector itself
    (e.g. a corrupt/edge-case frame) is caught and logged, never raised —
    same fail-safe-to-passive-PAD pattern as the geometry/sharpness gates,
    which are dependency-free and therefore cannot fail this way.
    """
    if not settings.POSE_CHECK_ENABLED or not _liveness_models_loaded or landmark_detector is None:
        return None
    try:
        face = landmark_detector.analyze(image_bgr)
    except Exception:
        log.exception("pose gate: landmark_detector.analyze() failed — failing open to passive-PAD")
        return None
    if face is None:
        return None
    result = check_face_pose(
        face.pose_yaw, face.pose_pitch,
        settings.POSE_YAW_REJECT_DEG, settings.POSE_PITCH_REJECT_DEG,
    )
    if result.ran and result.is_off_angle:
        return result
    return None


def _pose_signals(result: PoseCheckResult) -> dict:
    """Build the `signals` sub-dict shape used across endpoints for a pose-gate hit."""
    return {
        "pose_check": {
            "pose_yaw": result.pose_yaw,
            "pose_pitch": result.pose_pitch,
            "yaw_threshold": settings.POSE_YAW_REJECT_DEG,
            "pitch_threshold": settings.POSE_PITCH_REJECT_DEG,
        }
    }


def _run_edge_sharpness_diagnostic(image_bgr: np.ndarray) -> Optional[dict]:
    """Layer 0f — edge-vs-center sharpness DIAGNOSTIC (RZA, 2026-07-21).

    NOT a gate — see app/edge_sharpness_check.py's module docstring for why
    (an asymmetric-edge-blur hypothesis was tested against a real bona fide
    counter-example the same day and did not hold up). Returns a `signals`
    sub-dict to merge into the response when `EDGE_SHARPNESS_DIAGNOSTIC_
    ENABLED=True` and the measurement ran successfully, or `None` when
    disabled or the measurement failed (bad input) — the caller must NEVER
    branch on this to change `verdict`, only attach it as extra data.
    """
    if not settings.EDGE_SHARPNESS_DIAGNOSTIC_ENABLED:
        return None
    result = measure_edge_sharpness(image_bgr, settings.EDGE_SHARPNESS_EDGE_FRACTION)
    if not result.ran:
        return None
    return {
        "edge_sharpness_diagnostic": {
            "left_sharpness": result.left_sharpness,
            "right_sharpness": result.right_sharpness,
            "center_sharpness": result.center_sharpness,
            "left_to_center_ratio": result.left_to_center_ratio,
            "right_to_center_ratio": result.right_to_center_ratio,
            "min_edge_to_center_ratio": result.min_edge_to_center_ratio,
            "note": "DIAGNOSTIC ONLY, not wired into verdict — see app/edge_sharpness_check.py",
        }
    }


def _run_single(image_bgr: np.ndarray, byte_size: int = 0) -> dict:
    """Detect face + (Layer 0a geometry gate) + predict liveness for a single image.

    `byte_size` (raw upload size in bytes, 0 if the caller has none to give —
    e.g. a not-yet-updated call site) feeds the Layer 0e resolution/weight
    gate below; a caller passing 0 with `RESOLUTION_CHECK_ENABLED=True` would
    always fail the byte-size sub-check, so every call site in this file
    passes the real `len(...)` of the decoded bytes — see app/resolution_check.py.
    """
    reso_result = _run_resolution_gate(image_bgr, byte_size)
    if reso_result is not None:
        return {
            "is_real": False,
            "label": "low_quality",
            "score": reso_result.megapixels,
            "threshold": settings.MIN_IMAGE_MEGAPIXELS,
            "face_detected": False,
            "signals": _resolution_signals(reso_result),
        }

    aspect_result = _run_aspect_ratio_gate(image_bgr)
    if aspect_result is not None:
        return {
            "is_real": False,
            "label": "non_camera_geometry",
            "score": aspect_result.ratio,
            "threshold": settings.ASPECT_RATIO_MIN,
            "face_detected": False,
            "signals": _aspect_ratio_signals(aspect_result),
        }

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

    sharp_result = _run_sharpness_gate(bbox, image_bgr)
    if sharp_result is not None:
        return {
            "is_real": False,
            "label": "blurry",
            "score": round(sharp_result.sharpness, 4),
            "threshold": settings.MIN_FACE_SHARPNESS_224,
            "face_detected": True,
            "signals": _sharpness_signals(sharp_result),
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
    result = _run_single(img, len(data))
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

    decoded: list[tuple[np.ndarray, Optional[list[int]], str, Optional[ResolutionCheckResult], Optional[AspectRatioCheckResult]]] = []
    for upload in images:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "not an image", None, None))
            continue
        data = await upload.read()
        if not data:
            decoded.append((np.zeros((1, 1, 3), dtype=np.uint8), None, "empty file", None, None))
            continue
        _validate_image_size(data)
        img = _read_image(data)
        _validate_image_dimensions(img)
        # Layer 0e resolution/weight gate — bbox-independent (see
        # app/resolution_check.py), so it runs BEFORE face detection here to
        # skip that cost entirely on a frame this gate would reject anyway.
        reso_result = _run_resolution_gate(img, len(data))
        if reso_result is not None:
            decoded.append((img, None, "", reso_result, None))
            continue
        # Layer 0g aspect-ratio gate — same bbox-independent shape (see
        # app/aspect_ratio_check.py), runs right after the resolution gate,
        # still before face detection.
        aspect_result = _run_aspect_ratio_gate(img)
        if aspect_result is not None:
            decoded.append((img, None, "", None, aspect_result))
            continue
        bbox = detector.detect(img)
        decoded.append((img, bbox, "", None, None))

    crops: list[np.ndarray] = []
    crop_face_px: list[int] = []
    crop_indices: list[int] = []
    results: list[dict] = [{}] * len(decoded)

    for i, (img, bbox, err, reso_result, aspect_result) in enumerate(decoded):
        if err:
            results[i] = {"is_real": False, "label": "no_face", "score": 0.0,
                           "threshold": settings.LIVENESS_THRESHOLD,
                           "face_detected": False, "error": err}
        elif reso_result is not None:
            results[i] = {
                "is_real": False,
                "label": "low_quality",
                "score": reso_result.megapixels,
                "threshold": settings.MIN_IMAGE_MEGAPIXELS,
                "face_detected": False,
                "signals": _resolution_signals(reso_result),
            }
        elif aspect_result is not None:
            results[i] = {
                "is_real": False,
                "label": "non_camera_geometry",
                "score": aspect_result.ratio,
                "threshold": settings.ASPECT_RATIO_MIN,
                "face_detected": False,
                "signals": _aspect_ratio_signals(aspect_result),
            }
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

            sharp_result = _run_sharpness_gate(bbox, img)
            if sharp_result is not None:
                results[i] = {
                    "is_real": False,
                    "label": "blurry",
                    "score": round(sharp_result.sharpness, 4),
                    "threshold": settings.MIN_FACE_SHARPNESS_224,
                    "face_detected": True,
                    "signals": _sharpness_signals(sharp_result),
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

    result = _run_single(img, len(img_bytes))

    elapsed = round(time.perf_counter() - t0, 3)
    is_spoof = 0 if result["is_real"] else 1

    # Map to verdict enum per FACEID_PHASE1_PAD_GATE contract (§1).
    # IDENTICAL logic to /pad/check (app/main.py ~744-752) to prevent divergence.
    # This is an ADDITIVE field — existing consumers reading only
    # elapsed_time/is_spoof are unaffected (same values as before).
    label = result.get("label", "unknown")
    score = result.get("score", 0.0)

    # Explicit per-label verdict mapping (NOT via is_real boolean, and NOT a
    # bare score-vs-threshold fallback for every gate label — see fix below).
    # This prevents the regression where label="real" + score<threshold →
    # spoof. RZA, 2026-07-21: also fixes a real pre-existing bug this pass
    # found while wiring in the new "low_quality" (resolution-gate) label —
    # `check_face_sharpness`'s "blurry" label carries `score=sharpness`
    # (a Laplacian-variance value, typically tens-to-hundreds, e.g. 45.0),
    # which is virtually NEVER `< settings.LIVENESS_THRESHOLD` (0.5) — the
    # OLD fallback-only mapping below would have silently fallen through to
    # `verdict="live"` for a blur-gate hit the moment FRAME_SHARPNESS_CHECK_
    # ENABLED is turned on, defeating that gate specifically in THIS endpoint
    # (currently dormant/unexercised only because that flag still defaults
    # False). Same class of bug would have hit the NEW resolution gate's
    # "low_quality" label too (`score=megapixels`, can sit above 0.5 while
    # still below MIN_IMAGE_MEGAPIXELS) — both are now explicit branches
    # instead of relying on the numeric fallback.
    if label == "document_photo":
        verdict = "spoof"
        reason: Optional[str] = "DOCUMENT_PHOTO"
    elif label == "no_face":
        verdict = "low_quality"
        reason = "NO_FACE"
    elif label == "spoof":
        verdict = "spoof"
        reason = "PASSIVE_PAD_SPOOF"
    elif label == "blurry":
        verdict = "low_quality"
        reason = "BLURRY"
    elif label == "low_quality":
        # Layer 0e resolution/weight gate (app/resolution_check.py) — see
        # _run_single's own low_quality branch.
        verdict = "low_quality"
        reason = "LOW_RESOLUTION"
    elif label == "non_camera_geometry":
        # Layer 0g aspect-ratio gate (app/aspect_ratio_check.py) — see
        # _run_single's own non_camera_geometry branch. Same numeric-
        # fallback trap as BLURRY/LOW_RESOLUTION above: `score=ratio` (e.g.
        # 0.56 for a 9:16 frame) sits ABOVE LIVENESS_THRESHOLD (0.5) far
        # more often than not, so this needs its own explicit branch too.
        verdict = "low_quality"
        reason = "NON_CAMERA_GEOMETRY"
    elif score < settings.LIVENESS_THRESHOLD:
        # Real face but low passive-PAD score (bad lighting, occlusion,
        # etc.) → low quality. Only reached for labels NOT already handled
        # above, so this numeric comparison never sees a non-passive-PAD
        # score again.
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
    # NEW (RZA, 2026-07-20), OPTIONAL — backward compatible, a caller not
    # sending these fields is unaffected: dedup-by-phash (app/dedup_store.py)
    # does NOT need them (works on the frame alone), and the inspector/
    # abonent fraud-pattern alert is a complete no-op when either is absent.
    # `abonent_kod`/`pinfl` are explicitly OUT per the original PAD_GATE
    # contract's own transaction_ref wording ("NOT pinfl, NOT abonent_kod")
    # — an internal numeric/opaque id is expected here, not PII, matching
    # the existing correlation_id/transaction_ref privacy posture.
    abonent_id: Optional[str] = Field(None, description="Opaque abonent identifier, for fraud-pattern alerting only")
    inspector_id: Optional[str] = Field(None, description="Opaque inspector identifier, for fraud-pattern alerting only")


class PadCheckResponse(BaseModel):
    """PAD-gate response — contract per FACEID_PHASE1_PAD_GATE.md §1."""
    verdict: Literal["live", "spoof", "low_quality"] = Field(...)
    reason: Optional[str] = Field(
        None,
        description=(
            "PASSIVE_PAD_SPOOF | PRINT_PATTERN_SPOOF | DOCUMENT_PHOTO | BLURRY | OFF_ANGLE | "
            "LOW_RESOLUTION | NON_CAMERA_GEOMETRY | DUPLICATE_PHOTO | NO_FACE | LOW_QUALITY | "
            "TIMEOUT | INTERNAL_ERROR | null"
        ),
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
    # P0-2 (2026-07-18): constant-time comparison — a `!=` string compare
    # short-circuits on the first mismatched byte, which is a timing side
    # channel an attacker could use to guess SERVICE_TOKEN byte-by-byte.
    if not x_service_token or not hmac.compare_digest(x_service_token, settings.SERVICE_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Service-Token")


def _verify_replay_protection(x_request_timestamp: Optional[str]) -> None:
    """Anti-replay guard — bounded timestamp window layered ON TOP OF mTLS
    (deploy/mtls/, BUSTA RHYMES) + the existing X-Service-Token/IP-allowlist
    (both unchanged by this). mTLS authenticates the channel ("who is
    talking"); it does not stop a captured request from being replayed
    verbatim within its validity window — this closes that specific gap for
    the three money-path endpoints (/pad/check, /liveness/challenge,
    /liveness/verdict) WITHOUT a nonce-store or any new infrastructure, per
    explicit owner decision (KENDRICK security analysis, 2026-07-18).

    Deliberately coarse: this does not detect replay of the SAME request
    within the tolerance window (no nonce/dedup), it only bounds how long a
    captured request stays replayable at all.

    DISABLED BY DEFAULT (settings.REPLAY_PROTECTION_ENABLED=False) — the
    partner must start sending X-Request-Timestamp (unix seconds) on every
    money-path call BEFORE this flips on in prod, or their existing traffic
    starts failing 401. Mirrors the empty-SERVICE_TOKEN dev/prod pattern
    above: a settings flag gates the check, not a hardcoded constant.
    """
    if not settings.REPLAY_PROTECTION_ENABLED:
        return
    if not x_request_timestamp:
        raise HTTPException(status_code=401, detail="Missing X-Request-Timestamp")
    try:
        request_ts = float(x_request_timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid X-Request-Timestamp")
    # 2PAC (2026-07-18): `float("nan")` parses successfully (valid IEEE754),
    # and abs(now - nan) is nan — every comparison against nan (including
    # `nan > REPLAY_TOLERANCE_S`) is False per IEEE754, so the `>` check
    # below silently never raises for "nan"/"NaN"/"+nan"/"-nan" (any case).
    # That was a full bypass of this entire guard. math.isfinite() rejects
    # nan AND +-inf explicitly, rather than relying on comparison fallout.
    if not math.isfinite(request_ts):
        raise HTTPException(status_code=401, detail="Invalid X-Request-Timestamp")
    if abs(time.time() - request_ts) > settings.REPLAY_TOLERANCE_S:
        raise HTTPException(status_code=401, detail="X-Request-Timestamp outside allowed window")


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
    x_request_timestamp: Optional[str] = Header(None, alias="X-Request-Timestamp"),
) -> PadCheckResponse:
    """PAD-gate: classify a face frame as live/spoof/low_quality.

    Called by Laravel after Adliya match (BACKEND_REQUIREMENTS_2026-07-06 п.8).
    Contract: FACEID_PHASE1_PAD_GATE.md §1.
    """
    _verify_service_token(x_service_token)
    _verify_replay_protection(x_request_timestamp)

    t0 = time.perf_counter()

    # Decode + validate
    try:
        img_bytes = base64.b64decode(req.face_photo)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 in face_photo")

    _validate_image_size(img_bytes)
    img = _read_image(img_bytes)
    _validate_image_dimensions(img)

    # Layer 0e — resolution/weight gate (RZA, 2026-07-21, DEFAULT DISABLED —
    # see Settings.RESOLUTION_CHECK_ENABLED). Bbox-independent (see
    # app/resolution_check.py) — deliberately the EARLIEST gate in this
    # function, ahead of even dedup below: a too-small/re-compressed frame
    # is rejected before spending compute on phash/face-detection/anything
    # else. verdict=low_quality (not spoof) — same "reshoot, don't accuse"
    # posture as the blur/pose gates, a small image alone is not
    # independently confirmed fraud.
    reso_result = _run_resolution_gate(img, len(img_bytes))
    if reso_result is not None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        reso_signals = _resolution_signals(reso_result)
        _audit_entry(
            req.correlation_id, req.transaction_type, req.transaction_ref,
            "low_quality", reso_result.megapixels, reso_signals, ms, False, reason="LOW_RESOLUTION",
        )
        log.info(
            "PAD check: correlation=%s verdict=low_quality reason=LOW_RESOLUTION "
            "width=%d height=%d megapixels=%.3f byte_size=%d fired=%s ms=%.1f txn=%s",
            req.correlation_id, reso_result.width, reso_result.height, reso_result.megapixels,
            reso_result.byte_size, reso_result.reason, ms, req.transaction_ref,
        )
        return PadCheckResponse(
            verdict="low_quality", reason="LOW_RESOLUTION", score=round(reso_result.megapixels, 4),
            threshold=settings.MIN_IMAGE_MEGAPIXELS, face_detected=False,
            save_frame=False, signals=reso_signals, model_version=MODEL_VERSION, processing_ms=ms,
        )
    # Disabled, bad input, or did not flag the frame — continue unchanged.

    # Layer 0g — camera-aspect-ratio gate (RZA, 2026-07-21, DEFAULT DISABLED
    # — see Settings.ASPECT_RATIO_CHECK_ENABLED). Bbox-independent (see
    # app/aspect_ratio_check.py) — right after the resolution gate, still
    # ahead of dedup below: a real confirmed-fraud sample (real_fake_01.jpg)
    # and 174/199 of faces-dataset/'s Telegram-preview files share a 9:16
    # ("screen") aspect a phone camera still-photo never produces.
    # verdict=low_quality (NOT spoof) — same "reshoot, don't accuse" posture
    # as every other Layer 0 gate: a wrong-aspect frame alone is not
    # independently confirmed fraud (a legitimate integration bug or
    # non-standard device could also produce this) — see the module
    # docstring's own limitation #1 (trivially defeated by cropping the fake
    # to 3:4 first — this is a cheap first layer, not a final defense).
    aspect_result = _run_aspect_ratio_gate(img)
    if aspect_result is not None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        aspect_signals = _aspect_ratio_signals(aspect_result)
        _audit_entry(
            req.correlation_id, req.transaction_type, req.transaction_ref,
            "low_quality", aspect_result.ratio, aspect_signals, ms, False, reason="NON_CAMERA_GEOMETRY",
        )
        log.info(
            "PAD check: correlation=%s verdict=low_quality reason=NON_CAMERA_GEOMETRY "
            "width=%d height=%d ratio=%.4f ms=%.1f txn=%s",
            req.correlation_id, aspect_result.width, aspect_result.height, aspect_result.ratio,
            ms, req.transaction_ref,
        )
        return PadCheckResponse(
            verdict="low_quality", reason="NON_CAMERA_GEOMETRY", score=round(aspect_result.ratio, 4),
            threshold=settings.ASPECT_RATIO_MIN, face_detected=False,
            save_frame=False, signals=aspect_signals, model_version=MODEL_VERSION, processing_ms=ms,
        )
    # Disabled, bad input, or did not flag the frame — continue unchanged.

    # --- Frame-reuse dedup (HARD BLOCK) — runs next, before the models-ready
    # check, face detection, or any other gate below: this is a pure
    # image-hash comparison against prior /pad/check frames, independent of
    # whether models are loaded. See app/dedup_store.py module docstring —
    # built in direct response to a real production fraud incident (2026-07-20): the
    # SAME photo accepted for TWO DIFFERENT abonents, 46s apart, one
    # inspector. DEFAULT DISABLED (settings.DEDUP_ENABLED, app/config.py) —
    # see that flag's docstring for why (no real duplicate-photo corpus yet
    # to verify the Hamming threshold against, and flipping the default on
    # would break the existing test suite's shared fixture image). Zero-cost,
    # zero-DB-write no-op when disabled.
    dedup_phash = compute_phash(img)
    phash_recorded = False
    if settings.DEDUP_ENABLED:
        dedup_match = dedup_store.check_and_record_phash(
            dedup_phash, settings.DEDUP_PHASH_HAMMING_MAX,
            req.correlation_id, req.transaction_ref, req.abonent_id, req.inspector_id,
        )
        phash_recorded = True
        if dedup_match is not None:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            similarity = round(1.0 - dedup_match.hamming_distance / 64.0, 4)
            dedup_signals = {
                "dedup_check": {
                    "phash_match": True,
                    "hamming_distance": dedup_match.hamming_distance,
                    "matched_correlation_id": dedup_match.correlation_id,
                    "matched_transaction_ref": dedup_match.transaction_ref,
                    "matched_age_s": dedup_match.age_s,
                },
                # See docs/plans/HANDOFF-2026-07-21-cross-transaction-face-reuse.md
                # — same additive field as the normal-verdict response path.
                "image_phash": dedup_phash,
            }
            _audit_entry(
                req.correlation_id, req.transaction_type, req.transaction_ref,
                "spoof", similarity, dedup_signals, ms, True, reason="DUPLICATE_PHOTO",
            )
            log.warning(
                "PAD check: correlation=%s verdict=spoof reason=DUPLICATE_PHOTO "
                "matched_correlation_id=%s matched_transaction_ref=%s hamming=%d ms=%.1f txn=%s",
                req.correlation_id, dedup_match.correlation_id, dedup_match.transaction_ref,
                dedup_match.hamming_distance, ms, req.transaction_ref,
            )
            return PadCheckResponse(
                verdict="spoof", reason="DUPLICATE_PHOTO", score=similarity,
                threshold=round(1.0 - settings.DEDUP_PHASH_HAMMING_MAX / 64.0, 4), face_detected=False,
                save_frame=True, signals=dedup_signals, model_version=MODEL_VERSION, processing_ms=ms,
            )
    # DEDUP_ENABLED=False (or no match found) — continue unchanged.
    # `dedup_phash`/`phash_recorded` are kept for the AdaFace-embedding-alert
    # path further below.

    # --- Inspector/abonent fraud-pattern heuristic — SOFT, log-only, NEVER
    # blocks a verdict. See app/dedup_store.py module docstring §3. Complete
    # no-op unless the caller sends BOTH new optional fields (backward
    # compatible — no existing caller sends them today).
    fraud_signal: dict = {}
    if settings.FRAUD_INSPECTOR_ALERT_ENABLED and req.abonent_id and req.inspector_id:
        dedup_store.record_inspector_activity(
            req.inspector_id, req.abonent_id, req.correlation_id, req.transaction_ref,
        )
        fraud_alert = dedup_store.check_inspector_fraud_alert(
            req.inspector_id, settings.FRAUD_INSPECTOR_WINDOW_S, settings.FRAUD_INSPECTOR_DISTINCT_ABONENT_MAX,
        )
        if fraud_alert is not None:
            fraud_signal = {
                "fraud_alert": {
                    "type": "INSPECTOR_MULTI_ABONENT",
                    "inspector_id": fraud_alert.inspector_id,
                    "distinct_abonent_count": fraud_alert.distinct_abonent_count,
                    "window_s": fraud_alert.window_s,
                    "abonent_ids": fraud_alert.abonent_ids,
                }
            }
            log.warning(
                "PAD check: FRAUD ALERT inspector_id=%s distinct_abonents=%d window_s=%.0f correlation=%s txn=%s",
                fraud_alert.inspector_id, fraud_alert.distinct_abonent_count,
                fraud_alert.window_s, req.correlation_id, req.transaction_ref,
            )

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

    # Layer 0c — deterministic frame-sharpness gate (RZA, 2026-07-21). Runs
    # BEFORE passive-PAD, same bbox, no extra model — see app/blur_check.py.
    # verdict=low_quality (not spoof): a blurry SINGLE frame alone is not
    # independently confirmed as a fraud attempt (an honest customer's shaky
    # camera looks the same to this gate) — same "reshoot, don't accuse"
    # posture app/frame_qc.py already uses for the multi-frame session path.
    # Blocking the "live" verdict is what actually matters for the money
    # path: a blurry attack frame can never reach verdict=live through here.
    sharp_result = _run_sharpness_gate(bbox, img)
    if sharp_result is not None:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        sharp_signals = _sharpness_signals(sharp_result)
        _audit_entry(
            req.correlation_id, req.transaction_type, req.transaction_ref,
            "low_quality", sharp_result.sharpness, sharp_signals, ms, False, reason="BLURRY",
        )
        log.info(
            "PAD check: correlation=%s verdict=low_quality reason=BLURRY "
            "sharpness=%.1f ms=%.1f txn=%s",
            req.correlation_id, sharp_result.sharpness, ms, req.transaction_ref,
        )
        return PadCheckResponse(
            verdict="low_quality", reason="BLURRY", score=round(sharp_result.sharpness, 4),
            threshold=settings.MIN_FACE_SHARPNESS_224, face_detected=True,
            save_frame=False, signals=sharp_signals, model_version=MODEL_VERSION, processing_ms=ms,
        )
    # Below threshold, not ran (disabled or bad input) — continue unchanged.

    # Layer 0d — face-angle gate (RZA, 2026-07-21). DEFAULT DISABLED
    # (settings.POSE_CHECK_ENABLED) — see app/pose_check.py for why. When
    # enabled, runs a SECOND detector (SCRFD+landmark_3d_68) under its own
    # bounded timeout so a slow/stuck pose pass cannot blow past
    # INFERENCE_TIMEOUT_S's 2.0s budget for the passive-PAD call that follows
    # it; a timeout here fails OPEN (falls through to passive-PAD unchanged,
    # same as a disabled/no-landmark result), it never turns into an error
    # response — this gate is additive hardening, not a new failure mode.
    # verdict=low_quality (not spoof), same reasoning as the sharpness gate
    # above: an off-angle frame alone is not independently confirmed fraud.
    if settings.POSE_CHECK_ENABLED:
        try:
            pose_result = await asyncio.wait_for(
                asyncio.to_thread(_run_pose_gate, img), timeout=INFERENCE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning("pose gate: timeout for correlation=%s — failing open to passive-PAD", req.correlation_id)
            pose_result = None
        if pose_result is not None:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            pose_signals = _pose_signals(pose_result)
            _audit_entry(
                req.correlation_id, req.transaction_type, req.transaction_ref,
                "low_quality", 0.0, pose_signals, ms, False, reason="OFF_ANGLE",
            )
            log.info(
                "PAD check: correlation=%s verdict=low_quality reason=OFF_ANGLE "
                "yaw=%.1f pitch=%.1f ms=%.1f txn=%s",
                req.correlation_id, pose_result.pose_yaw, pose_result.pose_pitch, ms, req.transaction_ref,
            )
            return PadCheckResponse(
                verdict="low_quality", reason="OFF_ANGLE", score=0.0,
                threshold=settings.POSE_YAW_REJECT_DEG, face_detected=True,
                save_frame=False, signals=pose_signals, model_version=MODEL_VERSION, processing_ms=ms,
            )
    # Below threshold, disabled, or landmark_detector unavailable — continue unchanged.

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

    # Map to PAD-gate verdict (§1) — verdict/threshold logic unchanged, only naming aligned to spec.
    # reason is PRINT_PATTERN_SPOOF vs PASSIVE_PAD_SPOOF depending on which
    # override in _fuse() fired — see app/liveness.py::pad_check_reason
    # docstring (2026-07-22, 2PAC review round 2: Умид needs to filter false
    # rejects by signal, a single shared reason hid the new override).
    if label == "spoof":
        verdict = "spoof"
        reason: Optional[str] = pad_check_reason(label, signal_info)
    elif score < settings.LIVENESS_THRESHOLD:
        verdict = "low_quality"
        reason = "LOW_QUALITY"
    else:
        verdict = "live"
        reason = None

    # --- AdaFace-embedding-based dedup ALERT — NEVER blocks, see
    # app/dedup_store.py module docstring §2 for why this is deliberately
    # weaker than the reviewed spec's proposal (a same-person match across
    # two different transaction_refs is the EXPECTED case for a repeat
    # customer, not fraud). Requires the Phase-2 SCRFD+AdaFace models
    # already loaded (LIVENESS_ENDPOINTS_ENABLED) — see
    # DEDUP_EMBEDDING_ALERT_ENABLED docstring (app/config.py) for the
    # latency-budget reasoning this is gated on both flags and defaults off.
    # Any failure here is swallowed — this is a soft alert, never worth
    # failing the request over.
    if settings.DEDUP_EMBEDDING_ALERT_ENABLED and settings.LIVENESS_ENDPOINTS_ENABLED \
            and _liveness_models_loaded and landmark_detector is not None and adaface_embedder is not None:
        try:
            from app.face_landmarks import LandmarkDetector
            face = landmark_detector.analyze(img)
            if face is not None:
                aligned = LandmarkDetector.align_112(img, face.kps)
                embedding = adaface_embedder.embed_aligned(aligned)
                if not phash_recorded:
                    # DEDUP_ENABLED=False but the embedding layer still needs
                    # a row to attach to (and to be matchable by FUTURE
                    # requests) — record-only, result intentionally
                    # discarded: this flag alone never triggers the hard
                    # DUPLICATE_PHOTO block above.
                    dedup_store.check_and_record_phash(
                        dedup_phash, settings.DEDUP_PHASH_HAMMING_MAX,
                        req.correlation_id, req.transaction_ref, req.abonent_id, req.inspector_id,
                    )
                    phash_recorded = True
                dedup_store.record_embedding(req.correlation_id, embedding)
                embedding_matches = dedup_store.check_embedding_alert(
                    embedding, settings.DEDUP_EMBEDDING_COSINE_ALERT, req.transaction_ref,
                    exclude_abonent_id=req.abonent_id,
                )
                if embedding_matches:
                    signal_info = dict(signal_info)
                    signal_info["dedup_embedding_alert"] = [
                        {
                            "correlation_id": m.correlation_id, "transaction_ref": m.transaction_ref,
                            "abonent_id": m.abonent_id, "inspector_id": m.inspector_id,
                            "cosine_similarity": m.cosine_similarity, "age_s": m.age_s,
                        }
                        for m in embedding_matches
                    ]
        except Exception:
            log.exception(
                "dedup embedding-alert failed for correlation=%s — continuing without it "
                "(alert-only, never blocking)", req.correlation_id,
            )

    if fraud_signal:
        signal_info = {**signal_info, **fraud_signal}

    # Cross-transaction reuse handoff (RZA, 2026-07-21) — see
    # docs/plans/HANDOFF-2026-07-21-cross-transaction-face-reuse.md. This
    # service's own DEDUP_ENABLED only compares against its OWN short-lived
    # SQLite window (DEDUP_TTL_DAYS) and has no idea which abonent_id a
    # frame belongs to across the caller's full history — Laravel does.
    # `dedup_phash` is already computed above UNCONDITIONALLY (cheap, no
    # extra model) regardless of DEDUP_ENABLED; surfacing it here, on every
    # normal response, costs nothing extra and lets the caller persist it
    # against its own abonent/transaction records for a server-side
    # "same photo hash across N different abonents" rule with a much wider
    # window than this service could reasonably own.
    signal_info = {**signal_info, "image_phash": dedup_phash}

    # Layer 0f — edge-vs-center sharpness DIAGNOSTIC (RZA, 2026-07-21,
    # DEFAULT DISABLED). NEVER changes `verdict` — see
    # _run_edge_sharpness_diagnostic's own docstring and
    # app/edge_sharpness_check.py for why this stays diagnostic-only.
    edge_signal = _run_edge_sharpness_diagnostic(img)
    if edge_signal is not None:
        signal_info = {**signal_info, **edge_signal}

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

# Фаза 4 — quality_certified startup guard (CHALLENGE_ENTROPY_SPRINT_v1.md
# §7.3, требование Рустама §1 п.2: "quality_certified допустим только при
# числовых целях APCER/BPCER"). LIVENESS_QUALITY_CERTIFIED — РУЧНОЙ флаг
# (app/config.py, НЕ авто-вычисляемый ни из какой метрики в этом коде) —
# если он включён, задеплоенный model_version ОБЯЗАН совпадать с тем, на
# котором был прогнан сертифицирующий CALIBRATION_REPORT.md
# (LIVENESS_CERTIFIED_MODEL_VERSION) — иначе это "сертифицировали одну
# версию, задеплоили другую". WARNING-only (не хардфейл), тот же посыл, что
# и у dev-режима пустого SERVICE_TOKEN выше: quality_certified — это
# ЗАЯВЛЕНИЕ о качестве, не auth-контроль, который обязан fail-closed —
# видимого в логах WARNING достаточно, чтобы владелец поймал рассинхрон при
# ревью, до того как на это заявление будут полагаться ниже по цепочке.
if settings.LIVENESS_QUALITY_CERTIFIED:
    if settings.LIVENESS_TARGET_APCER is None or settings.LIVENESS_TARGET_BPCER is None:
        log.warning(
            "LIVENESS_QUALITY_CERTIFIED=True but LIVENESS_TARGET_APCER/_BPCER are not "
            "set — a quality_certified claim without numeric targets is not meaningful "
            "(CHALLENGE_ENTROPY_SPRINT_v1.md §7.2)."
        )
    if settings.LIVENESS_CERTIFIED_MODEL_VERSION != LIVENESS_MODEL_VERSION:
        log.warning(
            "LIVENESS_QUALITY_CERTIFIED=True but LIVENESS_CERTIFIED_MODEL_VERSION=%r "
            "does not match the deployed model_version=%r — certified a different build "
            "than what is actually running (CHALLENGE_ENTROPY_SPRINT_v1.md §7.3).",
            settings.LIVENESS_CERTIFIED_MODEL_VERSION, LIVENESS_MODEL_VERSION,
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


class StepWindow(BaseModel):
    """Фаза 2 (CHALLENGE_ENTROPY_SPRINT_v1.md §5.3) — НОВОЕ, аддитивное поле
    контракта. Случайное окно задержки ПОСЛЕ показа предыдущего шага/старта,
    сэмплированное тем же `secrets`-ГСЧ, что и выбор шагов
    (app/liveness_session.py::generate_step_windows). Диапазон
    (`LIVENESS_STEP_DELAY_MIN_MS`/`_MAX_MS`) — ПРЕДВАРИТЕЛЬНЫЙ, не
    согласован с Рустамом/UX (см. §9 п.2 плана), не считать финальным
    UX-контрактом. Клиент (мобильное приложение) должен показывать
    инструкцию к шагу не раньше `min_delay_ms` мс после предыдущего — это
    межрепозиторийная зависимость на egaz-mobile, не решается только этим
    сервисом (см. app/config.py::LIVENESS_TIMING_VALIDATION_ENABLED)."""
    step: str
    min_delay_ms: int
    max_delay_ms: int


class ChallengeSpec(BaseModel):
    steps: list[str] = Field(..., description="Randomized subset+order, e.g. ['TURN_RIGHT','TURN_LEFT']")
    min_frames: int
    max_frames: int
    step_windows: list[StepWindow] = Field(
        default_factory=list,
        description="НОВОЕ аддитивное поле (Фаза 2) — не ломает старых клиентов, читающих "
        "только steps/min_frames/max_frames. См. StepWindow docstring.",
    )


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
    # P0-4 (2026-07-18): narrowed from 4 to 3 values — "low_quality" had no
    # live code path in _run_liveness_verdict (every "not enough data" case
    # here uses "incomplete" instead, see docs/LIVENESS_CONTRACT_v1.md §2.1),
    # it was a reserved-but-dead enum member. Scoped to THIS response model
    # only — PadCheckResponse.verdict (/pad/check) legitimately uses
    # "low_quality" and is untouched.
    verdict: Literal["live", "spoof", "incomplete"]
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
    soft_validation_anomalies: Optional[dict] = None,
) -> None:
    """Metadata-only audit entry — NEVER the frame bytes themselves, same
    privacy posture as _audit_entry() for /pad/check.

    `soft_validation_anomalies` (Фаза 3.2/3.3, CHALLENGE_ENTROPY_SPRINT_v1.md
    §6.2/§6.3): additive/optional field — when `LIVENESS_CAPTURED_AT_
    VALIDATION_ENABLED`/`LIVENESS_TIMING_VALIDATION_ENABLED` are False (the
    soft-rollout default), a detected `captured_at`/timing anomaly does NOT
    fail the verdict, but IS written here so the existing audit-log
    mechanism captures it for later analysis — exactly the "только
    логируем" behavior the plan calls for, reusing this file/stdout audit
    trail rather than inventing a new one."""
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
    if soft_validation_anomalies:
        entry["soft_validation_anomalies"] = soft_validation_anomalies
    audit_log.info(json.dumps(entry, ensure_ascii=False))


def _parse_captured_at(raw: Optional[str]) -> Optional[float]:
    """Best-effort ISO-8601 -> unix-seconds parse for `LivenessFrame.
    captured_at`. Returns None on anything unparseable (malformed string,
    None, OR a naive string with no UTC/offset marker — see below) —
    callers treat that as an anomaly, they do not raise. Handles the
    trailing 'Z' UTC suffix from the contract's own example
    (docs/LIVENESS_CONTRACT_v1.md §2, "2026-07-17T14:32:00.100Z") — Python's
    `datetime.fromisoformat` only accepts 'Z' natively from 3.11+; this repo
    runs 3.12, but the explicit `.replace("Z", "+00:00")` keeps this correct
    even if that assumption ever changes.

    ⚠️ NAIVE STRINGS ARE REJECTED, NOT SILENTLY TREATED AS UTC (HIGH finding,
    MF DOOM code review, 2026-07-20 — decision by the owner/foreman): a
    naive ISO string (no 'Z', no '+HH:MM'/'-HH:MM' offset) parses fine via
    `datetime.fromisoformat`, but `datetime.timestamp()` on a naive
    `datetime` interprets it as LOCAL SERVER TIME (this service runs on
    egaz-02.uz, UTC+5) — silently treating a client-sent naive timestamp as
    UTC would be an unstated assumption baked into the code, not a contract
    guarantee. Since the client's actual timezone for a naive string is
    unknown (it could mean UTC, it could mean the device's local time, it
    could mean server-local by coincidence), the only honest behavior is to
    treat it the SAME as an unparseable string: return None, so the caller
    reports it as the `UNPARSEABLE` anomaly (soft-log today; hard-reject
    once `LIVENESS_CAPTURED_AT_VALIDATION_ENABLED=True`). The contract
    (docs/LIVENESS_CONTRACT_v1.md §7) requires `captured_at` to carry an
    explicit offset (UTC 'Z' recommended) precisely so this ambiguity never
    has to be silently resolved here."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed.timestamp()


def _validate_captured_at(
    frames: list["LivenessFrame"], t_instruction_shown: float, expires_at: float,
) -> Optional[dict]:
    """Фаза 3.2 (CHALLENGE_ENTROPY_SPRINT_v1.md §6.2) — ПЕРВЫЙ контур
    (независимый от Laravel'евской M2-валидации, требование Рустама §1 п.3):
    проверяет окно `[t_instruction_shown, expires_at]` и неубывание по `seq`
    для `captured_at`.

    Мягкий precondition: `captured_at` остаётся `Optional[str]` в схеме — эта
    функция запускает проверку, ТОЛЬКО если ВСЕ кадры сессии его прислали
    (партнёр ещё не подтвердил стабильную отправку на каждом кадре, см.
    app/config.py::LIVENESS_CAPTURED_AT_VALIDATION_ENABLED) — отсутствие
    поля НЕ является аномалией в этом инкременте, это ожидаемое переходное
    состояние, не ошибка.

    Возвращает None, если проверять нечего или всё в порядке; иначе dict с
    описанием, ЧТО именно не так — используется и для audit-лога (soft),
    и для reason=CAPTURED_AT_INVALID (hard)."""
    if not frames or any(f.captured_at is None for f in frames):
        return None

    ordered = sorted(frames, key=lambda f: f.seq)
    parsed: list[tuple[int, Optional[float]]] = [(f.seq, _parse_captured_at(f.captured_at)) for f in ordered]

    anomalies: list[dict] = []
    for seq, ts in parsed:
        if ts is None:
            anomalies.append({"seq": seq, "reason": "UNPARSEABLE"})
            continue
        if ts < t_instruction_shown or ts > expires_at:
            anomalies.append({"seq": seq, "captured_at_ts": ts, "reason": "OUT_OF_WINDOW"})

    prev_ts: Optional[float] = None
    for seq, ts in parsed:
        if ts is None:
            continue  # already reported as UNPARSEABLE above, do not double-report as non-monotonic
        if prev_ts is not None and ts < prev_ts:
            anomalies.append({"seq": seq, "captured_at_ts": ts, "reason": "NOT_MONOTONIC"})
        prev_ts = ts

    if not anomalies:
        return None
    return {"anomalies": anomalies}


def _validate_step_windows(
    frames: list["LivenessFrame"],
    step_windows: list[StepWindowDict],
    step_evidence_seq: dict[str, int],
    t_instruction_shown: float,
) -> Optional[dict]:
    """Фаза 3.3 (CHALLENGE_ENTROPY_SPRINT_v1.md §6.3) — приближённая проверка
    того, что клиент выдержал окно `[min_delay_ms, max_delay_ms]` (Фаза 2)
    перед КАЖДЫМ шагом.

    ⚠️ ЭТО ПРОКСИ, не точное измерение: сервер никогда не видит момент, когда
    клиент реально ПОКАЗАЛ инструкцию — только `captured_at` того кадра,
    который Layer 2 (`app/active_challenge.py::verify_challenge`,
    `detail["step_evidence_seq"]`) засчитал доказательством шага. Честная
    оговорка, а не додуманная точность.

    Тот же мягкий precondition, что и `_validate_captured_at`: запускается,
    только если `captured_at` присутствует на каждом кадре — иначе делать
    нечего, возвращает None."""
    if not frames or any(f.captured_at is None for f in frames):
        return None

    by_seq = {f.seq: f.captured_at for f in frames}
    anomalies: list[dict] = []
    prev_ts = t_instruction_shown
    for window in step_windows:
        step = window["step"]
        seq = step_evidence_seq.get(step)
        if seq is None or seq not in by_seq:
            # Step evidence missing entirely — Layer 2's own STEP_NOT_DETECTED
            # already covers this case, nothing new to say about timing here.
            continue
        ts = _parse_captured_at(by_seq[seq])
        if ts is None:
            anomalies.append({"step": step, "seq": seq, "reason": "CAPTURED_AT_UNPARSEABLE"})
            continue
        delay_ms = (ts - prev_ts) * 1000.0
        if delay_ms < window["min_delay_ms"] or delay_ms > window["max_delay_ms"]:
            anomalies.append({
                "step": step, "seq": seq, "delay_ms": round(delay_ms, 1),
                "expected_min_ms": window["min_delay_ms"], "expected_max_ms": window["max_delay_ms"],
                "reason": "OUTSIDE_WINDOW",
            })
        prev_ts = ts

    if not anomalies:
        return None
    return {"anomalies": anomalies}


def _liveness_not_ready_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Active-liveness endpoints are disabled or models failed to load"},
    )


@app.post("/liveness/challenge", response_model=LivenessChallengeResponse)
async def liveness_challenge(
    req: LivenessChallengeRequest,
    x_service_token: Optional[str] = Header(None, alias="X-Service-Token"),
    x_request_timestamp: Optional[str] = Header(None, alias="X-Request-Timestamp"),
):
    """Generates a randomized challenge_spec and mints session_id. Laravel
    relays session_id + challenge_spec to the client unchanged when it
    handles the public POST /liveness/start. THIS service is the sole
    source of truth for what challenge_spec a given session_id means —
    POST /liveness/verdict looks it up by session_id, it does not trust a
    client- or Laravel-supplied spec."""
    _verify_service_token(x_service_token)
    _verify_replay_protection(x_request_timestamp)

    if not settings.LIVENESS_ENDPOINTS_ENABLED or not _liveness_models_loaded:
        return _liveness_not_ready_response()

    pool = [s.strip() for s in settings.LIVENESS_CHALLENGE_STEPS_POOL.split(",") if s.strip()]
    steps = generate_challenge_spec(
        pool, settings.LIVENESS_CHALLENGE_STEP_COUNT_MIN, settings.LIVENESS_CHALLENGE_STEP_COUNT_MAX,
    )
    # Фаза 2 (§5.3): окна тайминга сэмплируются ОДИН раз здесь, тем же
    # secrets-ГСЧ (см. generate_step_windows), и хранятся в сессии — Фаза
    # 3.3 на /liveness/verdict читает ИМЕННО эту копию (не пересэмплирует),
    # тот же принцип единственного источника истины, что уже используется
    # для `steps`.
    step_windows = generate_step_windows(
        steps, settings.LIVENESS_STEP_DELAY_MIN_MS, settings.LIVENESS_STEP_DELAY_MAX_MS,
    )

    session = session_store.create(
        steps=steps,
        ttl_s=settings.LIVENESS_SESSION_TTL_S,
        correlation_id=req.correlation_id,
        transaction_type=req.transaction_type,
        transaction_ref=req.transaction_ref,
        step_windows=step_windows,
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
            step_windows=step_windows,
        ),
        t_instruction_shown=session.t_instruction_shown,
        expires_at=session.expires_at,
        model_version=LIVENESS_MODEL_VERSION,
    )


def _run_liveness_verdict(req: LivenessVerdictRequest) -> LivenessVerdictResponse:
    """Synchronous body of POST /liveness/verdict — run via asyncio.wait_for
    with LIVENESS_INFERENCE_TIMEOUT_S from the route handler below."""
    t0 = time.perf_counter()
    # Фаза 3.2/3.3 soft rollout (CHALLENGE_ENTROPY_SPRINT_v1.md §6.2/§6.3):
    # accumulates captured_at/timing anomalies that were DETECTED but did not
    # fail the verdict (flags disabled) — read by `_respond` below so every
    # response path (success or failure) writes them into the audit log.
    soft_anomalies: dict = {}

    def _respond(verdict: str, reason: Optional[str], *, frame_consistency_score: float = -1.0,
                 best_frame_seq: Optional[int] = None, signals: Optional[dict] = None) -> LivenessVerdictResponse:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        out_signals = dict(signals or {})
        n_valid = out_signals.pop("_n_valid", 0)  # audit-only counter, not part of the outward signals shape
        _liveness_audit_entry(
            req.correlation_id, req.session_id, req.transaction_type, req.transaction_ref,
            verdict, reason, frame_consistency_score, ms,
            soft_validation_anomalies=soft_anomalies or None,
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

    # --- Фаза 3.2 (§6.2): captured_at window + monotonic-by-seq — ПЕРВЫЙ
    # контур, до Laravel'евской M2-валидации (требование Рустама §1 п.3).
    # Мягкий rollout: если флаг выключен (дефолт), аномалия ТОЛЬКО
    # логируется в audit-log (см. soft_anomalies/_respond выше), вердикт не
    # режется. captured_at отсутствующий — не аномалия в этом инкременте,
    # см. _validate_captured_at docstring.
    captured_at_anomaly = _validate_captured_at(req.frames, session.t_instruction_shown, session.expires_at)
    if captured_at_anomaly is not None:
        if settings.LIVENESS_CAPTURED_AT_VALIDATION_ENABLED:
            return _respond(
                "spoof", "CAPTURED_AT_INVALID",
                signals={"captured_at_validation": captured_at_anomaly},
            )
        soft_anomalies["captured_at"] = captured_at_anomaly

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

    # --- Фаза 3.3 (§6.3): step_windows timing — тот же мягкий rollout, тоже
    # ПРОКСИ-измерение (см. _validate_step_windows docstring про ограничение
    # точности), зависит от Фазы 2 (session.step_windows) И от Layer 2 уже
    # отработавшего успешно (step_evidence_seq известен только когда все
    # шаги найдены).
    timing_anomaly = _validate_step_windows(
        req.frames, session.step_windows, active_result.detail.get("step_evidence_seq", {}),
        session.t_instruction_shown,
    )
    if timing_anomaly is not None:
        if settings.LIVENESS_TIMING_VALIDATION_ENABLED:
            return _respond(
                "spoof", "TIMING_WINDOW_VIOLATED",
                signals={**base_signals, "timing_validation": timing_anomaly},
            )
        soft_anomalies["timing"] = timing_anomaly

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
    x_request_timestamp: Optional[str] = Header(None, alias="X-Request-Timestamp"),
):
    _verify_service_token(x_service_token)
    _verify_replay_protection(x_request_timestamp)

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

    t0 = time.perf_counter()
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
    except HTTPException:
        raise
    except Exception:
        # P0-1 (2026-07-18): local try/except instead of relying on the
        # generic app.exception_handler(Exception) — by the time we get here
        # req is ALREADY a parsed LivenessVerdictRequest, so
        # session_id/correlation_id/transaction_type/transaction_ref are in
        # scope and can be echoed correctly (the generic handler cannot see
        # them and always echoes null). Deliberately local to this route, not
        # a middleware/contextvar — this is the only endpoint that needs it.
        ms = round((time.perf_counter() - t0) * 1000, 1)
        log.exception(
            "liveness_verdict: unhandled exception correlation=%s session=%s",
            req.correlation_id, req.session_id,
        )
        _liveness_audit_entry(
            req.correlation_id, req.session_id, req.transaction_type, req.transaction_ref,
            "incomplete", "INTERNAL_ERROR", -1.0, ms,
            n_frames_received=len(req.frames), n_frames_valid=0,
        )
        return JSONResponse(
            status_code=500,
            content={
                "verdict": "incomplete",
                "reason": "INTERNAL_ERROR",
                "model_version": LIVENESS_MODEL_VERSION,
                "frame_consistency_score": -1.0,
                "best_frame_seq": None,
                "session_id": req.session_id,
                "correlation_id": req.correlation_id,
                "transaction_type": req.transaction_type,
                "transaction_ref": req.transaction_ref,
                "processing_ms": ms,
                "signals": {},
            },
        )
