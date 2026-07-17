"""Pydantic settings loaded from environment variables."""

import logging
from pathlib import Path
from typing import Annotated
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration from env vars."""

    LIVENESS_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
    DEVICE: str = "auto"
    MAX_BATCH: int = 16

    # Phase 1 PAD-gate integration (BACKEND_REQUIREMENTS_2026-07-06)
    SERVICE_TOKEN: str = ""  # X-Service-Token shared secret with Laravel; empty = auth disabled
    RATE_LIMIT_BURST: int = 20  # max concurrent requests (per-second burst)
    RATE_LIMIT_SUSTAINED: float = 5.0  # sustained requests per second
    SAVE_FRAME_VERDICTS: str = "spoof"  # comma-separated verdicts that trigger save_frame=true

    # Layer 0 — document/passport-photo pre-filter (RZA, 2026-07-16).
    # DEFAULT DISABLED: calibration on incident_urgut (n=1 spoof / n=12 bonafide,
    # see app/document_check.py module docstring for full numbers) found 2 of 4
    # tested bonafide with a plain/simple background (indoor wall; painted wood
    # door) get flagged as a studio/document background by minicpm-v — an
    # ordinary home-selfie scenario, not a rare edge case. Latency (~50-90s/call
    # observed, worse under shared-GPU contention) is also far beyond the
    # existing /pad/check budget (INFERENCE_TIMEOUT_S=2s). Do not enable without
    # a larger calibration pass first.
    DOCUMENT_CHECK_ENABLED: bool = False
    # Vision model tag (Ollama). Kept configurable — a newer minicpm-v4.6 is
    # under parallel evaluation as of 2026-07-16; switching should be an env
    # change, not a code change.
    DOCUMENT_CHECK_MODEL: str = "minicpm-v:latest"
    DOCUMENT_CHECK_OLLAMA_URL: str = "http://127.0.0.1:11434/api/generate"
    # Confidence in "this is a document/studio photo" (0..1) required to
    # short-circuit to verdict=spoof BEFORE passive-PAD runs.
    DOCUMENT_REJECT_THRESHOLD: float = 0.70
    # Per-call timeout for the Ollama HTTP request. Observed single-flight
    # latency in testing was 50-90s (no GPU contention) — 20s is NOT long
    # enough to reliably complete a call under normal conditions on this
    # hardware; kept configurable so it can be raised, or the layer left
    # disabled, without a code change.
    DOCUMENT_CHECK_TIMEOUT_S: float = 20.0

    # Layer 0a — deterministic face-to-frame geometry gate (RZA, 2026-07-16,
    # RE-CALIBRATED 2026-07-16 evening after a second real incident slipped
    # through the original 0.35 threshold — see app/geometry_check.py module
    # docstring for the full numbers). Reuses the SAME RetinaFace bbox
    # passive-PAD already computes — no extra model, no network call,
    # microseconds.
    #
    # Calibrated on incident_urgut + the 2026-07-16 19:41 incident photo, all
    # with the REAL FaceDetector bbox:
    #   bonafide (12 files):     face_area_ratio 0.043-0.215
    #   document spoof (n=2):    face_area_ratio 0.322/0.472
    # 0.27 sits ~26% above the bonafide area-ratio max and ~16% below the
    # weaker of the two known spoofs (was 0.35 — too high, let the 0.3224
    # incident through). This is the ONLY ratio wired into the reject
    # decision (see app/main.py::_run_geometry_gate). DEFAULT ENABLED: unlike
    # the minicpm-v layer this is free (no latency/availability risk) — but
    # calibration is still n=12 bonafide / n=2 document-spoof phone photos,
    # NOT verified sale-transaction camera frames; production camera may sit
    # closer to the customer and shift the bonafide baseline upward. Re-check
    # against real sale-flow frames before trusting the FRR this implies.
    GEOMETRY_CHECK_ENABLED: bool = True
    FACE_RATIO_REJECT: float = 0.27

    # DIAGNOSTIC ONLY — NOT wired into the reject decision (2PAC review,
    # 2026-07-16: `face_width_ratio` is empirically ~1.09*sqrt(face_area_ratio)
    # on every one of the 14 calibration samples measured so far, so gating
    # on width in addition to area does not catch any attack area misses —
    # it only tightens the effective margin against a real customer standing
    # close to the camera, i.e. pure FRR cost with no FAR benefit on current
    # data). `face_width_ratio` is still computed and reported in
    # `signals.geometry_check` for every request (like `frame_aspect_ratio`)
    # so a future, larger, independently-collected sample can re-evaluate it
    # as a genuinely orthogonal signal. This constant is kept only so that
    # value is available if/when that recalibration wires it back in — do
    # not read it as an active threshold today.
    FACE_WIDTH_RATIO_REJECT: float = 0.55

    # CORS (MF DOOM review, 2026-07-16): DEFAULT EMPTY = middleware not
    # attached at all — no CORS headers, no wildcard attack surface. The
    # real production caller is server-side Laravel (not a browser), so CORS
    # is not needed in prod. Set to a list of origins (e.g.
    # ["http://127.0.0.1:8090"]) only for local manual browser testing via
    # testpage/.
    CORS_ALLOW_ORIGINS: Annotated[list[str], NoDecode] = []

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """Accept a plain comma-separated string (e.g. '*' or
        'http://a,http://b') from env, so ctl.sh/env files can set one or
        more origins without JSON-quoting them. NoDecode above stops
        pydantic-settings from trying (and failing) to json.loads() the raw
        env string before this validator ever runs."""
        if isinstance(v, str):
            stripped = v.strip()
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return v

    # ------------------------------------------------------------------
    # Active liveness (RZA, 2026-07-17) — POST /liveness/challenge +
    # POST /liveness/verdict, per docs/plans/FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md
    # Phase 2 and docs/plans/FACEID_LIVENESS_ML_CORE_v1.md Layer 0/2/3.
    #
    # DEFAULT DISABLED. Flipping this on pulls in NEW heavy dependencies
    # (insightface, onnxruntime) and a 260MB ONNX weight file
    # (models/liveness/adaface_ir101_webface12m.onnx) that are not
    # guaranteed to be installed/provisioned on every deploy yet. When
    # False, app/main.py must not import insightface/onnxruntime or touch
    # the weight file at all at startup — /verify, /verify_batch,
    # /spoof-server, /pad/check must keep working unmodified on a host
    # that has not rolled this out.
    LIVENESS_ENDPOINTS_ENABLED: bool = False

    # SCRFD detector input size (insightface buffalo_l, detection +
    # landmark_3d_68 submodules only — NOT the full buffalo_l pipeline, no
    # landmark_2d_106/genderage/recognition load, see app/face_landmarks.py).
    # 320 measured on docs/plans/calibration/incident_urgut (12 bonafide +
    # 12 spoof, single-photo, NOT session frames): ~100-160ms/frame on
    # i5-11400 CPU after model load. See app/face_landmarks.py docstring.
    LIVENESS_DET_SIZE: int = 320

    # AdaFace embedder for Layer 3 cross-frame identity consistency.
    # MEASURED, NOT the model ML_CORE §2.3/§7 recommended: that document
    # explicitly argues for AdaFace IR-18 or IR-50 over IR-101 on CPU
    # latency grounds, but no IR-18/IR-50 checkpoint exists in this repo or
    # the sibling face_id/tracker project today — only the IR-101 ONNX
    # (adaface_ir101_webface12m.onnx, reused from face_id/tracker/weights/,
    # already CPU-turbo-proven there for a DIFFERENT latency budget: a
    # live door-camera stream, not a <2s synchronous HTTP call). Measured
    # HERE on this repo's calibration set (2026-07-17, i5-11400, 12
    # threads, onnxruntime CPUExecutionProvider):
    #   steady-state (warm session, explicit intra_op_num_threads=12):
    #       ~342ms/frame
    #   cold/mixed average across 24 single-shot calls (no warmup
    #       amortization — closer to what an idle low-QPS internal service
    #       actually pays per call): ~524ms/frame
    # At 4-6 key frames/session this is ~1.4-3.1s for embeddings ALONE,
    # before SCRFD detection (~100-160ms/frame) or the existing Layer 1
    # passive-PAD pass (~20ms/frame, already measured in
    # FACEID_PHASE1_PAD_GATE.md §3). Do not treat this as a finished
    # decision — it is the SAME risk ML_CORE flagged, now with real
    # numbers instead of a guess. Swapping to IR-18/50 (or moving this
    # service to GPU) is the clear next step once a lighter checkpoint is
    # available; this constant exists so that swap is an env change, not a
    # code change.
    ADAFACE_ONNX_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "liveness" / "adaface_ir101_webface12m.onnx"

    # Layer 3 cross-frame identity — UNCALIBRATED. ML_CORE §2.3 cites a
    # literature-only working hypothesis (cosine >= ~0.4-0.5 same-person
    # for ArcFace/AdaFace-family embeddings under same-session conditions).
    # Attempted to calibrate this on docs/plans/calibration/incident_urgut
    # (2026-07-17): that dataset turned out to be single photos of
    # DIFFERENT people (not repeat captures of one person across a
    # session) — confirmed by pipeline self-consistency checks (identical
    # input embedded twice -> cosine=1.0, mirror-flip of the same face ->
    # 0.962, same crop re-encoded at JPEG q80 -> 0.980, all sane) followed
    # by cross-file cosine on the actual calibration images, which came
    # back near-zero (bonafide-bonafide pairs: min=-0.079, max=0.203,
    # mean=0.058 — nowhere near a same-person range). This is NOT a
    # pipeline bug; it means the calibration set cannot answer the
    # question Layer 3 needs answered. Kept at the ML_CORE literature
    # floor (0.40) rather than inventing a new number — genuinely
    # UNCALIBRATED, needs a real same-session multi-frame corpus (see
    # ML_CORE §6.2 item 1, "bona fide corpus") before this can be trusted
    # as a security threshold rather than a placeholder.
    IDENTITY_MIN: float = 0.40

    # Layer 2 active challenge — randomization pool. BLINK is now IMPLEMENTED
    # (2026-07-17, app/active_challenge.py + app/face_landmarks.py::
    # eye_aspect_ratios) using EAR from the SAME landmark_3d_68 model already
    # loaded for pose — no landmark_2d_106, no MediaPipe, no new dependency.
    # The EYE-CONTOUR INDEX MAPPING (36-41 right eye, 42-47 left eye — the
    # standard dlib/iBUG-300W 68-point convention landmark_3d_68 is fit to)
    # was VERIFIED against a real photo, not assumed — see
    # FrameFace.landmark_68 docstring in app/face_landmarks.py.
    #
    # BLINK IS DELIBERATELY STILL EXCLUDED FROM THIS POOL, though: index
    # correctness and THRESHOLD calibration (LIVENESS_EAR_BLINK_MAX below)
    # are different questions, and only the first one is solved. ML_CORE
    # §2.2 already flagged its own EAR threshold as "typical value from
    # literature ... NOT calibrated under our camera/resolution —
    # placeholder"; shipping an uncalibrated BLINK gate as an ACTIVE
    # security control (default pool member) is exactly the "don't lower
    # FAR for a good-looking number" trap the honesty rules warn against.
    # Detection stays reachable (a caller can put "BLINK" in a session's
    # own steps) for exactly this future recalibration to build on, without
    # a second implementation pass.
    LIVENESS_CHALLENGE_STEPS_POOL: str = "TURN_LEFT,TURN_RIGHT"
    # How many steps to sample from the pool per session. With only 2
    # supported steps today the entropy against a pre-recorded video-replay
    # attack is low (2 possible orders = 1 bit) — ML_CORE §2.2 calls
    # exactly this out as the reason randomization exists at all; 1 bit is
    # a real but weak deterrent until BLINK (or another distinguishable
    # step) is added. Tracked as an explicit known limitation, not silently
    # accepted as "good enough".
    LIVENESS_CHALLENGE_STEP_COUNT: int = 2

    # Required yaw deviation (degrees) from the frontal reference for a
    # TURN_LEFT/TURN_RIGHT step to count as satisfied. ML_CORE §2.2's own
    # number (+/-20 deg), carried over UNVERIFIED against real device/pose
    # convention — see app/active_challenge.py module docstring for the
    # sign-convention caveat (which physical direction maps to positive
    # yaw has not been confirmed with a labeled real capture).
    LIVENESS_YAW_TURN_MIN_DEG: float = 20.0
    # Max |yaw| for a frame to count as the frontal reference/return-to-
    # center. Placeholder, not calibrated.
    LIVENESS_YAW_FRONTAL_MAX_DEG: float = 10.0

    # BLINK closed-eye cutoff for min(right_ear, left_ear) — see
    # app/active_challenge.py + app/face_landmarks.py::eye_aspect_ratios.
    # UNCALIBRATED: 0.20 is the widely-cited literature value (Soukupová &
    # Čech 2016; the pyimagesearch EAR-blink tutorial most implementations
    # trace back to), NOT measured on this camera/landmark-model domain — no
    # real closed-eye frame exists in this repo to calibrate against (only
    # single bonafide/spoof photos, none mid-blink). A same-domain sanity
    # check on 3 real OPEN-eye frontal selfies (2026-07-17, i5-11400 CPU)
    # found EAR from THIS landmark_3d_68-derived measurement ranging
    # 0.214-0.317 — i.e. one genuinely-open eye already sits at 0.214, only
    # 0.014 above this cutoff. That is a real warning sign the literature
    # number may sit too close to this model's open-eye noise floor, not
    # cosmetic — do NOT add BLINK to LIVENESS_CHALLENGE_STEPS_POOL on the
    # strength of this constant alone; a real open+closed-eye session
    # corpus is needed first (see final report / owner handoff for the
    # concrete ask: a handful of people each captured performing a real
    # blink under production-like lighting).
    LIVENESS_EAR_BLINK_MAX: float = 0.20

    # Frame count bounds per FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md §1.2
    # ("4-6 ключевых кадров"). Below MIN -> verdict=incomplete; above MAX
    # is rejected as a malformed request (protects against a client
    # uploading many more frames than the protocol calls for).
    LIVENESS_MIN_FRAMES: int = 4
    LIVENESS_MAX_FRAMES: int = 6

    # Challenge-session validity window. Session generated by
    # POST /liveness/challenge must be consumed by POST /liveness/verdict
    # before this many seconds elapse (matches ML_CORE §2.2 "полная сессия
    # должна уложиться в общее окно" — exact UX duration is an open owner
    # decision per ML_CORE §8 item 2; 90s is a generous placeholder ceiling
    # covering the 5-6s UX target plus network/retry slack, NOT the target
    # UX duration itself).
    LIVENESS_SESSION_TTL_S: float = 90.0

    # In-memory session store ONLY in this increment — sessions do not
    # survive a process restart and are NOT shared across multiple uvicorn
    # workers/replicas. Fine for a single-process dev/smoke deploy; a
    # horizontally-scaled prod deployment (see FACEID_PHASE1_PAD_GATE.md
    # §2 item 10, "конкурентность/процессы") needs a shared store (Redis)
    # before this can run behind more than one worker process — tracked,
    # not solved here.

    # Inference timeout for POST /liveness/verdict — deliberately larger
    # than /pad/check's INFERENCE_TIMEOUT_S=2.0. Derived from the measured
    # numbers above: up to LIVENESS_MAX_FRAMES=6 key frames x
    # (~524ms AdaFace + ~160ms SCRFD + ~20ms passive-PAD) ~= 4.2s, plus
    # margin for JSON decode/base64 of up to 6 frames. This is a STOPGAP,
    # not an accepted UX budget — ML_CORE §8 item 2 wants total user-facing
    # challenge time at 5-6s, and a multi-second SERVER compute tail on top
    # of that is a real architecture risk to escalate (see IR-101 note on
    # ADAFACE_ONNX_PATH above), not something to quietly accept.
    LIVENESS_INFERENCE_TIMEOUT_S: float = 8.0

    model_config = {"env_prefix": ""}


def resolve_device(requested: str) -> str:
    """Map device string to actual torch device name."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def onnx_providers(device: str) -> list[str]:
    """Map a RESOLVED device string ("cuda"/"cpu", i.e. already passed through
    resolve_device() above) to an onnxruntime provider list — the same
    auto|cpu|cuda knob app/liveness.py's torch-based LivenessEngine already
    gets, extended to this service's onnxruntime/insightface consumers
    (app/adaface.py::AdaFaceEmbedder, app/face_landmarks.py::LandmarkDetector
    — Layer 0/2/3 of the active-liveness pipeline; see
    Settings.ADAFACE_ONNX_PATH docstring for the measured CPU latency
    (342-524ms/frame) this exists to cut down).

    CRITICAL (prod is CPU-only today, egaz-02.uz has no GPU): this is
    deliberately conservative and NEVER assumes a GPU-capable onnxruntime is
    actually installed just because device=="cuda" was requested. It always
    checks onnxruntime's OWN ort.get_available_providers() first —
    requirements.txt keeps `onnxruntime-gpu` an OPTIONAL install (the base
    `onnxruntime` CPU wheel stays mandatory); on a host that only has the CPU
    wheel, "CUDAExecutionProvider" simply is not in that list and this
    silently returns CPU-only, no exception, no GPU package required. The
    caller (AdaFaceEmbedder / LandmarkDetector) additionally wraps session
    creation in a try/except and retries CPU-only if a CUDA provider that
    LOOKED available still fails to actually initialize (e.g. cuDNN/CUDA
    runtime version mismatch) — this function only handles the "package not
    installed at all" case, not every possible runtime failure.
    """
    if device != "cuda":
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        log.debug("onnxruntime provider check failed, falling back to CPU", exc_info=True)
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
