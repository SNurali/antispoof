"""Pydantic settings loaded from environment variables."""

from pathlib import Path
from typing import Annotated
from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode


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

    model_config = {"env_prefix": ""}


def resolve_device(requested: str) -> str:
    """Map device string to actual torch device name."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
