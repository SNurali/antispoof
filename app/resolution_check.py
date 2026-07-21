"""Layer 0e — image resolution/weight pre-filter (no ML, no network).

WHY THIS GATE EXISTS (RZA, 2026-07-21, owner request) — a real calibration
dataset (`faces-dataset/`, 199 photos collected from a Telegram group for
antispoof calibration) turned out to be UNUSABLE for that purpose: every
single file was a Telegram *preview/thumbnail* re-encode (height <= 800px,
mean weight ~59KB, max 128KB) — Telegram silently downsizes/recompresses any
photo it relays through a chat, GPS/EXIF/high-frequency edge detail is gone,
and on frames like that motion-blur/edge smear (the real spoof signature the
owner is chasing — a printed/replayed photo with a smear stripe along one
edge, in FULL quality) becomes invisible: `blur_check.py`'s Laplacian-variance
signal reads a heavily downsampled, already re-JPEG'd crop as "sharp enough"
because Telegram's own re-encode already threw away the high-frequency energy
the blur gate depends on. The passive-PAD ensemble reportedly passed these as
`verdict=live` with high confidence for the same underlying reason.

This gate does NOT try to detect "this came from Telegram" specifically (no
metadata signal survives re-encoding to check for that). It targets the
proxy the owner actually described: a genuine phone-camera capture straight
from `egaz-mobile`'s own capture pipeline is reliably BIGGER (both in pixels
and in bytes) than a photo that has been saved out of a messaging app,
screenshotted, or re-forwarded one or more times — every one of those paths
re-encodes through a lossy preview/thumbnail step at some point. Rejecting
anything below a floor comfortably under our own client's real output, but
comfortably above what a Telegram-relayed (or similarly re-compressed) image
measures at, closes that specific gap.

GROUND TRUTH ON OUR OWN CLIENT (RZA, 2026-07-21) — read directly from
`egaz-mobile/core/core-faceid-capture` before picking any number here, not
guessed:

    - `internal/FaceFrameProcessor.kt` (the ONLY place `face_photo` bytes are
      produced today, for BOTH `POST /sales` and `POST /face/verify` —
      Phase-2 burst-liveness's own tighter `~100KB/frame` budget mentioned in
      that file's docstring is NOT implemented yet, nothing reaches this
      service through that path today):
        MAX_DIMENSION_PX  = 1280   (long side capped here)
        TARGET_BYTES      = 350 * 1024  (~350KB — an UPPER bound the binary
                                          search compresses DOWN to, not a
                                          floor; a low-detail/plain-background
                                          frame can legitimately end up
                                          smaller than this)
        MIN_JPEG_QUALITY  = 40     (floor quality if 350KB is unreachable)
    - `internal/FaceCaptureGeometry.kt`: capture aspect ratio is 3:4
      (`PREVIEW_ASPECT_RATIO_WIDTH=3` / `_HEIGHT=4`), fed into CameraX's
      `ViewPort`+`UseCaseGroup` (`FaceCameraEngine.bind()`) shared across
      `Preview`+`ImageCapture`+`ImageAnalysis` — the documented purpose of
      that CameraX construct is exactly to make all three use cases see the
      SAME crop, so the still-capture JPEG this service receives should
      already be ~3:4, not the camera's raw (often 4:3 or wider) sensor
      aspect.
    - No modern front camera sold on a phone in this market natively
      captures below 1280px on its long side (even budget devices are
      several MP) — `MAX_DIMENSION_PX=1280` is a DOWNSCALE cap, so in
      practice the client's real output is at or very near 1280px on the
      long side, essentially always.

    => Expected real client output: long side ~1280px, short side ~960px
    (3:4 aspect) => ~1.23 megapixels, weight usually in the low hundreds of
    KB (bounded above by ~350KB, no hard floor below that from the code
    itself).

MEASURED ON THE 199-FILE TELEGRAM-PREVIEW DATASET (RZA, 2026-07-21,
`faces-dataset/real` n=173 + `faces-dataset/fake` n=26 — NOT a spoof-calibration
signal, this dataset is ALL Telegram-preview-shaped regardless of its
real/fake label; used here purely as a same-shape sample of what a
re-compressed/relayed photo measures at):

    megapixels: min 0.080, max 0.498 (both folders combined), p95 ~0.48
    min(width,height): min 249px, max 623px, p95 ~600-623px
    weight: min 9.7KB, max 128.3KB, p95 ~77-90KB

Every one of the 199 files sits at or below 0.4984 megapixels and 623px on
its short side.

THRESHOLDS CHOSEN (RZA, 2026-07-21):

    MIN_IMAGE_MIN_SIDE_PX  = 700    (~12% above the dataset's observed max of
                                      623px; ~27% below the client's expected
                                      ~960px short side)
    MIN_IMAGE_MEGAPIXELS   = 0.55   (~10% above the dataset's observed max of
                                      0.498MP; ~55% below the client's
                                      expected ~1.23MP)
    MIN_IMAGE_BYTES        = 15360  (15KB) — DELIBERATELY LOW, see limitation
                                      #2 below. This is a corrupted/near-blank
                                      -image floor, not a Telegram-detection
                                      signal.

Any ONE of the three failing is enough to reject (`is_low_resolution=True`,
OR not AND) — min-side and megapixels are correlated (both derived from the
same width/height) and kept together deliberately: min-side alone can be
fooled by a very wide-but-short crop that still has plenty of total area
(rare for a face photo, but not impossible), megapixels alone can be fooled
by a very elongated frame with one large dimension. Together they cover more
of the aspect-ratio space than either alone, at near-zero extra cost.

KNOWN LIMITATIONS — same honesty bar as blur_check.py/pose_check.py:

1. **The `faceS-dataset/` 199 files are NOT a spoof-calibration set for this
   gate.** They prove "Telegram preview shape", not "attack photo shape" —
   an attacker could easily upload a large, heavy, non-recompressed image
   (a printed photo photographed by a good camera, or a screen replay shot
   close-up on a modern phone) that clears every threshold here. This gate
   is explicitly NOT a spoof detector on its own — see
   `FRAME_SHARPNESS_CHECK_ENABLED`/`blur_check.py` for the signal that
   targets the actual visual artifact (edge smear/blur) the owner described.
   **This gate and the blur gate are a PAIR, not substitutes**: an attacker
   who upscales a small/recompressed photo to clear THIS gate's pixel
   floor does not gain real high-frequency detail back — the upscale itself,
   or the original re-compression, tends to read as low Laplacian-variance
   sharpness, which `blur_check.py` (when enabled) still catches. Neither
   gate alone is a defense against a determined attacker; only together do
   they meaningfully cover both "too small/recompressed" AND "was, at some
   point, saved through a lossy step" cases without needing to trust either
   signal in isolation.
2. **`MIN_IMAGE_BYTES` is NOT tuned against the Telegram dataset on purpose.**
   `TARGET_BYTES=350*1024` in `FaceFrameProcessor.kt` is an upper bound the
   client's own binary search compresses DOWN to when needed — it is NOT a
   floor. A real client photo of a face against a plain/low-detail
   background (a real, common case at a checkout counter) can legitimately
   JPEG-encode at 90% quality, 1.23MP, to well under 100KB — closer to the
   Telegram dataset's own weight range (9.7-128KB) than the naive "camera
   photos are heavy" intuition suggests. Setting `MIN_IMAGE_BYTES` anywhere
   near the dataset's own max (128KB) risks rejecting genuine low-detail
   client frames — real FRR risk, not just theoretical. 15KB is chosen only
   as a floor against a corrupted/near-blank upload, not as a
   Telegram-detection signal; the min-side/megapixel checks above do the
   actual work.
3. **No live-traffic sample from `egaz-mobile` itself.** Every number
   attributed to "the client" above is read directly from
   `FaceFrameProcessor.kt`/`FaceCaptureGeometry.kt` source, not measured on
   a real captured JPEG from a real device — CameraX's actual cropping
   behavior for `ImageCapture` inside a `ViewPort`+`UseCaseGroup` (whether
   the JPEG bytes are ACTUALLY cropped to 3:4, vs. only the crop hint/EXIF
   being set on some devices) is a live-device concern this repo cannot
   verify. If it turns out some devices ship the full (e.g. 4:3 or wider)
   sensor aspect uncropped, the real client short-side floor could be lower
   than the ~960px assumed here — `MIN_IMAGE_MIN_SIDE_PX=700` was picked
   with a margin specifically to survive that uncertainty (700 clears even a
   16:9-aspect worst case at 1280 long side: 720px short side), but this has
   not been confirmed against a real device. DEFAULT DISABLED for exactly
   this reason — see `Settings.RESOLUTION_CHECK_ENABLED` docstring.
4. **Defense-in-depth, not a spoof verdict.** Like `blur_check.py`/
   `pose_check.py`, a hit here returns `verdict="low_quality"`, never
   `verdict="spoof"` — a small/re-compressed image alone is not independently
   confirmed fraud (an honest customer could, in principle, be on a very old
   device or a degraded network path that mangled the upload); it means
   "reshoot", not "accuse".
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolutionCheckResult:
    """Outcome of the Layer 0e resolution/weight check. Always constructible
    without raising — failure states are represented as data, not
    exceptions."""

    ran: bool  # False => bad input (should not happen given a real decoded image); caller fails safe
    is_low_resolution: bool = False
    width: int = 0
    height: int = 0
    min_side: int = 0
    megapixels: float = 0.0
    byte_size: int = 0
    reason: Optional[str] = None  # e.g. "MIN_SIDE+MEGAPIXELS" — which sub-check(s) fired
    error: Optional[str] = None


def check_image_resolution(
    width: int,
    height: int,
    byte_size: int,
    min_side_px: int,
    min_megapixels: float,
    min_bytes: int,
) -> ResolutionCheckResult:
    """Pure arithmetic on already-known dimensions/byte size — no image
    decode, no model, no bbox needed (unlike blur_check.py/pose_check.py,
    this gate does not depend on a detected face at all, so it can run
    BEFORE face detection in every caller).

    Never raises: malformed input (non-positive width/height) degrades to
    `ran=False` so the caller can unconditionally fail safe (treat as "gate
    did not fire", fall through unchanged) rather than block a legitimate
    request on this gate's own bug.
    """
    try:
        if width <= 0 or height <= 0:
            return ResolutionCheckResult(ran=False, error="INVALID_DIMENSIONS")

        min_side = min(width, height)
        megapixels = round((width * height) / 1_000_000, 4)

        fired: list[str] = []
        if min_side < min_side_px:
            fired.append("MIN_SIDE")
        if megapixels < min_megapixels:
            fired.append("MEGAPIXELS")
        if byte_size < min_bytes:
            fired.append("BYTES")

        return ResolutionCheckResult(
            ran=True,
            is_low_resolution=bool(fired),
            width=width,
            height=height,
            min_side=min_side,
            megapixels=megapixels,
            byte_size=byte_size,
            reason="+".join(fired) if fired else None,
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the request path
        return ResolutionCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
