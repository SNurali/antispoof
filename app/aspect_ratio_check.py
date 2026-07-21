"""Layer 0g — camera-aspect-ratio pre-filter (no ML, no network).

WHY THIS GATE EXISTS (RZA, 2026-07-21, owner-supplied signal) — a real
confirmed-fraud sample (`faces-dataset/real-fakes/real_fake_01.jpg`, a
genuine photo of a live person reused across many sales, see
`docs/plans/HANDOFF-2026-07-21-cross-transaction-face-reuse.md`) is
720x1280 — a 9:16 ratio (`width/height=0.5625`). A phone CAMERA's still-photo
output is never 9:16: that ratio is a SCREEN/video aspect (phone
screenshots, video frames, most social-app preview crops), not a sensor
still-capture ratio. Camera stills are 3:4 (0.75), 4:5 (0.80), or similar —
see the client ground-truth below. An image arriving at 9:16 (or any other
screen-shaped ratio) is evidence the bytes did NOT come from a live camera
shutter press — it was saved/downloaded/screenshotted and injected as a
file upload instead.

MEASURED ON `faces-dataset/` (RZA, 2026-07-21, real/+fake/, 199 files —
the same Telegram-preview calibration set used for
`app/resolution_check.py`): 174 of the 199 files are EXACTLY 450x800 or
its scaled equivalents (720x1280, 1080x1920 — all reduce to the same
0.5625 ratio) — a 9:16 shape. The remaining ~25 files sit at 0.75-0.78
(600x800, 602x800, 249x320, 623x800) — camera-shaped. This lines up with
the earlier resolution-gate finding (the whole dataset is Telegram-
preview-shaped) PLUS a sharper, independent signal: within that same
dataset, the specific 9:16 SUBSET is disproportionately represented, and
`real_fake_01.jpg` (the one CONFIRMED real fraud sample, not a Telegram
scrape guess) sits exactly on that 9:16 shape.

GROUND TRUTH ON OUR OWN CLIENT AND A REAL BONA FIDE SAMPLE (RZA,
2026-07-21):

    - egaz-mobile's own capture pipeline (`core/core-faceid-capture/.../
      FaceCaptureGeometry.kt`, `PREVIEW_ASPECT_RATIO_WIDTH=3` /
      `_HEIGHT=4`, fed into CameraX `ViewPort`) captures at 3:4 (0.75) —
      see app/resolution_check.py's own module docstring for the full
      derivation of this same fact.
    - A real bona fide photo the owner supplied for comparison ("grandma
      in car", 1920x2560, NOT a staged/synthetic sample) is ALSO exactly
      3:4 (0.75).

THRESHOLDS CHOSEN (RZA, 2026-07-21): `ASPECT_RATIO_MIN=0.70` /
`ASPECT_RATIO_MAX=0.85` — a band centered between the two camera ratios the
task called out (3:4=0.75, 4:5=0.80), wide enough to clear every camera-
shaped sample measured above (0.75-0.7788) with margin, narrow enough to
reject every 9:16-shaped sample (0.5625, ~24% below the band floor) and a
1:1 square (1.0, ~15% above the band ceiling) or any wider screen ratio
(16:9 landscape reduces to the SAME 0.5625 as a portrait 9:16 once measured
as min(w,h)/max(w,h) — see below).

ORIENTATION-AGNOSTIC BY DESIGN: the ratio measured is `min(width,height) /
max(width,height)`, not `width/height` — a client image could in principle
arrive with width/height swapped (e.g. EXIF-rotation edge cases, though
this pipeline's own client never emits EXIF at all, see
`FaceCaptureResult.kt`) without this gate flipping its verdict depending on
which axis happens to be "width" in the uploaded bytes.

KNOWN LIMITATIONS — same honesty bar as every other Layer 0 gate here:

1. **Trivially defeated by cropping the fake to 3:4 before upload.** This
   is a CHEAP, FIRST-LAYER filter against the specific "inject a ready-made
   9:16 screenshot/download file" pattern observed in the real sample — NOT
   a general anti-spoof control. An attacker who crops/pads their reused
   photo to 3:4 (or otherwise fakes the aspect) clears this gate trivially;
   the crop itself does not restore any of the original camera's actual
   capture provenance. See the module docstring warnings on
   `app/resolution_check.py` and `app/edge_sharpness_check.py` for the same
   "defense-in-depth, not a silver bullet" posture — this gate, the
   resolution gate, and (once real data supports it) a genuine
   liveness/capture-provenance control are meant to stack, not substitute
   for each other.
2. **n=1 confirmed-fraud sample, n=1 owner-supplied bona fide sample.**
   The `faces-dataset/` 199-file distribution is corroborating evidence
   (174/199 files at 9:16), not a controlled attack-vs-bona-fide
   experiment — that dataset's own labels (`real`/`fake` folders) are
   themselves Telegram-preview scrapes of unknown provenance, not a
   verified ground truth for THIS specific signal.
3. **DEFAULT DISABLED** (`ASPECT_RATIO_CHECK_ENABLED=False`, see
   `app/config.py`) — pending review, same rollout posture as every other
   gate added in this pass.
4. **verdict=low_quality, never spoof** — matching the "reshoot, don't
   accuse" posture already established for the blur/pose/resolution gates:
   a wrong-aspect frame alone is not independently confirmed fraud (a
   legitimate integration bug, a non-standard device, or a manual test
   upload could also produce this).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AspectRatioCheckResult:
    """Outcome of the Layer 0g camera-aspect-ratio check. Always
    constructible without raising — failure states are represented as
    data, not exceptions."""

    ran: bool  # False => bad input (should not happen given a real decoded image); caller fails safe
    is_non_camera_geometry: bool = False
    width: int = 0
    height: int = 0
    ratio: float = 0.0
    error: Optional[str] = None


def check_aspect_ratio(
    width: int,
    height: int,
    min_ratio: float,
    max_ratio: float,
) -> AspectRatioCheckResult:
    """Pure arithmetic on already-known dimensions — no image decode, no
    model, no bbox needed (same bbox-independent shape as
    app/resolution_check.py's gate, can run before face detection).

    `ratio = min(width, height) / max(width, height)` — orientation
    agnostic, see module docstring. Never raises: malformed input
    (non-positive width/height) degrades to `ran=False` so the caller can
    unconditionally fail safe (treat as "gate did not fire").
    """
    try:
        if width <= 0 or height <= 0:
            return AspectRatioCheckResult(ran=False, error="INVALID_DIMENSIONS")

        ratio = min(width, height) / max(width, height)
        is_non_camera = not (min_ratio <= ratio <= max_ratio)

        return AspectRatioCheckResult(
            ran=True,
            is_non_camera_geometry=is_non_camera,
            width=width,
            height=height,
            ratio=round(ratio, 4),
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the request path
        return AspectRatioCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
