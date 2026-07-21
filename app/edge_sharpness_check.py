"""Layer 0f — edge-vs-center sharpness DIAGNOSTIC (NOT a reject gate).

READ THIS BEFORE EVER WIRING THIS INTO A REJECT DECISION.

WHY THIS EXISTS (RZA, 2026-07-21) — a real fake sample, `real_fake_01.jpg`
(`faces-dataset/real-fakes/`, a genuine photo of a live person, reused
across many unrelated sales — see `docs/plans/HANDOFF-2026-07-21-
cross-transaction-face-reuse.md` for the full incident) measured with a
striking left-edge blur while its center (the face) stayed sharp:
`cv2.Laplacian` variance on 10%-width column strips came back left=9.7,
right=68.5, center=354.5 (raw, no resize) — the left edge is ~36x softer
than the center. The initial hypothesis (from a forensic pass on this one
file) was that this asymmetry is itself an attack signature: natural
depth-of-field/bokeh falls off symmetrically, so a ONE-SIDED smear looked
like it should be a clean tell.

**THAT HYPOTHESIS WAS TESTED AGAINST A REAL BONA FIDE PHOTO AND FAILED
(RZA, 2026-07-21, same day, coordinator-supplied counter-example):** a
genuine live photo ("grandma in car", 1920x2560, NOT staged, NOT this
repo's s001 calibration rig) measured left=1.3 / center=3.6 / right=2.7 —
an edge/center ratio of 0.38, i.e. the SAME "soft edge, sharp face" pattern
as the fake (whose own edge/center ratio on that measurement was 0.28, in
the same ballpark, not cleanly separated). A rough same-methodology check
against this repo's own s001 bona fide corpus
(`datasets/bona_fide/real/`, n=1 subject, n=10 frames, 1080x1920, a
DIFFERENT domain again — see limitation #2 below) also showed the min
edge/center ratio (0.15 at a 10% edge fraction) sitting uncomfortably close
to the fake's 0.027 at the SAME fraction — no clean, reproducible margin
once a second real-world reference point existed.

**CONCLUSION: edge-vs-center sharpness asymmetry, on the data available
today, does NOT reliably separate "this photo was smeared as part of an
attack" from "this is an ordinary phone selfie with a naturally softer
background/edge and a sharp face"** — plain lens/JPEG-compression
softness at the frame boundary, encoding artifacts, or just where the
subject stood relative to the focal plane can all produce the same
low-edge/high-center pattern on a completely genuine capture. Wiring this
into a reject decision on the strength of ONE fake sample and THREE
inconsistent bona fide reference points (this repo's own s001 rig, a
coordinator-supplied live photo, and the general "any selfie has a busier
foreground than background" intuition) would very likely reject real
customers — exactly the FRR risk the honesty rules in this repo's
`superpowers`/RZA brief warn against.

WHAT THIS MODULE IS INSTEAD: a DIAGNOSTIC-ONLY measurement, computed and
attached to `/pad/check`'s response `signals` (when explicitly enabled,
default OFF) purely so a FUTURE, larger real corpus can be checked against
it without a second implementation pass — the same "kept reachable for
future recalibration, not wired into a decision" posture this repo already
uses for `FACE_WIDTH_RATIO_REJECT` (`app/geometry_check.py`/`app/config.py`)
and the (still-unadded) BLINK/SMILE challenge steps
(`LIVENESS_EAR_BLINK_MAX`/`LIVENESS_MAR_SMILE_MIN`). **No `is_*_flagged`
boolean from this module is ever read by a gate in `app/main.py` — only the
raw numbers are surfaced.**

KNOWN LIMITATIONS:

1. **n=1 attack sample, n=1 counter-example, n=1 unrelated calibration
   rig.** None of these is a population — every number in this docstring
   is a single data point, not a distribution.
2. **Domain mismatch across every reference point used here** — the fake
   (720x1280, JPEG re-encoded, EXIF stripped), the coordinator's live photo
   (1920x2560, in-car, different lighting/lens entirely), and this repo's
   s001 rig (1080x1920, video frame extraction, single staged subject) are
   three different capture pipelines/resolutions/lens systems, not three
   samples of the same population.
3. **No resize/scale-normalization applied here** (unlike
   `app/blur_check.py`'s 224x224 crop resize) — raw column-strip Laplacian
   variance is reported as-is, matching the exact methodology both the
   original forensic pass and the coordinator's counter-example used, so
   the numbers in this docstring stay directly comparable to the
   measurements that motivated (and then disproved) the original
   hypothesis. A resize-normalized variant might behave differently and is
   NOT what is implemented here.
4. **NEVER gate on this without a real, larger, independently-collected
   corpus first** — see the final report / owner handoff for the concrete
   ask (more real fake samples with genuine one-sided smear + a broader
   bona fide sample across devices/lighting) before even considering
   turning this into a reject decision.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeSharpnessDiagnostic:
    """Raw measurement only — NO reject/flag field on purpose (see module
    docstring: this is diagnostic data, not a gate input). `ran=False`
    degrades safely (bad input) — caller should just omit the signal, never
    treat it as an error."""

    ran: bool
    left_sharpness: float = 0.0
    right_sharpness: float = 0.0
    center_sharpness: float = 0.0
    left_to_center_ratio: float = 0.0
    right_to_center_ratio: float = 0.0
    min_edge_to_center_ratio: float = 0.0
    error: Optional[str] = None


def measure_edge_sharpness(image_bgr: np.ndarray, edge_fraction: float = 0.12) -> EdgeSharpnessDiagnostic:
    """Pure measurement, full frame (NOT the face bbox — the smear that
    motivated this module sits OUTSIDE the detected face bbox on
    `real_fake_01.jpg`, confirmed by running this repo's own
    `FaceDetector` against it: bbox x-range [177, 575] on a 720px-wide
    frame leaves the left ~25% of the frame, where the smear lives, entirely
    outside the crop `app/blur_check.py` measures). Never raises.
    """
    try:
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0 or not (0.0 < edge_fraction < 0.5):
            return EdgeSharpnessDiagnostic(ran=False, error="INVALID_INPUT")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edge_w = max(1, int(w * edge_fraction))
        if edge_w * 2 >= w:
            return EdgeSharpnessDiagnostic(ran=False, error="FRAME_TOO_NARROW")

        left = gray[:, 0:edge_w]
        right = gray[:, w - edge_w:w]
        center = gray[:, edge_w:w - edge_w]

        left_var = float(cv2.Laplacian(left, cv2.CV_64F).var())
        right_var = float(cv2.Laplacian(right, cv2.CV_64F).var())
        center_var = float(cv2.Laplacian(center, cv2.CV_64F).var())

        if center_var <= 0:
            return EdgeSharpnessDiagnostic(
                ran=True, left_sharpness=round(left_var, 2), right_sharpness=round(right_var, 2),
                center_sharpness=0.0, left_to_center_ratio=0.0, right_to_center_ratio=0.0,
                min_edge_to_center_ratio=0.0,
            )

        left_ratio = left_var / center_var
        right_ratio = right_var / center_var
        return EdgeSharpnessDiagnostic(
            ran=True,
            left_sharpness=round(left_var, 2),
            right_sharpness=round(right_var, 2),
            center_sharpness=round(center_var, 2),
            left_to_center_ratio=round(left_ratio, 4),
            right_to_center_ratio=round(right_ratio, 4),
            min_edge_to_center_ratio=round(min(left_ratio, right_ratio), 4),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic path must never crash the request
        return EdgeSharpnessDiagnostic(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
