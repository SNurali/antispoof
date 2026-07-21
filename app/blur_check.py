"""Layer 0c — deterministic frame-sharpness gate (no ML, no network).

Reuses the SAME face bbox from `FaceDetector` (RetinaFace) that passive-PAD
and the Layer 0a geometry gate (app/geometry_check.py) already compute for
every /pad/check request — no additional model, pure OpenCV Laplacian
variance on a resize of the bbox crop, microseconds.

WHY THIS GATE EXISTS (RZA, 2026-07-21) — a real bypass pattern: an attacker
presenting a printed/screen photo of a face, held at an angle and/or with
deliberate motion blur, was observed passing /pad/check as `verdict=live`.
Root cause investigated in app/main.py::pad_check's existing pipeline:

    1. Layer 0a (geometry) only rejects on face-to-frame AREA ratio — it has
       no opinion on sharpness at all.
    2. Layer 1 (passive-PAD, app/liveness.py + app/multisignal.py) DOES react
       to blur, but only indirectly: `multisignal.recapture_spoof_score`
       treats low Laplacian variance as "spoof-like" (correct direction —
       blur pushes recapture UP, not down), but `_fuse()` only HARD-overrides
       an NN "real" call when `recap >= RECAPTURE_SPOOF_THRESHOLD` (0.5) AND
       is "confirmed" by `lbp > 0.1` or `moire > 0.1`. Motion blur is
       specifically the kind of degradation that can suppress those
       CONFIRMING texture/periodicity signals (screen pixel-grid, print
       halftone, LBP micro-texture) while only partially raising the
       blended-weight `recapture` score — leaving a real gap: NN stays
       confident "real" (MiniFASNet was trained to tolerate natural motion
       blur/low light as a real-world nuisance, not penalize it) AND the
       weighted `spoof_probability` sits inside the (0.35, 0.6) band where
       `_fuse()` only flips to spoof if `nn_score < 0.7` — a real attack that
       keeps the CNN confident stays classified `live`. See
       app/liveness.py::_fuse docstring and this module's calibration notes
       below for the numbers behind this.

This gate closes that gap the CHEAP way: reject outright, before passive-PAD
ever runs, any frame whose face crop is too blurry to trust — independent of
whether the multisignal ensemble's confirming signals happen to fire. This
does not replace passive-PAD's own blur-sensitivity, it backstops it.

CALIBRATION (2026-07-21, RZA) — `docs/plans/calibration/s001_2026-07-20/`
(8 bona fide frames, ONE subject, s001, frontal/left15/right15/left30/
right30/up/down, extracted from `_capture_staging/s001/20260720T191516/`),
sharpness measured as `cv2.Laplacian(gray, CV_64F).var()` on the SAME 224x224
resize of the RetinaFace bbox crop `app/multisignal.py::recapture_spoof_score`
already uses (this crop is resize-based, not raw-pixel, so it stays
reasonably scale-invariant across different face-to-frame framings — the
metric this gate reads is the same one already proven inside `recapture`):

    bona fide, sharp/no blur (n=8):     93.0 .. 437.1  (min = "down" tilt)
    same 8 frames, motion-blur k=9
    (9px linear kernel, simulating a
    deliberately smeared attack photo): 25.6 .. 89.4

`MIN_FACE_SHARPNESS_224=60.0` sits inside that gap: below the bona fide floor
(93.0, ~35% margin) and above most of the k=9-blurred values (6 of 8 fall
below 60; the 2 that don't — 83.4, 89.4 — still lose their multisignal
"real" cover to `_fuse()`'s spoof_prob rise, i.e. still layered defense, not
solely reliant on this gate).

KNOWN LIMITATIONS — same honesty bar as app/geometry_check.py:

1. **n=1 subject, n=8 frames.** This is a single-person staged capture, not
   a distribution. The 93.0 bona fide floor could be this subject's or this
   capture rig's own floor, not a population number.
2. **Domain mismatch risk, the SAME one geometry_check.py already flags for
   its own calibration set**: s001's face-to-frame area ratio (0.30-0.55,
   measured 2026-07-21) is MUCH tighter framing than the phone-selfie
   incident_urgut set geometry_check.py was calibrated on (0.04-0.21) — s001
   is closer to the camera than the assumed checkout-camera distance. Using a
   scale-invariant (resize-based) crop for sharpness mitigates this more than
   a raw-pixel metric would, but it is NOT proven equivalent across framings
   with zero real checkout-camera frames to check against.
3. **No real attack sample.** The blur values above come from a SYNTHETIC
   motion-blur kernel applied to bona fide frames in this repo, not the
   actual photo that slipped through in production. Synthetic attempts to
   reconstruct the reported bypass in this pass (recapture-simulated print +
   synthetic motion blur + perspective tilt) did NOT cleanly reproduce a
   false-`live` verdict on available data — several landed as false-REJECTS
   instead once padding artifacts were accounted for, and the true production
   photo may differ significantly (real second-camera sensor noise a
   synthetic downscale/re-JPEG cannot fully mimic). Recommend collecting the
   actual accepted attack frame(s) from production (if retained) to verify
   `MIN_FACE_SHARPNESS_224` against the real thing, not just this synthetic
   proxy — see final report for the concrete ask.

CONSIDERED, NOT PICKED: gating on `app/frame_qc.py::MIN_SHARPNESS` (40.0,
on an ALIGNED 112x112 crop) instead of a new module — not reused here
because that crop requires `LandmarkDetector.align_112` (5-point landmark
alignment), which needs the SCRFD+landmark_3d_68 model that `/pad/check`'s
RetinaFace-bbox-only detector does not load by default (see app/config.py::
LIVENESS_ENDPOINTS_ENABLED). This gate is deliberately dependency-free so it
can default ON like GEOMETRY_CHECK_ENABLED, unlike the pose gate in
app/pose_check.py which DOES need that heavier model and defaults OFF.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SharpnessCheckResult:
    """Outcome of the Layer 0c sharpness check. Always constructible without
    raising — failure states are represented as data, not exceptions."""

    ran: bool  # False => bad input (should not happen given a real bbox); caller fails safe
    is_blurry: bool = False
    sharpness: float = 0.0
    error: Optional[str] = None


def check_face_sharpness(
    bbox: list[int],
    image_bgr: np.ndarray,
    threshold: float,
    crop_size: int = 224,
) -> SharpnessCheckResult:
    """Compute Laplacian-variance sharpness on a resize of the already-
    detected face bbox — no re-detection, no alignment, no extra model.

    `bbox` = [x, y, w, h] as returned by `FaceDetector.detect()` — pass the
    SAME bbox already computed for passive-PAD/geometry. `crop_size=224`
    matches `app/multisignal.py::recapture_spoof_score`'s own native-crop
    scale, so this gate reads the same signal that signal already proves
    reacts correctly to blur — this module just makes it a hard, independent
    reject instead of one ensemble-weighted vote.

    Never raises: any malformed input degrades to `ran=False` so the caller
    can unconditionally fail safe to passive-PAD.
    """
    try:
        h, w = image_bgr.shape[:2]
        x, y, bw, bh = bbox
        if bw <= 0 or bh <= 0 or h <= 0 or w <= 0:
            return SharpnessCheckResult(ran=False, error="INVALID_DIMENSIONS")

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            return SharpnessCheckResult(ran=False, error="EMPTY_CROP")

        crop = image_bgr[y1:y2, x1:x2]
        crop = cv2.resize(crop, (crop_size, crop_size))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return SharpnessCheckResult(
            ran=True,
            is_blurry=sharpness < threshold,
            sharpness=round(sharpness, 2),
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the request path
        return SharpnessCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
