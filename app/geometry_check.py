"""Layer 0a — deterministic face-to-frame geometry gate (no ML, no network).

Reuses the SAME face bbox from `FaceDetector` (RetinaFace) that passive-PAD
already computes for every request — no additional model, no extra
inference cost, no network call. Runs in microseconds (pure arithmetic on
numbers already on hand) and cannot time out or be affected by GPU
contention, unlike app/document_check.py (the minicpm-v/Ollama Layer 0
attempt, kept in the repo but DISABLED — see that module's docstring for
why: ~50% FRR on real bonafide selfies with a plain background, plus
30-90s latency incompatible with this service's <2s budget).

RE-CALIBRATION 2026-07-16 (evening) — the original 0.35 threshold was
beaten same-day by a second real incident. Full numbers below, measured
with the ACTUAL `FaceDetector.detect()` bbox (not a manual crop) on every
image in `docs/plans/calibration/incident_urgut/`:

    face_area_ratio  = (bbox_w * bbox_h) / (frame_w * frame_h)
    face_width_ratio = bbox_w / frame_w

    bonafide (12 files, original/):
        face_area_ratio  0.0434 .. 0.2147
        face_width_ratio 0.2333 .. 0.5042
        (max on both = photo_2026-07-06_11-36-03.jpg, the "painted wood
        door" bonafide that is historically the hardest case for every
        other signal in this service too — see app/liveness.py comments)

    document spoof, profile 1 — urgut_v2_passport/passport_style_spoof_01.jpg
    (993x1275, bbox 748x799): face_area_ratio=0.4721, face_width_ratio=0.7533

    document spoof, profile 2 — the 2026-07-16 19:41 incident
    (993x1275, bbox 611x668): face_area_ratio=0.3224, face_width_ratio=0.6153
    THIS is the sample that slipped through the old 0.35 area threshold
    (0.3224 < 0.35) and was then ALSO waved through by passive-PAD's
    `_fuse()` (see app/liveness.py NN_TRUST_REAL fix, same incident) —
    verdict was `real`, combined_score=0.9997, before both fixes.

Both known document spoofs share the same 993x1275 frame size (a common
passport/ID-scan resolution) despite being different source images —
consistent with "photo of a printed/scanned ID document", the attack class
this gate targets.

Production threshold (app/config.py): `FACE_RATIO_REJECT=0.27` (area) is the
ONLY ratio wired into the `is_document` decision (see margins below).

    ratio   bonafide max   threshold   weaker spoof   margin above bonafide   margin below spoof
    area    0.2147         0.27        0.3224         +25.8%                  -16.3%

`face_width_ratio` is computed and reported alongside (diagnostic-only
field, like `frame_aspect_ratio` below) but is NOT part of the reject
decision. 2PAC review (2026-07-16) found width_ratio / sqrt(area_ratio) is
a near-constant ~1.09 (range 1.084-1.096) across all 14 measured
bonafide+spoof samples, because bbox aspect ratio (~0.89-0.94, RetinaFace's
inner-face crop) barely varies across this whole calibration set. In other
words `face_width_ratio` is mathematically close to
`1.09 * sqrt(face_area_ratio)` for every sample measured so far — gating on
it in ADDITION to area does not catch any attack that area misses (both
known document spoofs are already caught by area alone with clean margin —
see `test_2026_07_16_incident_caught_by_area_alone`), it only narrows the
effective margin against a real customer standing close to the camera
(margin above bonafide max for width would be only ~9.1%, vs ~25.8% for
area). That is pure FRR cost with no FAR benefit on current data, so it was
NOT wired into production. `check_face_geometry()` still accepts an
optional `width_threshold` parameter for a FUTURE, independently-collected
calibration to re-evaluate width as a genuinely orthogonal signal (e.g. if
bbox-aspect variance turns out to be larger on a bigger sample) — do not
pass it in production code without new numbers backing it up.

A document/ID photo held up to fill the camera's view produces a much
larger face-to-frame ratio than a normal selfie taken at arm's length or
further — a real, physically-grounded geometric difference, not a
heuristic tuned to one image. But see limitation #1 below: with n=2 spoof
samples the exact threshold value is still a judgment call with margin on
both sides, not a statistically tight bound.

KNOWN LIMITATIONS — read before trusting this as a strong security control:

1. **n=2 spoof samples, n=12 bonafide.** Two data points is more than one,
   but still not a distribution — the README for this calibration set
   (docs/plans/calibration/incident_urgut/README.md) puts the honest
   confidence interval at roughly +/-(15-20)% on n~12 per class. Treat the
   threshold above as "clearly better than 0.35, with real margin on real
   numbers" — not as a proven FAR/FRR guarantee.

2. **This dataset is phone selfies at arbitrary arm's-length distances,
   NOT verified sale-transaction camera frames.** The production checkout
   camera may have a different fixed distance/framing than these bonafide
   photos (taken for a different, unrelated incident). If the real POS
   camera sits closer to the customer than these phone selfies did, real
   sale frames could have a HIGHER baseline face_area_ratio than this
   calibration set shows, silently eating into the safety margin above
   (+25.8%). Confirm against a sample of actual sale-flow frames before
   fully trusting the FAR/FRR implied by the numbers above; monitor
   `document_photo` rejection rate after deploy as an early warning.

3. **Trivially evadable by a smarter attacker.** This gate only catches the
   SPECIFIC attack profile actually observed here (document held close,
   filling the frame). An attacker who holds the printed/screen photo
   further from the camera reduces both ratios to an ordinary selfie-like
   value and this gate will not fire — it falls through to passive-PAD,
   which judges on texture/recapture signals instead of composition (see
   the NN_TRUST_REAL fix in app/liveness.py for the corresponding
   defense-in-depth fix there). This is one cheap additional layer, not a
   replacement for the existing passive-PAD signals.

CONSIDERED AND MEASURED, NOT IMPLEMENTED:
   - EXIF Make/Model absence as a confirming signal (owner's suggestion):
     measured on this dataset and found NON-DISCRIMINATING — ALL 12
     bonafide AND both spoof samples lack camera EXIF (messenger
     recompression strips it from genuine phone selfies too, per this
     dataset's own README: several bonafide are noted as re-compressed by
     a messenger). Adding "no EXIF => more likely spoof" here would not
     separate the classes at all on real data and risks false confidence;
     not implemented.
   - Frame aspect ratio (~7:9 = 0.778, a common passport-photo aspect): both
     spoof samples ARE at 0.779 (993x1275) vs the bonafide files' uniform
     0.75 (960x1280 / 1080x1440, standard phone 3:4) — a real difference,
     and now confirmed on 2 independent spoof sources sharing the exact
     same frame size, but a genuine gallery-cropped or pre-cropped photo
     could share this aspect and be falsely rejected, and n=2 is still thin.
     Computed and exposed as a DIAGNOSTIC-ONLY field in the result/signals,
     NOT used in the reject decision.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GeometryCheckResult:
    """Outcome of the Layer 0a geometry check. Always constructible without
    raising — failure states are represented as data, not exceptions."""

    ran: bool  # False => bad input (should not happen given a real bbox); caller fails safe
    is_document: bool = False
    face_area_ratio: float = 0.0
    face_width_ratio: float = 0.0
    frame_aspect_ratio: float = 0.0  # diagnostic only, NOT part of the reject decision
    error: Optional[str] = None


def check_face_geometry(
    bbox: list[int],
    frame_shape: tuple[int, int],
    threshold: float,
    width_threshold: Optional[float] = None,
) -> GeometryCheckResult:
    """Compute face-to-frame ratios from an already-detected bbox + frame shape.

    `bbox` = [x, y, w, h] as returned by `FaceDetector.detect()` — pass the
    SAME bbox already computed for passive-PAD, do not re-run detection.
    `frame_shape` = (height, width), e.g. `image_bgr.shape[:2]`.
    `threshold` gates `face_area_ratio` (always active — this is the ONLY
    ratio production code passes, see app/main.py::_run_geometry_gate).
    `width_threshold` optionally ALSO gates `face_width_ratio` — `is_document`
    fires if EITHER ratio crosses its threshold (OR, not AND). Pass `None`
    (default; what production uses today) to keep area-only behavior. Kept
    for a FUTURE independent width calibration — see the module docstring
    for why the two ratios are correlated, not independent evidence, in the
    current calibration data, and do not wire this into production without
    new numbers.

    Never raises: any malformed input (should not happen with a real
    detector bbox, but defused anyway) degrades to `ran=False` so the
    caller can unconditionally fail safe to passive-PAD.
    """
    try:
        frame_h, frame_w = frame_shape
        _, _, face_w, face_h = bbox
        if frame_h <= 0 or frame_w <= 0 or face_w <= 0 or face_h <= 0:
            return GeometryCheckResult(ran=False, error="INVALID_DIMENSIONS")

        face_area_ratio = (face_w * face_h) / (frame_w * frame_h)
        face_width_ratio = face_w / frame_w
        frame_aspect_ratio = frame_w / frame_h

        is_document = face_area_ratio >= threshold
        if width_threshold is not None:
            is_document = is_document or face_width_ratio >= width_threshold

        return GeometryCheckResult(
            ran=True,
            is_document=is_document,
            face_area_ratio=round(face_area_ratio, 4),
            face_width_ratio=round(face_width_ratio, 4),
            frame_aspect_ratio=round(frame_aspect_ratio, 4),
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the request path
        return GeometryCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
