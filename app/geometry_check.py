"""Layer 0a — deterministic face-to-frame geometry gate (no ML, no network).

Reuses the SAME face bbox from `FaceDetector` (RetinaFace) that passive-PAD
already computes for every request — no additional model, no extra
inference cost, no network call. Runs in microseconds (pure arithmetic on
numbers already on hand) and cannot time out or be affected by GPU
contention, unlike app/document_check.py (the minicpm-v/Ollama Layer 0
attempt, kept in the repo but DISABLED — see that module's docstring for
why: ~50% FRR on real bonafide selfies with a plain background, plus
30-90s latency incompatible with this service's <2s budget).

Rationale (2026-07-16, incident_urgut calibration — measured with the
ACTUAL FaceDetector.detect() bbox on the real dataset images, not a manual
crop):

    face_area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)

    spoof    passport_style_spoof_01.jpg:            0.472
    bonafide (12 files, original/ folder):    0.043 .. 0.215

Clean margin between the bonafide max (0.215, incidentally the same
"painted wood door" bonafide that is historically the hardest case for
every other signal in this service — see app/liveness.py comments) and the
one spoof sample (0.472). A document/ID photo held up to fill the camera's
view produces a much larger face-to-frame ratio than a normal selfie taken
at arm's length or further, which is a real, physically-grounded geometric
difference — not a heuristic tuned to this specific image.

KNOWN LIMITATIONS — read before trusting this as a strong security control:

1. **n=1 spoof sample.** 0.472 is one data point, not a distribution. The
   chosen threshold (FACE_RATIO_REJECT, app/config.py) has margin on both
   sides of the two numbers we actually have — it is not a statistically
   tight bound. A second, independently-sourced document-spoof sample
   (different attacker, different holding distance/camera) is needed
   before treating this threshold as validated.

2. **This dataset is phone selfies at arbitrary arm's-length distances,
   NOT verified sale-transaction camera frames.** The production checkout
   camera may have a different fixed distance/framing than these bonafide
   photos (taken for a different, unrelated incident). If the real POS
   camera sits closer to the customer than these phone selfies did, real
   sale frames could have a HIGHER baseline face_area_ratio than this
   calibration set shows, silently eating into the safety margin. Confirm
   against a sample of actual sale-flow frames before fully trusting the
   FAR/FRR implied by the numbers above.

3. **Trivially evadable by a smarter attacker.** This gate only catches the
   SPECIFIC attack profile actually observed here (document held close,
   filling the frame). An attacker who holds the printed/screen photo
   further from the camera reduces face_area_ratio to an ordinary
   selfie-like value and this gate will not fire — it falls through to
   passive-PAD, which judges on texture/recapture signals instead, not
   composition. This is one cheap additional layer, not a replacement for
   the existing passive-PAD signals.

CONSIDERED AND MEASURED, NOT IMPLEMENTED:
   - EXIF Make/Model absence as a confirming signal (owner's suggestion):
     measured on this dataset and found NON-DISCRIMINATING — ALL 12
     bonafide AND the spoof sample lack camera EXIF (messenger
     recompression strips it from genuine phone selfies too, per this
     dataset's own README: several bonafide are noted as re-compressed by
     a messenger). Adding "no EXIF => more likely spoof" here would not
     separate the classes at all on real data and risks false confidence;
     not implemented.
   - Frame aspect ratio (~7:9 = 0.778, a common passport-photo aspect): the
     one spoof sample IS at 0.779 (993x1275) vs the bonafide files' uniform
     0.75 (960x1280 / 1080x1440, standard phone 3:4) — a real difference,
     but n=1 makes it too fragile to gate on by itself (a genuine
     gallery-cropped or pre-cropped photo could share this aspect and be
     falsely rejected). Computed and exposed as a DIAGNOSTIC-ONLY field in
     the result/signals, NOT used in the reject decision.
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
) -> GeometryCheckResult:
    """Compute face-to-frame ratios from an already-detected bbox + frame shape.

    `bbox` = [x, y, w, h] as returned by `FaceDetector.detect()` — pass the
    SAME bbox already computed for passive-PAD, do not re-run detection.
    `frame_shape` = (height, width), e.g. `image_bgr.shape[:2]`.

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

        return GeometryCheckResult(
            ran=True,
            is_document=face_area_ratio >= threshold,
            face_area_ratio=round(face_area_ratio, 4),
            face_width_ratio=round(face_width_ratio, 4),
            frame_aspect_ratio=round(frame_aspect_ratio, 4),
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the request path
        return GeometryCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}")
