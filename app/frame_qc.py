"""Layer 0 — per-frame QC gate for the active-liveness pipeline
(POST /liveness/verdict), per docs/plans/FACEID_LIVENESS_ML_CORE_v1.md §2.0.

Distinct from app/geometry_check.py (Layer 0a, single-photo document/passport
composition gate used by /pad/check and friends) — this module gates
individual frames WITHIN a multi-frame liveness session before they are
handed to Layer 2 (active challenge) / Layer 3 (identity) / Layer 1
(passive PAD). A frame that fails here is dropped from the session, not
necessarily a spoof verdict — too few surviving frames becomes
verdict="incomplete" (UX "reshoot"), not a security accusation. See
app/main.py::_run_liveness_verdict for how per-frame QC failures are
aggregated into the session-level verdict.

Thresholds below are heuristic starting points (same family as the
already-deployed app/face_qc.py in face_id/tracker, NOT independently
calibrated against E-GAZ sale-flow camera frames) — flagged, not asserted as
measured.
"""
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.face_landmarks import FrameFace


@dataclass(frozen=True)
class FrameQCResult:
    valid: bool
    reason: Optional[str]  # NO_FACE | MULTIPLE_FACES | TOO_SMALL | BLURRY | TOO_DARK | TOO_BRIGHT | None
    metrics: dict = field(default_factory=dict)


# Minimum face bbox edge (px) in the ORIGINAL frame — below this the crop is
# too small for AdaFace/pose to be reliable. Placeholder, not calibrated on
# real sale-flow frames (see face_id/tracker MIN_FACE_SIZE=35 as a rough
# analog, but that is a door-camera domain, not phone-selfie-distance).
MIN_FACE_EDGE_PX = 60

# Laplacian-variance sharpness floor on the aligned 112x112 crop. Same
# formula as face_id/tracker/app/face_qc.py::_laplacian_var; threshold NOT
# re-calibrated for this project's camera/JPEG-quality domain.
MIN_SHARPNESS = 40.0

MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 235.0


def assess_frame(image_bgr: np.ndarray, face: Optional[FrameFace], aligned_112: Optional[np.ndarray] = None) -> FrameQCResult:
    """`aligned_112` — pass the ALREADY-computed landmark-aligned 112x112 crop
    when the caller needs it downstream anyway (e.g. Layer 3 embedding in
    app/main.py::_run_liveness_verdict), so this function does not redo the
    alignment warp a second time. Falls back to computing it here (backward
    compatible for standalone/test use) if omitted."""
    if face is None:
        return FrameQCResult(valid=False, reason="NO_FACE")
    if face.n_faces_detected > 1:
        return FrameQCResult(
            valid=False, reason="MULTIPLE_FACES",
            metrics={"n_faces_detected": face.n_faces_detected},
        )

    x1, y1, x2, y2 = face.bbox_xyxy
    edge = min(x2 - x1, y2 - y1)
    if edge < MIN_FACE_EDGE_PX:
        return FrameQCResult(valid=False, reason="TOO_SMALL", metrics={"face_edge_px": round(edge, 1)})

    if aligned_112 is None:
        from app.face_landmarks import LandmarkDetector
        aligned_112 = LandmarkDetector.align_112(image_bgr, face.kps)
    aligned = aligned_112
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())

    metrics = {
        "face_edge_px": round(edge, 1),
        "sharpness": round(sharp, 1),
        "brightness": round(bright, 1),
        "det_score": round(face.det_score, 3),
        "pose_yaw": round(face.pose_yaw, 1),
        "pose_pitch": round(face.pose_pitch, 1),
    }

    if sharp < MIN_SHARPNESS:
        return FrameQCResult(valid=False, reason="BLURRY", metrics=metrics)
    if bright < MIN_BRIGHTNESS:
        return FrameQCResult(valid=False, reason="TOO_DARK", metrics=metrics)
    if bright > MAX_BRIGHTNESS:
        return FrameQCResult(valid=False, reason="TOO_BRIGHT", metrics=metrics)

    return FrameQCResult(valid=True, reason=None, metrics=metrics)
