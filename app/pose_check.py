"""Layer 0d — face-angle (yaw/pitch) gate for /pad/check.

Distinct from every OTHER Layer 0 gate in this repo (geometry_check.py,
blur_check.py): those are dependency-free arithmetic on the bbox RetinaFace
(`app/face_detect.py`) already returns. Angle needs actual pose, which
RetinaFace's Caffe SSD detector here does NOT provide (bbox only, no
landmarks) — the only pose source in this codebase is
`app/face_landmarks.py::LandmarkDetector` (SCRFD + landmark_3d_68 via
insightface buffalo_l), which is ONLY loaded when
`Settings.LIVENESS_ENDPOINTS_ENABLED=True` (app/main.py::_load_liveness_models)
— a deliberately optional, heavier dependency (insightface/onnxruntime + a
second detector pass, ~100-160ms CPU measured in app/face_landmarks.py's own
module docstring) not guaranteed provisioned on every deploy.

This module does NOT load its own detector — it takes an already-analyzed
`FrameFace` (see app/face_landmarks.py) so it can be called with whatever
`landmark_detector` singleton app/main.py already owns (same reuse pattern
as `_run_geometry_gate` reusing the bbox already computed for passive-PAD).
Callers MUST check `Settings.POSE_CHECK_ENABLED` AND that a landmark_detector
instance actually exists before calling this — see app/main.py::
_run_pose_gate for the full guard.

WHY (RZA, 2026-07-21): the reported bypass is a printed/screen photo held at
an ANGLE to the camera. `/pad/check` today has literally no signal about
face angle at all — not a bad threshold, an ABSENT gate. A live human can be
asked to face the camera; a static photo held at a steep angle to fool a
distance/geometry heuristic is exactly the scenario this closes, and it is
also a legitimate anti-replay property on its own (an attacker cannot
correct a printed photo's angle on request the way a live person can turn
back to frontal).

CALIBRATION (2026-07-21, RZA) — s001 (ONE subject, real captured video
frames, `_capture_staging/s001/20260720T191516/pose_measurements.json`,
insightface landmark_3d_68 pose, NOT this repo's own crop/pipeline but the
SAME pose convention `app/face_landmarks.py::FrameFace.pose_yaw` reads):

    frontal_1: yaw=-0.54   frontal_2: yaw=-5.37
    right15:   yaw=-12.2   left15:    yaw=19.64
    right30:   yaw=-32.42  left30:    yaw=32.79
    up (tilt): pitch=36.88, yaw=2.63
    down(tilt):pitch=-34.08, yaw=0.31

The "30-degree" turn label measured at ~32-33 degrees actual yaw — labels
are nominal, not exact. `POSE_YAW_REJECT_DEG=40.0` clears the observed
bona fide max (32.79) with only ~7 degree margin — THIN, on n=1 subject.
`POSE_PITCH_REJECT_DEG=45.0` similarly clears the observed up/down tilt max
(36.88) with ~8 degree margin, covering an ordinary "glance up/down" during
checkout without asking the customer to hold unnaturally still.

KNOWN LIMITATIONS:

1. **n=1 subject.** A wider population may naturally reach larger yaw/pitch
   during a normal checkout glance (children, wheelchair users, cluttered
   queue) than this one staged capture shows — the ~7-8 degree margins above
   are NOT a proven FRR bound.
2. **No real attack angle sample.** The angle a real attacker actually used
   to fool /pad/check is unknown — this threshold is set to "clearly wider
   than known bona fide, narrower than a document filling the geometry
   gate's own frame" but not verified against the real incident photo.
3. **DEFAULT DISABLED** (`POSE_CHECK_ENABLED=False`, see app/config.py) —
   unlike blur_check.py's gate, this one is NOT dependency-free: it only
   does anything when `LIVENESS_ENDPOINTS_ENABLED=True` AND the
   landmark_detector singleton loaded successfully at startup (insightface
   installed, buffalo_l weights provisioned). Flipping POSE_CHECK_ENABLED=True
   without LIVENESS_ENDPOINTS_ENABLED=True is a silent no-op (fail-open to
   passive-PAD unchanged, by design — see app/main.py::_run_pose_gate),
   never a startup failure and never a request failure — do not assume it is
   protecting anything until both flags are confirmed True AND
   `/health`'s `liveness_models_loaded` is confirmed true on the live host.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PoseCheckResult:
    """Outcome of the Layer 0d pose check. Always constructible without
    raising — failure states are represented as data, not exceptions."""

    ran: bool  # False => no landmark_detector result available; caller fails safe
    is_off_angle: bool = False
    pose_yaw: float = 0.0
    pose_pitch: float = 0.0
    error: Optional[str] = None


def check_face_pose(
    pose_yaw: Optional[float],
    pose_pitch: Optional[float],
    yaw_threshold_deg: float,
    pitch_threshold_deg: float,
) -> PoseCheckResult:
    """Pure comparison — takes already-computed yaw/pitch (degrees, from
    `FrameFace.pose_yaw`/`pose_pitch`) so this module never touches a model
    or an image itself; the caller (app/main.py::_run_pose_gate) owns
    running `LandmarkDetector.analyze()` and deciding whether to call this
    at all.

    Never raises: `None` inputs (detector found no face / analysis failed)
    degrade to `ran=False` so the caller can unconditionally fail safe to
    passive-PAD, same pattern as check_face_geometry/check_face_sharpness.
    """
    if pose_yaw is None or pose_pitch is None:
        return PoseCheckResult(ran=False, error="NO_POSE")
    is_off_angle = abs(pose_yaw) > yaw_threshold_deg or abs(pose_pitch) > pitch_threshold_deg
    return PoseCheckResult(
        ran=True,
        is_off_angle=is_off_angle,
        pose_yaw=round(pose_yaw, 2),
        pose_pitch=round(pose_pitch, 2),
    )
