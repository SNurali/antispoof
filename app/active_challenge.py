"""Layer 2 — active challenge-response verification.

Per docs/plans/FACEID_LIVENESS_ML_CORE_v1.md §2.2: this MUST be an
obligatory gate ("клиент никогда не шлёт `passed`" applied to liveness), not
one signal blended into a weighted score — a perfect passive-PAD score must
NOT be able to compensate for a failed/faked active challenge, because that
would silently reproduce the exact vulnerability this whole project exists
to close. See app/main.py::_run_liveness_verdict for the gate ORDER (Layer 2
runs and can reject BEFORE Layer 3/Layer 1 are even computed).

SUPPORTED: TURN_LEFT, TURN_RIGHT (yaw from landmark_3d_68 pose) and, as of
2026-07-17, BLINK (EAR from the SAME landmark_3d_68 model's raw 68 points —
see app/face_landmarks.py::eye_aspect_ratios). No extra model or dependency
for either: landmark_3d_68 was already loaded for Layer 0/3, only its pose
output was being read before.

BLINK IS NOT IN THE DEFAULT POOL (app/config.py::LIVENESS_CHALLENGE_STEPS_POOL
stays "TURN_LEFT,TURN_RIGHT"). The EAR index mapping was verified against a
real photo (see face_landmarks.py), but LIVENESS_EAR_BLINK_MAX — the
open-vs-closed cutoff — is still an unverified literature placeholder with
no real closed-eye calibration data; see app/config.py for the full caveat,
including a same-domain sanity check that found one genuinely-open real eye
sitting uncomfortably close to a naive 0.20 cutoff. Detection is IMPLEMENTED
and reachable if a caller explicitly puts "BLINK" in a session's steps
(app/liveness_session.py never does this on its own, since the pool
excludes it) — this is a deliberate "capability exists, not yet trusted as
a production security gate" state, not a half-finished feature.

SIGN-CONVENTION CAVEAT (read before trusting TURN_LEFT vs TURN_RIGHT):
insightface's raw pose yaw sign (positive = ? / negative = ?) is used
here with the assumption "positive yaw = subject turns their face toward
the camera's right (viewer's right), i.e. their own left" — this is the
common convention for this pose model, but it has NOT been confirmed
against a real, labeled "person turns left on command" capture from an
E-GAZ device. If this mapping is backwards, TURN_LEFT and TURN_RIGHT swap
meaning but the SECURITY property (some real rotation happened, in some
direction, at the requested moment) still holds up to a mirrored labeling —
what would NOT be caught by a sign error is a UX bug (wrong on-screen
instruction), not a spoof getting through. Verify with real device captures
before relying on the LEFT/RIGHT distinction for anything beyond "a
rotation occurred".
"""
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.face_landmarks import FrameFace, min_eye_aspect_ratio

SUPPORTED_STEPS = {"TURN_LEFT", "TURN_RIGHT", "BLINK"}


@dataclass(frozen=True)
class ActiveChallengeResult:
    passed: bool
    reason: Optional[str]  # None | UNSUPPORTED_STEP | STEP_NOT_DETECTED | NO_FRONTAL_REFERENCE
    detail: dict = field(default_factory=dict)


def verify_challenge(
    steps: list[str],
    frames_with_seq: list[tuple[int, FrameFace]],  # Layer-0-QC-passed frames, ORDERED by seq
    settings: Settings,
) -> ActiveChallengeResult:
    """`frames_with_seq` are the frames that survived Layer 0 QC, in capture
    order. Verifies EVERY requested step occurred somewhere in the series
    (not tied to a specific frame index — the client is not required to
    label which frame belongs to which step, the server looks for evidence
    of each requested motion across the whole valid series)."""
    for step in steps:
        if step not in SUPPORTED_STEPS:
            return ActiveChallengeResult(
                passed=False, reason="UNSUPPORTED_STEP",
                detail={"step": step, "note": "unknown step"},
            )

    if not frames_with_seq:
        return ActiveChallengeResult(passed=False, reason="NO_FRONTAL_REFERENCE")

    frontal_max = settings.LIVENESS_YAW_FRONTAL_MAX_DEG
    turn_min = settings.LIVENESS_YAW_TURN_MIN_DEG
    ear_blink_max = settings.LIVENESS_EAR_BLINK_MAX

    has_frontal = any(abs(f.pose_yaw) <= frontal_max for _, f in frames_with_seq)
    if not has_frontal:
        return ActiveChallengeResult(
            passed=False, reason="NO_FRONTAL_REFERENCE",
            detail={"yaws": [round(f.pose_yaw, 1) for _, f in frames_with_seq]},
        )

    detail: dict = {"yaws": {seq: round(f.pose_yaw, 1) for seq, f in frames_with_seq}}
    missing = []
    for step in steps:
        if step == "TURN_LEFT":
            found = any(f.pose_yaw >= turn_min for _, f in frames_with_seq)
        elif step == "TURN_RIGHT":
            found = any(f.pose_yaw <= -turn_min for _, f in frames_with_seq)
        else:  # BLINK — see module docstring: implemented, not in the default pool
            ears = {seq: round(min_eye_aspect_ratio(f.landmark_68), 3)
                    for seq, f in frames_with_seq if f.landmark_68 is not None}
            detail.setdefault("min_ear", ears)
            found = any(v <= ear_blink_max for v in ears.values())
        if not found:
            missing.append(step)

    if missing:
        detail["missing_steps"] = missing
        return ActiveChallengeResult(passed=False, reason="STEP_NOT_DETECTED", detail=detail)

    return ActiveChallengeResult(passed=True, reason=None, detail=detail)
