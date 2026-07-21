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

As of 2026-07-20 (CHALLENGE_ENTROPY_SPRINT_v1.md §4, Фаза 1), also
NOD_UP/NOD_DOWN (pitch from the SAME landmark_3d_68 pose, symmetric to
TURN_LEFT/TURN_RIGHT's yaw check) and SMILE (mouth width/height ratio —
app/face_landmarks.py::mouth_aspect_ratio, SAME landmark_3d_68 raw 68
points as BLINK, no new model).

AS OF 2026-07-21 (owner request: 4 challenge actions — left/right/up/down),
NOD_UP/NOD_DOWN GRADUATED INTO THE DEFAULT POOL
(app/config.py::LIVENESS_CHALLENGE_STEPS_POOL is now
"TURN_LEFT,TURN_RIGHT,NOD_UP,NOD_DOWN") — see LIVENESS_PITCH_NOD_MIN_DEG's
docstring in app/config.py for why: real s001 device-captured pitch
evidence exists (intentional look-up/look-down tilts measured at
pitch=+36.88/-34.08), the same class of evidence TURN_LEFT/TURN_RIGHT's own
yaw threshold already relies on, and materially stronger than what BLINK or
SMILE have. BLINK and SMILE remain in the "capability exists, not yet
trusted" state below — implemented and reachable via SUPPORTED_STEPS,
deliberately NOT in the default pool, because neither has any real
same-domain capture (open+closed eye / neutral+smiling mouth) to calibrate
against, only literature/single-baseline placeholders.

BLINK IS NOT IN THE DEFAULT POOL. The EAR index mapping was verified against
a real photo (see face_landmarks.py), but LIVENESS_EAR_BLINK_MAX — the
open-vs-closed cutoff — is still an unverified literature placeholder with
no real closed-eye calibration data; see app/config.py for the full caveat,
including a same-domain sanity check that found one genuinely-open real eye
sitting uncomfortably close to a naive 0.20 cutoff. Detection is IMPLEMENTED
and reachable if a caller explicitly puts "BLINK" in a session's steps
(app/liveness_session.py never does this on its own, since the pool
excludes it) — this is a deliberate "capability exists, not yet trusted as
a production security gate" state, not a half-finished feature. The mouth
index mapping for SMILE was verified the same way (real photo, same method
— see face_landmarks.py); LIVENESS_MAR_SMILE_MIN is an EVEN WEAKER
placeholder than LIVENESS_EAR_BLINK_MAX (no literature-cited MAR-for-smile
constant exists to anchor it against, only a single neutral-mouth baseline
measurement — see app/config.py for the full caveat).

SIGN-CONVENTION CAVEAT (read before trusting TURN_LEFT vs TURN_RIGHT, and
identically before trusting NOD_UP vs NOD_DOWN): insightface's raw pose
yaw/pitch sign (positive = ? / negative = ?) is used here with the
assumption "positive yaw = subject turns their face toward the camera's
right (viewer's right), i.e. their own left" and, symmetrically, "positive
pitch = subject tilts their chin up" — these are common conventions for
this pose model. NEITHER has been confirmed against a real, labeled "person
turns/nods on command" capture from an E-GAZ device — a real capture DOES
exist now (s001, app/pose_check.py) whose "left15/left30"/"up (tilt)"
labels happen to align with both assumptions above, but that capture's own
labeling protocol (camera-observed direction vs. subject-self-reported
direction) is not preserved/reviewable in this repo, only the aggregate
pitch/yaw numbers are — so this counts as mildly SUPPORTIVE, not
CONFIRMING, evidence, and it applies EQUALLY to yaw and pitch (pitch is not
worse-off than yaw here, despite being the newer addition). If either
mapping is backwards, the corresponding pair (TURN_LEFT/TURN_RIGHT or
NOD_UP/NOD_DOWN) swaps meaning but the SECURITY property (some real
rotation happened, in some direction, at the requested moment) still holds
up to a mirrored labeling — what would NOT be caught by a sign error is a
UX bug (wrong on-screen instruction), not a spoof getting through. Verify
with real device captures before relying on either LEFT/RIGHT or UP/DOWN
distinctions for anything beyond "a rotation occurred".
"""
from dataclasses import dataclass, field
from typing import Optional

from app.config import Settings
from app.face_landmarks import FrameFace, min_eye_aspect_ratio, mouth_aspect_ratio

SUPPORTED_STEPS = {"TURN_LEFT", "TURN_RIGHT", "BLINK", "NOD_UP", "NOD_DOWN", "SMILE"}


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
    order. Verifies EVERY requested step occurred, in the REQUESTED ORDER
    (Phase 3.1, CHALLENGE_ENTROPY_SPRINT_v1.md §6.1 — direct requirement from
    Rustam's review §1 p.3: this is the FIRST anti-tamper contour, run before
    and independent of Laravel's own M2 `captured_at` validation).

    ORDER-BY-EVIDENCE: evidence for `steps[i]` is searched only among frames
    with `seq` STRICTLY GREATER than the `seq` where `steps[i-1]`'s evidence
    was found (a monotonically advancing pointer over `seq`, which is always
    present and client-assigned — unlike the optional `captured_at`, see
    Phase 3.2). Before this change the search was `any(... for _, f in
    frames_with_seq)` over the WHOLE series regardless of order — a
    TURN_RIGHT motion captured BEFORE a requested TURN_LEFT would silently
    satisfy a `["TURN_LEFT", "TURN_RIGHT"]` challenge. This is a deliberate
    TIGHTENING: a session that used to pass with frames in the wrong order
    now fails with `STEP_NOT_DETECTED`, same reason code as "step not found
    at all" (the client-facing/Laravel-facing signal does not need to
    distinguish "never happened" from "happened too early" — see
    docs/LIVENESS_CONTRACT_v1.md §2.1)."""
    for step in steps:
        if step not in SUPPORTED_STEPS:
            return ActiveChallengeResult(
                passed=False, reason="UNSUPPORTED_STEP",
                detail={"step": step, "note": "unknown step"},
            )

    if not frames_with_seq:
        return ActiveChallengeResult(passed=False, reason="NO_FRONTAL_REFERENCE")

    # Defensive sort — callers are documented/expected to already pass this
    # ordered by seq (see docstring above and app/main.py's call site), but
    # the order-by-evidence pointer logic below is only correct if that
    # invariant actually holds, so it costs nothing to guarantee it here too.
    frames_with_seq = sorted(frames_with_seq, key=lambda item: item[0])

    frontal_max = settings.LIVENESS_YAW_FRONTAL_MAX_DEG
    turn_min = settings.LIVENESS_YAW_TURN_MIN_DEG
    ear_blink_max = settings.LIVENESS_EAR_BLINK_MAX
    nod_min = settings.LIVENESS_PITCH_NOD_MIN_DEG
    mar_smile_min = settings.LIVENESS_MAR_SMILE_MIN

    has_frontal = any(abs(f.pose_yaw) <= frontal_max for _, f in frames_with_seq)
    if not has_frontal:
        return ActiveChallengeResult(
            passed=False, reason="NO_FRONTAL_REFERENCE",
            detail={"yaws": [round(f.pose_yaw, 1) for _, f in frames_with_seq]},
        )

    detail: dict = {"yaws": {seq: round(f.pose_yaw, 1) for seq, f in frames_with_seq}}
    if any(step == "BLINK" for step in steps):
        # Diagnostic only — computed over the WHOLE series (not windowed by
        # the order pointer below), same as before this change, purely for
        # debugging/audit visibility.
        detail["min_ear"] = {
            seq: round(min_eye_aspect_ratio(f.landmark_68), 3)
            for seq, f in frames_with_seq if f.landmark_68 is not None
        }
    if any(step in ("NOD_UP", "NOD_DOWN") for step in steps):
        # Diagnostic only, same pattern as "yaws"/"min_ear" above — the
        # WHOLE series, not windowed by the order pointer below.
        detail["pitches"] = {seq: round(f.pose_pitch, 1) for seq, f in frames_with_seq}
    if any(step == "SMILE" for step in steps):
        detail["mar"] = {
            seq: round(mouth_aspect_ratio(f.landmark_68), 3)
            for seq, f in frames_with_seq if f.landmark_68 is not None
        }

    def _matches(step: str, face: FrameFace) -> bool:
        if step == "TURN_LEFT":
            return face.pose_yaw >= turn_min
        if step == "TURN_RIGHT":
            return face.pose_yaw <= -turn_min
        if step == "NOD_UP":
            return face.pose_pitch >= nod_min
        if step == "NOD_DOWN":
            return face.pose_pitch <= -nod_min
        if step == "SMILE":
            if face.landmark_68 is None:
                return False
            return mouth_aspect_ratio(face.landmark_68) >= mar_smile_min
        # BLINK — see module docstring: implemented, not in the default pool
        if face.landmark_68 is None:
            return False
        return min_eye_aspect_ratio(face.landmark_68) <= ear_blink_max

    missing: list[str] = []
    step_evidence_seq: dict[str, int] = {}  # step -> seq of the frame that satisfied it
    last_matched_seq: Optional[int] = None
    for step in steps:
        match_seq = next(
            (
                seq for seq, f in frames_with_seq
                if (last_matched_seq is None or seq > last_matched_seq) and _matches(step, f)
            ),
            None,
        )
        if match_seq is None:
            missing.append(step)
            continue
        step_evidence_seq[step] = match_seq
        last_matched_seq = match_seq

    if step_evidence_seq:
        # Consumed by Phase 3.3 (timing-window validation, app/main.py) to
        # know WHICH frame's captured_at to measure each step's delay
        # against — see app/main.py::_validate_step_windows.
        detail["step_evidence_seq"] = step_evidence_seq

    if missing:
        detail["missing_steps"] = missing
        return ActiveChallengeResult(passed=False, reason="STEP_NOT_DETECTED", detail=detail)

    return ActiveChallengeResult(passed=True, reason=None, detail=detail)
