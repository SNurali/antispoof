"""Tests for app/active_challenge.py — Layer 2 active challenge verification."""
import numpy as np

from app.active_challenge import verify_challenge
from app.config import Settings
from app.face_landmarks import FrameFace


def _eye_landmarks(mode: str) -> np.ndarray:
    """Synthetic (68, 3) landmark_68 array with only the eye-contour indices
    (36-41 right eye, 42-47 left eye — see face_landmarks.py::FrameFace for
    why these indices) populated; everything else stays zero (unused by
    eye_aspect_ratios). 'open' geometry gives EAR ~0.4 (well above any
    plausible threshold), 'closed' gives EAR ~0.04 (well below) — synthetic
    shapes chosen for a clean separation, NOT measured from a real blink
    (see app/config.py::LIVENESS_EAR_BLINK_MAX for why no real closed-eye
    calibration data exists yet)."""
    lmk = np.zeros((68, 3), dtype=np.float32)
    half_gap = 2.0 if mode == "open" else 0.2
    right_eye = {36: (0.0, 0.0), 37: (2.0, -half_gap), 38: (7.0, -half_gap),
                 39: (10.0, 0.0), 40: (7.0, half_gap), 41: (2.0, half_gap)}
    for i, (x, y) in right_eye.items():
        lmk[i] = (x, y, 0.0)
    for i, base in zip(range(42, 48), range(36, 42)):
        x, y, _ = lmk[base]
        lmk[i] = (x + 50.0, y, 0.0)  # left eye, offset so it doesn't overlap
    return lmk


def _mouth_landmarks(mode: str) -> np.ndarray:
    """Synthetic (68, 3) landmark array with only the 6 outer-mouth indices
    consumed by mouth_aspect_ratio() populated (48 left corner, 50/52 upper
    lip, 54 right corner, 56/58 lower lip — see face_landmarks.py's
    mouth-index verification comment for why these are the right indices);
    everything else stays zero. 'smile' geometry (wide, flat) gives
    MAR=20/2=10.0 (well above any plausible threshold); 'neutral' geometry
    gives MAR=10/4=2.5 — deliberately close to the REAL neutral-mouth
    baseline measured on a real photo in face_landmarks.py (2.56), so this
    synthetic case is a meaningful regression guard against the same number
    LIVENESS_MAR_SMILE_MIN's placeholder was set relative to, not an
    arbitrary shape."""
    lmk = np.zeros((68, 3), dtype=np.float32)
    width, gap = (20.0, 1.0) if mode == "smile" else (10.0, 2.0)
    lmk[48] = (0.0, 0.0, 0.0)
    lmk[54] = (width, 0.0, 0.0)
    lmk[50] = (width * 0.25, -gap, 0.0)
    lmk[58] = (width * 0.25, gap, 0.0)
    lmk[52] = (width * 0.75, -gap, 0.0)
    lmk[56] = (width * 0.75, gap, 0.0)
    return lmk


def _face(yaw: float, pitch: float = 0.0, landmark_68: np.ndarray = None) -> FrameFace:
    return FrameFace(
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        kps=None,
        pose_pitch=pitch,
        pose_yaw=yaw,
        pose_roll=0.0,
        det_score=0.9,
        n_faces_detected=1,
        landmark_68=landmark_68,
    )


def _settings() -> Settings:
    return Settings(SERVICE_TOKEN="")


class TestVerifyChallenge:
    def test_unknown_step_rejected(self):
        frames = [(0, _face(0.0))]
        result = verify_challenge(["NOT_A_REAL_STEP"], frames, _settings())
        assert result.passed is False
        assert result.reason == "UNSUPPORTED_STEP"

    def test_no_frames_rejected(self):
        result = verify_challenge(["TURN_LEFT"], [], _settings())
        assert result.passed is False
        assert result.reason == "NO_FRONTAL_REFERENCE"

    def test_no_frontal_reference_rejected(self):
        # every frame far off-center — no baseline to compare against
        frames = [(0, _face(35.0)), (1, _face(40.0))]
        result = verify_challenge(["TURN_LEFT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "NO_FRONTAL_REFERENCE"

    def test_turn_left_detected_passes(self):
        # frontal reference + a frame turned past the threshold (positive yaw)
        frames = [(0, _face(0.0)), (1, _face(25.0))]
        result = verify_challenge(["TURN_LEFT"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_turn_right_detected_passes(self):
        frames = [(0, _face(0.0)), (1, _face(-25.0))]
        result = verify_challenge(["TURN_RIGHT"], frames, _settings())
        assert result.passed is True

    def test_missing_step_fails_as_challenge_failed(self):
        # frontal + a turn in the WRONG direction only
        frames = [(0, _face(0.0)), (1, _face(25.0))]
        result = verify_challenge(["TURN_RIGHT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "TURN_RIGHT" in result.detail["missing_steps"]

    def test_static_series_fails_multi_step_challenge(self):
        """No rotation anywhere in the series (e.g. a printed photo with
        slight natural pose jitter) — must not pass a multi-step challenge."""
        frames = [(0, _face(-2.2)), (1, _face(-1.2)), (2, _face(3.9)), (3, _face(0.4))]
        result = verify_challenge(["TURN_RIGHT", "TURN_LEFT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_both_steps_satisfied_by_different_frames(self):
        frames = [(0, _face(0.0)), (1, _face(25.0)), (2, _face(0.0)), (3, _face(-25.0))]
        result = verify_challenge(["TURN_LEFT", "TURN_RIGHT"], frames, _settings())
        assert result.passed is True


class TestVerifyChallengeOrderByEvidence:
    """Фаза 3.1 (CHALLENGE_ENTROPY_SPRINT_v1.md §6.1) — direct requirement
    from Rustam's review §1 p.3: evidence for steps[i] must be found ONLY
    among frames with seq strictly greater than the seq that satisfied
    steps[i-1]. Before this change `verify_challenge` searched the WHOLE
    series regardless of order — a session presenting TURN_LEFT evidence
    before TURN_RIGHT evidence would satisfy a challenge asking for
    ["TURN_RIGHT", "TURN_LEFT"] just as well. This must no longer pass."""

    def test_reversed_order_fails_step_not_detected(self):
        """Same frames as test_both_steps_satisfied_by_different_frames
        (TURN_LEFT evidence at seq1, TURN_RIGHT evidence at seq3) — but the
        challenge now asks for the OPPOSITE order. Previously this passed
        (passed=True); it must now fail."""
        frames = [(0, _face(0.0)), (1, _face(25.0)), (2, _face(0.0)), (3, _face(-25.0))]
        result = verify_challenge(["TURN_RIGHT", "TURN_LEFT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "TURN_LEFT" in result.detail["missing_steps"]

    def test_correct_order_still_passes(self):
        """Same frames, requested in the order they actually occurred —
        must still pass (regression guard against over-tightening)."""
        frames = [(0, _face(0.0)), (1, _face(25.0)), (2, _face(0.0)), (3, _face(-25.0))]
        result = verify_challenge(["TURN_LEFT", "TURN_RIGHT"], frames, _settings())
        assert result.passed is True
        assert result.detail["step_evidence_seq"] == {"TURN_LEFT": 1, "TURN_RIGHT": 3}

    def test_evidence_reused_before_pointer_is_rejected(self):
        """A single frame satisfying BOTH directions is impossible by
        construction here, but a frame at seq=0 that would satisfy
        steps[1] must not be usable once the pointer has already advanced
        past it — evidence strictly EARLIER than the previous step's match
        cannot count, even if it is technically present in the series."""
        # TURN_RIGHT evidence only at seq0 (before any TURN_LEFT evidence
        # exists at all) — requesting TURN_LEFT first must not let TURN_RIGHT
        # "borrow" that earlier frame.
        frames = [(0, _face(-25.0)), (1, _face(0.0)), (2, _face(25.0))]
        result = verify_challenge(["TURN_LEFT", "TURN_RIGHT"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "TURN_RIGHT" in result.detail["missing_steps"]

    def test_blink_combined_with_turn_requires_correct_order(self):
        """Order-by-evidence applies across step TYPES too, not just
        TURN_LEFT/TURN_RIGHT — BLINK evidence before the requested TURN_LEFT
        must not satisfy a ["TURN_LEFT", "BLINK"] challenge if the blink
        happened first."""
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(0.0, landmark_68=_eye_landmarks("closed"))),  # BLINK evidence, seq1
            (2, _face(25.0, landmark_68=_eye_landmarks("open"))),   # TURN_LEFT evidence, seq2
        ]
        result = verify_challenge(["TURN_LEFT", "BLINK"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "BLINK" in result.detail["missing_steps"]

    def test_missing_middle_step_does_not_advance_pointer_for_next_step(self):
        """LOW finding, documenting test (2PAC code review, 2026-07-20):
        `verify_challenge`'s order-by-evidence pointer (`last_matched_seq`)
        only advances on a MATCH — a step that fails to find evidence
        (`continue`s into `missing`) leaves the pointer exactly where the
        PREVIOUS successfully-matched step left it. This documents that
        behavior explicitly with a 3-step challenge where the MIDDLE step
        has no evidence anywhere in the series: the third step's search
        must still start right after the FIRST step's evidence seq, not be
        blocked or otherwise shifted by the second step's failure.

        steps = [TURN_LEFT, TURN_RIGHT, NOD_UP]:
        - seq0: frontal reference (yaw=0, pitch=0)
        - seq1: TURN_LEFT evidence (yaw=+25) -> pointer advances to seq1
        - seq2: NOD_UP evidence (pitch=+25, yaw=0) — NOT a TURN_RIGHT match
          (yaw=0, needs <=-20) anywhere in seq>1, so TURN_RIGHT is reported
          missing and the pointer stays at seq1 (unchanged).
        - NOD_UP is then searched starting from seq>last_matched_seq(=1),
          i.e. seq2 is still in range and matches — proving the search
          window for the step AFTER a missing one was not narrowed/shifted
          by the missing step itself."""
        frames = [
            (0, _face(0.0, pitch=0.0)),
            (1, _face(25.0, pitch=0.0)),   # TURN_LEFT evidence
            (2, _face(0.0, pitch=25.0)),   # NOD_UP evidence, no TURN_RIGHT anywhere
        ]
        result = verify_challenge(["TURN_LEFT", "TURN_RIGHT", "NOD_UP"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert result.detail["missing_steps"] == ["TURN_RIGHT"]
        # The pointer for NOD_UP's search resumed right after TURN_LEFT's
        # seq (1), not after some seq that only a matched TURN_RIGHT would
        # have produced — evidenced by NOD_UP being found at seq2 at all.
        assert result.detail["step_evidence_seq"] == {"TURN_LEFT": 1, "NOD_UP": 2}


class TestVerifyChallengeBlink:
    """BLINK is implemented (2026-07-17) but deliberately not in the default
    pool — see app/config.py::LIVENESS_EAR_BLINK_MAX. These tests exercise
    the detection MECHANISM with synthetic geometry (open EAR~0.4 vs closed
    EAR~0.04, both far from the 0.20 cutoff in either direction) — they
    prove the wiring is correct, they do NOT stand in for a real-data
    threshold calibration."""

    def test_blink_is_a_supported_step(self):
        """BLINK must no longer be rejected as UNSUPPORTED_STEP — it moved
        from the old UNSUPPORTED_STEPS set into SUPPORTED_STEPS."""
        frames = [(0, _face(0.0, landmark_68=_eye_landmarks("open")))]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.reason != "UNSUPPORTED_STEP"

    def test_blink_detected_when_ear_dips_passes(self):
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(0.0, landmark_68=_eye_landmarks("closed"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("open"))),
        ]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_eyes_always_open_fails_blink(self):
        """A static photo (or a live face that never blinks in the captured
        window) must NOT pass a BLINK challenge."""
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("open"))),
        ]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "BLINK" in result.detail["missing_steps"]

    def test_missing_landmark_68_treated_as_no_evidence(self):
        """A frame with landmark_68=None (should not happen once the SCRFD
        pass succeeds, kept defensive) must not crash and must not count as
        blink evidence."""
        frames = [(0, _face(0.0, landmark_68=None))]
        result = verify_challenge(["BLINK"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_blink_combined_with_turn_requires_both(self):
        frames = [
            (0, _face(0.0, landmark_68=_eye_landmarks("open"))),
            (1, _face(25.0, landmark_68=_eye_landmarks("open"))),
            (2, _face(0.0, landmark_68=_eye_landmarks("closed"))),
        ]
        result = verify_challenge(["TURN_LEFT", "BLINK"], frames, _settings())
        assert result.passed is True


class TestVerifyChallengeNod:
    """NOD_UP/NOD_DOWN (CHALLENGE_ENTROPY_SPRINT_v1.md §4.1, Фаза 1;
    PROMOTED INTO THE DEFAULT POOL 2026-07-21, see app/config.py::
    LIVENESS_CHALLENGE_STEPS_POOL and LIVENESS_PITCH_NOD_MIN_DEG for the
    calibration rationale and the sign-convention caveat). Symmetric to
    TURN_LEFT/TURN_RIGHT's yaw check, so these tests mirror
    TestVerifyChallenge's TURN tests 1:1 with pitch instead of yaw."""

    def test_nod_up_is_a_supported_step(self):
        frames = [(0, _face(0.0))]
        result = verify_challenge(["NOD_UP"], frames, _settings())
        assert result.reason != "UNSUPPORTED_STEP"

    def test_nod_down_is_a_supported_step(self):
        frames = [(0, _face(0.0))]
        result = verify_challenge(["NOD_DOWN"], frames, _settings())
        assert result.reason != "UNSUPPORTED_STEP"

    def test_nod_up_detected_passes(self):
        # frontal reference (yaw=0) + a frame pitched past the threshold
        frames = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=25.0))]
        result = verify_challenge(["NOD_UP"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_nod_down_detected_passes(self):
        frames = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=-25.0))]
        result = verify_challenge(["NOD_DOWN"], frames, _settings())
        assert result.passed is True

    def test_nod_in_wrong_direction_fails(self):
        # frontal + pitch in the WRONG direction only
        frames = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=25.0))]
        result = verify_challenge(["NOD_DOWN"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "NOD_DOWN" in result.detail["missing_steps"]

    def test_static_series_fails_nod_challenge(self):
        """No pitch movement anywhere in the series — must not pass, same
        class of guard as TestVerifyChallenge::test_static_series_fails_
        multi_step_challenge."""
        frames = [(0, _face(0.0, pitch=-2.2)), (1, _face(0.0, pitch=1.2)),
                  (2, _face(0.0, pitch=3.9)), (3, _face(0.0, pitch=0.4))]
        result = verify_challenge(["NOD_UP", "NOD_DOWN"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_nod_reversed_order_fails_order_by_evidence(self):
        """Same order-by-evidence rule already covered for TURN in
        TestVerifyChallengeOrderByEvidence — NOD_DOWN evidence appears
        BEFORE NOD_UP evidence in seq, but the challenge asks for
        ["NOD_UP", "NOD_DOWN"] — must fail, not silently reuse the
        earlier-seq NOD_DOWN evidence out of order."""
        frames = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=-25.0)),
                  (2, _face(0.0, pitch=0.0)), (3, _face(0.0, pitch=25.0))]
        result = verify_challenge(["NOD_UP", "NOD_DOWN"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "NOD_DOWN" in result.detail["missing_steps"]

    def test_nod_correct_order_passes_with_evidence_seq_recorded(self):
        frames = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=-25.0)),
                  (2, _face(0.0, pitch=0.0)), (3, _face(0.0, pitch=25.0))]
        result = verify_challenge(["NOD_DOWN", "NOD_UP"], frames, _settings())
        assert result.passed is True
        assert result.detail["step_evidence_seq"] == {"NOD_DOWN": 1, "NOD_UP": 3}

    def test_nod_combined_with_turn_requires_both(self):
        frames = [
            (0, _face(0.0, pitch=0.0)),
            (1, _face(25.0, pitch=0.0)),   # TURN_LEFT evidence
            (2, _face(0.0, pitch=25.0)),   # NOD_UP evidence
        ]
        result = verify_challenge(["TURN_LEFT", "NOD_UP"], frames, _settings())
        assert result.passed is True


class TestNodDefaultPoolMembership:
    """RZA, 2026-07-21 (owner request: 4 challenge actions —
    left/right/up/down): NOD_UP/NOD_DOWN moved from "implemented but
    excluded" to genuine default-pool members. This class proves that at
    the REAL `Settings()` config layer (not a hand-built pool list), not
    just at the `verify_challenge`/SUPPORTED_STEPS mechanism level already
    covered above."""

    def test_nod_steps_are_in_the_real_default_pool(self):
        from app.config import Settings

        pool = {s.strip() for s in Settings(SERVICE_TOKEN="").LIVENESS_CHALLENGE_STEPS_POOL.split(",")}
        assert {"TURN_LEFT", "TURN_RIGHT", "NOD_UP", "NOD_DOWN"} == pool

    def test_blink_and_smile_still_excluded_from_default_pool(self):
        """Regression guard against accidentally widening the pool further
        than the owner's 4-action request — BLINK/SMILE stay excluded
        (uncalibrated, see app/config.py::LIVENESS_EAR_BLINK_MAX /
        LIVENESS_MAR_SMILE_MIN)."""
        from app.config import Settings

        pool = {s.strip() for s in Settings(SERVICE_TOKEN="").LIVENESS_CHALLENGE_STEPS_POOL.split(",")}
        assert "BLINK" not in pool
        assert "SMILE" not in pool

    def test_default_nod_min_deg_is_18(self):
        """Locks in the recalibrated value (20.0 -> 18.0, see app/config.py::
        LIVENESS_PITCH_NOD_MIN_DEG docstring for the s001-derived rationale)
        as a regression guard against a silent drift."""
        from app.config import Settings

        assert Settings(SERVICE_TOKEN="").LIVENESS_PITCH_NOD_MIN_DEG == 18.0

    def test_nod_evidence_at_default_threshold_boundary(self):
        """Using the REAL default Settings() (18.0 pitch threshold, not a
        hardcoded test constant) — a pitch of 19.0 (just above) must satisfy
        NOD_UP; 17.0 (just below) must not, mirroring
        TestCheckFacePose's boundary-exclusive convention in
        tests/test_pose_check.py for the sibling Layer 0d gate."""
        settings = Settings(SERVICE_TOKEN="")
        frames_pass = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=19.0))]
        result_pass = verify_challenge(["NOD_UP"], frames_pass, settings)
        assert result_pass.passed is True

        frames_fail = [(0, _face(0.0, pitch=0.0)), (1, _face(0.0, pitch=17.0))]
        result_fail = verify_challenge(["NOD_UP"], frames_fail, settings)
        assert result_fail.passed is False
        assert result_fail.reason == "STEP_NOT_DETECTED"

    def test_all_four_pool_steps_in_order_by_evidence(self):
        """End-to-end order-by-evidence (Phase 3.1) across ALL FOUR default
        pool steps in one challenge — TURN and NOD evidence interleaved,
        proving the pitch axis and yaw axis do not cross-contaminate each
        other's evidence search (a NOD_UP frame, yaw=0, must not accidentally
        satisfy a TURN step search window, and vice versa) using the REAL
        default Settings() thresholds."""
        settings = Settings(SERVICE_TOKEN="")
        frames = [
            (0, _face(0.0, pitch=0.0)),     # frontal reference
            (1, _face(25.0, pitch=0.0)),    # TURN_LEFT evidence
            (2, _face(0.0, pitch=25.0)),    # NOD_UP evidence
            (3, _face(-25.0, pitch=0.0)),   # TURN_RIGHT evidence
            (4, _face(0.0, pitch=-25.0)),   # NOD_DOWN evidence
        ]
        steps = ["TURN_LEFT", "NOD_UP", "TURN_RIGHT", "NOD_DOWN"]
        result = verify_challenge(steps, frames, settings)
        assert result.passed is True
        assert result.detail["step_evidence_seq"] == {
            "TURN_LEFT": 1, "NOD_UP": 2, "TURN_RIGHT": 3, "NOD_DOWN": 4,
        }

    def test_all_four_pool_steps_wrong_order_fails(self):
        """Same evidence frames as above, but requested in the WRONG order
        (NOD_DOWN before TURN_RIGHT actually occurred) — must fail, not
        silently reuse out-of-order evidence, same guard already proven for
        the 2-step TURN-only case in TestVerifyChallengeOrderByEvidence."""
        settings = Settings(SERVICE_TOKEN="")
        frames = [
            (0, _face(0.0, pitch=0.0)),
            (1, _face(25.0, pitch=0.0)),    # TURN_LEFT evidence
            (2, _face(0.0, pitch=25.0)),    # NOD_UP evidence
            (3, _face(-25.0, pitch=0.0)),   # TURN_RIGHT evidence
            (4, _face(0.0, pitch=-25.0)),   # NOD_DOWN evidence
        ]
        steps = ["TURN_LEFT", "NOD_DOWN", "NOD_UP", "TURN_RIGHT"]
        result = verify_challenge(steps, frames, settings)
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"


class TestVerifyChallengeSmile:
    """SMILE (CHALLENGE_ENTROPY_SPRINT_v1.md §4.1, Фаза 1) — implemented but
    deliberately not in the default pool (LIVENESS_MAR_SMILE_MIN is an
    even weaker placeholder than BLINK's, see app/config.py). These tests
    exercise the detection MECHANISM with synthetic geometry (smile
    MAR=10.0 vs neutral MAR=2.5, the latter deliberately close to the real
    neutral-mouth baseline measured in face_landmarks.py) — they prove the
    wiring is correct, they do NOT stand in for a real-data threshold
    calibration (no real smiling photo exists in this repo either)."""

    def test_smile_is_a_supported_step(self):
        frames = [(0, _face(0.0, landmark_68=_mouth_landmarks("neutral")))]
        result = verify_challenge(["SMILE"], frames, _settings())
        assert result.reason != "UNSUPPORTED_STEP"

    def test_smile_detected_when_mar_rises_passes(self):
        frames = [
            (0, _face(0.0, landmark_68=_mouth_landmarks("neutral"))),
            (1, _face(0.0, landmark_68=_mouth_landmarks("smile"))),
            (2, _face(0.0, landmark_68=_mouth_landmarks("neutral"))),
        ]
        result = verify_challenge(["SMILE"], frames, _settings())
        assert result.passed is True
        assert result.reason is None

    def test_mouth_always_neutral_fails_smile(self):
        """A static photo (or a live face that never smiles in the captured
        window) must NOT pass a SMILE challenge — mirrors
        TestVerifyChallengeBlink::test_eyes_always_open_fails_blink."""
        frames = [
            (0, _face(0.0, landmark_68=_mouth_landmarks("neutral"))),
            (1, _face(0.0, landmark_68=_mouth_landmarks("neutral"))),
        ]
        result = verify_challenge(["SMILE"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "SMILE" in result.detail["missing_steps"]

    def test_missing_landmark_68_treated_as_no_evidence_for_smile(self):
        frames = [(0, _face(0.0, landmark_68=None))]
        result = verify_challenge(["SMILE"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"

    def test_smile_combined_with_turn_requires_correct_order(self):
        """Order-by-evidence applies to SMILE too — SMILE evidence before
        the requested TURN_LEFT must not satisfy a ["TURN_LEFT", "SMILE"]
        challenge if the smile happened first."""
        frames = [
            (0, _face(0.0, landmark_68=_mouth_landmarks("neutral"))),
            (1, _face(0.0, landmark_68=_mouth_landmarks("smile"))),  # SMILE evidence, seq1
            (2, _face(25.0, landmark_68=_mouth_landmarks("neutral"))),  # TURN_LEFT evidence, seq2
        ]
        result = verify_challenge(["TURN_LEFT", "SMILE"], frames, _settings())
        assert result.passed is False
        assert result.reason == "STEP_NOT_DETECTED"
        assert "SMILE" in result.detail["missing_steps"]
