"""Tests for app/liveness.py::_fuse — the recapture-vs-NN_TRUST_REAL fix
from the 2026-07-16 19:41 incident.

Uses the exact `signal_scores` measured on the real dataset (via
app.multisignal.analyze_face + LivenessEngine on the actual images) rather
than synthetic numbers, per the "measure, don't guess" rule for this
service. See docs/plans/calibration/incident_urgut/ and
app/liveness.py::_fuse docstring for the full calibration table.
"""

import pytest

from app.liveness import _fuse


def _signal_info(recapture: float, spoof_probability: float, lbp: float = 0.0, moire: float = 0.0) -> dict:
    return {
        "signal_scores": {"recapture": recapture, "lbp": lbp, "moire": moire},
        "spoof_probability": spoof_probability,
    }


class TestFuseRecaptureOverride:
    def test_2026_07_16_incident_now_rejected_despite_high_nn_confidence(self):
        """The exact numbers from the incident photo (measured with the real
        FaceDetector + LivenessEngine, confirmed against the owner's live
        production screenshot): nn_label=real, nn_score=0.9997,
        recapture=0.541, lbp=0.3, spoof_probability=0.3035.

        OLD behavior (before this fix): nn_very_confident_real (0.9997>=0.90)
        vetoed the recapture override => combined_label='real',
        combined_score=0.9997 (verified reproduced with the pre-fix code).

        NEW behavior: the veto is removed from this branch, so a confirmed
        recapture (lbp>0.1) always overrides an NN 'real' call.
        """
        signal_info = _signal_info(recapture=0.541, spoof_probability=0.3035, lbp=0.3)
        label, score = _fuse(nn_label=1, nn_score=0.9997, signal_info=signal_info)
        assert label == "spoof"
        assert score == pytest.approx(0.541)

    def test_passport_v2_spoof_now_rejected_despite_high_nn_confidence(self):
        """Real numbers from urgut_v2_passport/passport_style_spoof_01.jpg:
        nn_label=real, nn_score=0.9909, recapture=0.6309, lbp=0.3,
        spoof_probability=0.3589. Same bug, independently confirmed on a
        second real spoof sample."""
        signal_info = _signal_info(recapture=0.6309, spoof_probability=0.3589, lbp=0.3)
        label, score = _fuse(nn_label=1, nn_score=0.9909, signal_info=signal_info)
        assert label == "spoof"
        assert score == pytest.approx(0.6309)

    @pytest.mark.parametrize(
        "recapture,spoof_probability,nn_score",
        [
            (0.0, 0.015, 0.9999),
            (0.0, 0.015, 0.9735),
            (0.0, 0.015, 0.9947),
            (0.0, 0.015, 0.9994),
            (0.0, 0.03, 0.9996),
            (0.1176, 0.0679, 0.9995),
            (0.0235, 0.0256, 0.9994),
            (0.0, 0.015, 0.9028),
            (0.0, 0.015, 0.9992),
            (0.0, 0.025, 1.0),
            (0.5315, 0.2542, 0.7189),  # 2026-07-06 false-reject regression, lbp=0.0
        ],
    )
    def test_all_bonafide_original_still_pass_after_veto_removal(self, recapture, spoof_probability, nn_score):
        """Every bonafide sample in incident_urgut/original/ has lbp=0.0 (=>
        recapture_confirmed=False), so removing the NN_TRUST_REAL veto from
        the recapture branch must NOT change any bonafide verdict — the
        branch never fires for them regardless of NN confidence, veto or not.
        Real signal_scores from the calibration run (lbp omitted => 0.0)."""
        signal_info = _signal_info(recapture=recapture, spoof_probability=spoof_probability, lbp=0.0)
        label, score = _fuse(nn_label=1, nn_score=nn_score, signal_info=signal_info)
        assert label == "real"
        assert score == pytest.approx(nn_score)

    def test_bald_outdoor_selfie_hard_case_still_passes(self):
        """The historically hardest bonafide case (photo_2026-07-06_11-36-03.jpg,
        bald man, bright sun, wood-grain door): nn_label=spoof(0.5542),
        recapture=0.3148 (below RECAPTURE_SPOOF_THRESHOLD), lbp=0.0,
        spoof_probability=0.1567. Must still resolve to 'real' via the
        nn_label!=1 override branch (untouched by this fix)."""
        signal_info = _signal_info(recapture=0.3148, spoof_probability=0.1567, lbp=0.0)
        label, score = _fuse(nn_label=0, nn_score=0.5542, signal_info=signal_info)
        assert label == "real"

    def test_recapture_confirmed_still_requires_lbp_or_moire(self):
        """A high recapture score with NO texture/moire confirmation must NOT
        trigger the override (unchanged behavior — this is the 2026-07-06
        messenger-recompression fix, not touched by this incident fix)."""
        signal_info = _signal_info(recapture=0.9, spoof_probability=0.5, lbp=0.0, moire=0.0)
        label, score = _fuse(nn_label=1, nn_score=0.99, signal_info=signal_info)
        assert label == "real"
