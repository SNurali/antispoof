"""Tests for app/liveness.py::_fuse — the recapture-vs-NN_TRUST_REAL fix
from the 2026-07-16 19:41 incident.

Uses the exact `signal_scores` measured on the real dataset (via
app.multisignal.analyze_face + LivenessEngine on the actual images) rather
than synthetic numbers, per the "measure, don't guess" rule for this
service. See docs/plans/calibration/incident_urgut/ and
app/liveness.py::_fuse docstring for the full calibration table.
"""

import pytest

from app.liveness import _fuse, pad_check_reason


def _signal_info(
    recapture: float,
    spoof_probability: float,
    lbp: float = 0.0,
    moire: float = 0.0,
    fft: float = 0.0,
    color: float = 0.0,
) -> dict:
    return {
        "signal_scores": {"recapture": recapture, "lbp": lbp, "moire": moire, "fft": fft, "color": color},
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


class TestFusePrintPatternOverride:
    """2026-07-22 incident: a sharp, high-resolution photo of a printed
    passport page scored verdict=live, combined_score=0.5671 on the real
    /pad/check endpoint (confirmed via a live HTTP call against the running
    service, correlation_id logged, no PII reproduced here — see
    app/liveness.py::_fuse docstring and app/multisignal.py's
    PRINT_PATTERN_FFT_MIN/PRINT_PATTERN_COLOR_MIN docstring for the full
    corpus numbers this override was calibrated against).

    Real signal_scores measured on the actual incident file (via the real
    FaceDetector + LivenessEngine, NOT reproduced as an image fixture here to
    avoid committing PII into the repo): nn_label=real, nn_score=0.5671,
    recapture=0.003, fft=0.6, color=0.6, spoof_probability=0.0914.
    """

    def test_2026_07_22_passport_fullpage_incident_now_rejected(self):
        """The exact numbers from the incident: recapture reads near-zero
        (sharp photo of a highly-textured printed page looks "detailed",
        i.e. real-like, to that signal — see the root-cause docstring), but
        fft (print halftone-dot periodicity) and color (near-zero
        chrominance, i.e. sepia/monochrome) both independently fire at 0.6 —
        the print-pattern override must now catch this even though nn_label
        is 'real' and spoof_probability (0.0914) is under every pre-existing
        soft threshold in `_fuse()`."""
        signal_info = _signal_info(
            recapture=0.003, spoof_probability=0.0914, lbp=0.0, moire=0.0, fft=0.6, color=0.6,
        )
        label, score = _fuse(nn_label=1, nn_score=0.5671, signal_info=signal_info)
        assert label == "spoof"
        assert score == pytest.approx(0.6)

    def test_override_disabled_by_flag_falls_through_to_old_behavior(self):
        """`print_pattern_override_enabled=False` must reproduce the exact
        pre-fix (buggy) verdict — this is the rollback path documented in
        app/config.py::PRINT_PATTERN_OVERRIDE_ENABLED."""
        signal_info = _signal_info(
            recapture=0.003, spoof_probability=0.0914, lbp=0.0, moire=0.0, fft=0.6, color=0.6,
        )
        label, score = _fuse(
            nn_label=1, nn_score=0.5671, signal_info=signal_info,
            print_pattern_override_enabled=False,
        )
        assert label == "real"
        assert score == pytest.approx(0.5671)

    def test_color_alone_without_fft_does_not_trigger(self):
        """Calibration finding: `color>=0.5` ALONE has a false positive in
        the faces-dataset/real corpus (fft=0.3, color=0.6, recapture=0.6735
        — a plausible desaturated/grayscale-filtered bonafide photo). The
        override MUST require both `fft` AND `color` to independently clear
        their thresholds — this test locks that AND-composite, not an OR."""
        signal_info = _signal_info(
            recapture=0.6735, spoof_probability=0.3781, lbp=0.0, moire=0.0, fft=0.3, color=0.6,
        )
        label, score = _fuse(nn_label=1, nn_score=0.9999, signal_info=signal_info)
        assert label == "real"

    def test_fft_alone_without_color_does_not_trigger(self):
        """Symmetric case: `fft>=0.5` alone is common in ordinary bonafide
        photos (mean fft 0.32-0.60 across the calibration corpus) and must
        NOT trigger the override on its own — only the AND-composite with a
        genuinely narrow-chroma `color` reading is discriminating."""
        signal_info = _signal_info(
            recapture=0.18, spoof_probability=0.111, lbp=0.0, moire=0.0, fft=0.6, color=0.0,
        )
        label, score = _fuse(nn_label=1, nn_score=0.991, signal_info=signal_info)
        assert label == "real"

    @pytest.mark.parametrize(
        "fft,color,recapture,spoof_probability,nn_score",
        [
            # Real measured bonafide values from the calibration corpus
            # (BONAFIDE_urgut_orig + faces-dataset/real, both < 0.5 on at
            # least one of fft/color so the AND-composite never fires).
            (0.3, 0.0, 0.0, 0.06, 0.9914),
            (0.6, 0.0, 0.5563, 0.2803, 0.9897),
            (0.6, 0.35, 0.7918, 0.4213, 0.9782),
            (0.6, 0.35, 0.8812, 0.4616, 0.9638),
        ],
    )
    def test_bonafide_corpus_values_still_pass(self, fft, color, recapture, spoof_probability, nn_score):
        signal_info = _signal_info(
            recapture=recapture, spoof_probability=spoof_probability, fft=fft, color=color,
        )
        label, score = _fuse(nn_label=1, nn_score=nn_score, signal_info=signal_info)
        assert label == "real"


class TestPadCheckReason:
    """2PAC review round 2 (2026-07-22): the print-pattern override must get
    its OWN /pad/check `reason` (PRINT_PATTERN_SPOOF), separate from the
    recapture override and every other passive-PAD spoof path (which all
    keep reason=PASSIVE_PAD_SPOOF) — so Умид's dashboard can filter false
    rejects by signal. `_fuse()` tags `signal_info["spoof_trigger"]` in
    place; `pad_check_reason()` (used directly by app/main.py's /pad/check
    handler) reads that tag. These tests exercise the real `_fuse()` mutation
    end-to-end (not a hand-built spoof_trigger key) so a refactor that stops
    tagging `signal_info` would be caught here."""

    def test_print_pattern_override_gets_dedicated_reason(self):
        """Same incident numbers as TestFusePrintPatternOverride above."""
        signal_info = _signal_info(
            recapture=0.003, spoof_probability=0.0914, lbp=0.0, moire=0.0, fft=0.6, color=0.6,
        )
        label, _score = _fuse(nn_label=1, nn_score=0.5671, signal_info=signal_info)
        assert pad_check_reason(label, signal_info) == "PRINT_PATTERN_SPOOF"

    def test_recapture_override_keeps_passive_pad_spoof_reason(self):
        """Same incident numbers as TestFuseRecaptureOverride's first test —
        must NOT be relabeled PRINT_PATTERN_SPOOF just because it is also a
        `_fuse()` override path."""
        signal_info = _signal_info(recapture=0.541, spoof_probability=0.3035, lbp=0.3)
        label, _score = _fuse(nn_label=1, nn_score=0.9997, signal_info=signal_info)
        assert pad_check_reason(label, signal_info) == "PASSIVE_PAD_SPOOF"

    def test_plain_ensemble_spoof_keeps_passive_pad_spoof_reason(self):
        """A spoof verdict reached via the soft ensemble thresholds (no
        override branch at all) must also stay PASSIVE_PAD_SPOOF."""
        signal_info = _signal_info(recapture=0.0, spoof_probability=0.7, lbp=0.0, moire=0.0)
        label, _score = _fuse(nn_label=1, nn_score=0.4, signal_info=signal_info)
        assert label == "spoof"
        assert pad_check_reason(label, signal_info) == "PASSIVE_PAD_SPOOF"

    def test_live_verdict_has_no_reason(self):
        signal_info = _signal_info(recapture=0.0, spoof_probability=0.015, lbp=0.0)
        label, _score = _fuse(nn_label=1, nn_score=0.9999, signal_info=signal_info)
        assert pad_check_reason(label, signal_info) is None
