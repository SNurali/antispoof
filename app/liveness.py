"""Model loading and inference for liveness detection."""

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.multisignal import (
    analyze_face,
    SIGNAL_WEIGHTS,
    RECAPTURE_SPOOF_THRESHOLD,
    PRINT_PATTERN_FFT_MIN,
    PRINT_PATTERN_COLOR_MIN,
)

# HISTORICAL CONSTANT — no longer read anywhere in `_fuse` (MF DOOM review,
# 2026-07-16: the `nn_very_confident_real` variable that used to gate on
# this was removed from the recapture-override branch below, and it was
# never computed/logged anywhere else — this constant is currently dead
# code). Kept only as a documented data point (0.90 was the confidence line
# the old, now-disproven, veto used) in case a future revision wants to
# reintroduce a similar guard with new evidence. Do not resurrect a
# `not nn_very_confident_real` condition on the recapture branch without
# re-reading the incident fix below first.
#
# Above this NN real-confidence, the OLD code trusted the trained CNN over
# the resolution-sensitive `recapture` cue. Rationale from testing: genuine
# photo/screen fakes never pushed the CNN past ~0.85 real-confidence, while
# a live (even low-res) webcam face scores ~0.99. That assumption was
# DISPROVEN 2026-07-16: a high-quality passport-style photo spoof
# (docs/plans/calibration/incident_urgut/urgut_v2_passport/passport_style_spoof_01.jpg,
# and again same-day with a second incident photo) scored nn_score=0.9909 /
# 0.9997 — comfortably above this 0.90 line — while `recapture` (0.631 /
# 0.541) correctly flagged it AND was confirmed by `lbp` (0.300 in both
# cases). The veto let both spoofs through with combined_label=real. See
# `_fuse` below for the fix.
NN_TRUST_REAL = 0.90

# Below this NN spoof-confidence, the raw CNN call is treated as "weak, not a
# confirmed detection" and can be overridden back to real by the independent
# multi-signal analysis (mirrors NN_TRUST_REAL, the symmetric case: the NN can
# be wrong in the *false-reject* direction too, not just false-accept).
NN_TRUST_SPOOF = 0.90

# Max multi-signal `spoof_probability` allowed when overriding a weak NN
# "spoof" call back to "real". Calibrated on incident_urgut (n=11 spoof,
# spoof_probability range 0.297-0.578) vs the 2026-07-06 11:36 false-reject
# (bald outdoor selfie against a wood-grain door, spoof_probability=0.157):
# 0.20 sits in the ~0.14 gap between them with margin on both sides.
SIGNAL_TRUST_REAL_MAX = 0.20


def _fuse(
    nn_label: int,
    nn_score: float,
    signal_info: dict,
    print_pattern_override_enabled: bool = True,
) -> tuple[str, float]:
    """Combine NN output with multi-signal analysis into a final verdict.

    A high `recapture` score (confirmed by `lbp`/`moire`) is a physical,
    upscale-robust cue that the frame is a low-detail screen/print recapture;
    it OVERRIDES the NN "real" verdict unconditionally (this is exactly the
    class of fake the NN misses — see the 2026-07-16 fix below for why this
    is no longer gated on NN confidence). Otherwise fall back to the NN,
    letting weaker signals tip borderline cases to spoof.

    2026-07-06 incident fix: a real phone selfie re-compressed by a messenger
    (960x1280, 148KB vs 180-370KB for the same resolution in the calibration
    set) scored recapture=0.53 — inside the observed spoof range (0.43-0.93 on
    the "urgut" incident set, n=11) purely from JPEG-recompression detail loss,
    not an actual recapture. Raising RECAPTURE_SPOOF_THRESHOLD cannot fix this:
    the false-positive score (0.53) sits INSIDE the spoof distribution, so any
    threshold that excludes it also drops the weakest real spoof (0.43).
    Instead require a second, independent confirming signal before the
    recapture override fires. On the calibration set (n=10 real / n=11 urgut
    spoof) `lbp` cleanly separates the classes with zero overlap: every real
    sample (incl. the false-positive photo) scores lbp=0.0, every spoof sample
    scores lbp>=0.30 (texture uniformity from print/screen recapture). `moire`
    is OR'd in as a second independent confirming cue (screen pixel-grid
    artifacts) since it did not fire on this dataset but is a plausible signal
    for other recapture profiles (e.g. screen replay vs this print/scan-like
    incident). Risk: on out-of-distribution recaptures where BOTH lbp and
    moire fail to fire (e.g. a very high quality print/scan with naturally
    varied texture), the override no longer catches them and we fall through
    to the NN — this is a real trade-off, not eliminated, only reduced;
    n=10/11 calibration set is a smoke test (README caveat), not a guarantee.

    2026-07-16 19:41 incident fix (`NN_TRUST_REAL` veto removed from this
    branch): a high-quality passport-style photo held up to the camera hit
    recap=0.541/0.631 (>= RECAPTURE_SPOOF_THRESHOLD) with recapture_confirmed
    True (lbp=0.300 > 0.1) — by design this should have returned "spoof" —
    but the CNN itself scored nn_score=0.9997/0.9909 (real), tripping the old
    `not nn_very_confident_real` veto (NN_TRUST_REAL=0.90) and falling
    through to `return "real", nn_score`. Verified `combined_label=real,
    combined_score=0.9997` reproduced with the OLD code on the real incident
    photo before this fix (see calibration table in
    docs/plans/calibration/incident_urgut/). The veto is now REMOVED from
    this branch: on the full calibration set (12 bonafide + 11 "urgut"
    print/screen attacks + 2 passport-style photo attacks, all measured with
    the real FaceDetector + LivenessEngine), every single bonafide sample has
    lbp=0.0 (=> recapture_confirmed=False regardless of NN confidence) — so
    removing the veto changes ZERO bonafide verdicts on known data while
    fixing both passport-style spoofs. Risk (documented, not eliminated): a
    genuine low-res live capture that somehow also trips lbp/moire (texture
    signals, computed on a DIFFERENT 160x160 downscaled crop than
    `recapture`'s native-res crop) would now hard-reject instead of being
    saved by NN confidence — no such case exists in the current calibration
    set, but the risk is real given only n=12 bonafide. Owner's directive
    (2026-07-16) is to bias toward this trade-off: reject a borderline live
    frame over letting a document spoof through.

    2026-07-22 incident fix (print-pattern override added): a SHARP,
    high-resolution photo of a full printed passport page (held at normal
    selfie distance, not filling the frame — so the Layer 0a geometry gate's
    face_area_ratio=0.0735 stayed far below its 0.27 threshold too) scored
    nn_score=0.5671 (barely-confident "real", essentially a coin flip) with
    recapture=0.003 — recapture reads this as "high detail, real-like"
    because a sharp close-up of a printed page's own halftone/text/fibre
    texture genuinely IS high-frequency detail; recapture cannot distinguish
    "detailed because real skin" from "detailed because sharp photo of a
    highly-textured printed surface". This is a blind spot of that signal
    for THIS attack sub-class (full-page document photographed sharply),
    distinct from the low-detail screen/print recapture class the existing
    override above targets. `fft`=0.6 (print halftone-dot periodicity) and
    `color`=0.6 (near-zero chrominance spread — a sepia/monochrome scan)
    BOTH correctly fired on this frame, but at their production ensemble
    weights (0.05 / 0.10 of SIGNAL_WEIGHTS) only contributed
    spoof_probability=0.0914 — under every soft threshold below (nearest is
    `> 0.1`). Adding a dedicated, symmetric override — like the recapture
    override above, but for the "print pattern + desaturated tone" signature
    instead of "low detail" — catches this without re-weighting the whole
    ensemble (which would risk destabilizing every other already-calibrated
    case). See app/multisignal.py's PRINT_PATTERN_FFT_MIN/
    PRINT_PATTERN_COLOR_MIN docstring for the full corpus numbers (0 hits
    across 79 bonafide/unverified samples — 12 confirmed bonafide + 42
    unverified real + 25 unverified fake — 1/1 catch on this incident).
    `print_pattern_override_enabled` defaults True but is threaded through
    from app.config.Settings.PRINT_PATTERN_OVERRIDE_ENABLED via
    LivenessEngine's constructor — flip that flag to False to revert this
    specific change without a code rollback if it turns out to cost real FRR
    in production traffic this repo's calibration corpus cannot see.
    """
    recap = signal_info["signal_scores"].get("recapture", 0.0)
    spoof_prob = signal_info["spoof_probability"]
    lbp = signal_info["signal_scores"].get("lbp", 0.0)
    moire = signal_info["signal_scores"].get("moire", 0.0)
    recapture_confirmed = lbp > 0.1 or moire > 0.1

    if recap >= RECAPTURE_SPOOF_THRESHOLD and recapture_confirmed:
        # Tag the trigger so pad_check_reason() below can tell this apart
        # from the print-pattern override — same "spoof" verdict, DIFFERENT
        # reason string in the /pad/check response (2026-07-22, RZA, for
        # 2PAC review round 2: Умид needs to filter false-reject rate by
        # signal, and a single shared PASSIVE_PAD_SPOOF reason hid that).
        signal_info["spoof_trigger"] = "recapture_override"
        return "spoof", max(recap, spoof_prob)

    fft = signal_info["signal_scores"].get("fft", 0.0)
    color = signal_info["signal_scores"].get("color", 0.0)
    if print_pattern_override_enabled and fft >= PRINT_PATTERN_FFT_MIN and color >= PRINT_PATTERN_COLOR_MIN:
        signal_info["spoof_trigger"] = "print_pattern_override"
        return "spoof", max(fft, color, spoof_prob)

    if nn_label != 1:
        # 2026-07-06 incident fix #2: a real outdoor selfie (bright sun, bald
        # head, high-contrast wood-grain door background) made the CNN itself
        # call "spoof" at only 0.554 confidence, while every independent
        # texture/color/moire/sharpness signal read 0.0 and recapture (0.315)
        # stayed below the recapture-override threshold — spoof_probability
        # 0.157 was BELOW every genuine urgut spoof (min 0.297). The old logic
        # trusted any NN "spoof" label unconditionally, with no path back to
        # "real" even when every independent signal disagreed. Require the
        # NN to not be highly confident (<0.90, symmetric with NN_TRUST_REAL)
        # AND no recapture/texture confirmation AND spoof_probability clearly
        # in the real range before overriding a weak/wrong NN spoof call.
        nn_weak_spoof = nn_score < NN_TRUST_SPOOF
        signals_say_real = spoof_prob < SIGNAL_TRUST_REAL_MAX and not recapture_confirmed and recap < RECAPTURE_SPOOF_THRESHOLD
        if nn_weak_spoof and signals_say_real:
            return "real", 1.0 - spoof_prob
        return "spoof", max(nn_score, spoof_prob)
    if spoof_prob > 0.6:
        return "spoof", spoof_prob
    if spoof_prob > 0.35 and nn_score < 0.7:
        return "spoof", max(nn_score * 0.5, spoof_prob)
    if spoof_prob > 0.1 and nn_score < 0.6:
        return "spoof", max(nn_score * 0.4, spoof_prob)
    return "real", nn_score


def pad_check_reason(label: str, signal_info: dict) -> Optional[str]:
    """Map an engine.predict()/predict_batch() verdict to the /pad/check
    `reason` string (2026-07-22, RZA, 2PAC review round 2).

    Every spoof path through `_fuse()` used to collapse to the same
    reason="PASSIVE_PAD_SPOOF", indistinguishable from the recapture
    override and from the plain soft-threshold ensemble spoof paths. That
    hid the print-pattern override (see `_fuse` docstring) from Умид's
    monitoring — he needs to filter the false-reject rate of THIS specific
    signal separately, since it is new and calibrated on a thin (n=1)
    positive-class sample (see PRINT_PATTERN_FFT_MIN/PRINT_PATTERN_COLOR_MIN
    docstring in app/multisignal.py).

    `_fuse()` tags `signal_info["spoof_trigger"]` in place when the
    print-pattern or recapture override fires; every other spoof path
    (NN-only, soft ensemble thresholds) leaves it unset and keeps the
    original reason="PASSIVE_PAD_SPOOF".
    """
    if label != "spoof":
        return None
    if signal_info.get("spoof_trigger") == "print_pattern_override":
        return "PRINT_PATTERN_SPOOF"
    return "PASSIVE_PAD_SPOOF"


# Add Silent-Face repo src to path for model definitions
_REPO_SRC = Path(__file__).resolve().parent.parent / "src" / "model_lib"
if str(_REPO_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC.parent))

from model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE


MODEL_MAP = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


def _parse_model_name(name: str) -> tuple[int, int, str, Optional[float]]:
    """Extract h, w, model_type, scale from filename like '2.7_80x80_MiniFASNetV2.pth'."""
    info = name.split("_")[0:-1]
    h_str, w_str = info[-1].split("x")
    model_type = name.split(".pth")[0].split("_")[-1]
    scale = None if info[0] == "org" else float(info[0])
    return int(h_str), int(w_str), model_type, scale


def _get_kernel(height: int, width: int) -> tuple[int, int]:
    return ((height + 15) // 16, (width + 15) // 16)


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR numpy image to CHW float tensor (matches Silent-Face ToTensor)."""
    rgb = bgr[:, :, ::-1].copy()
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def _crop_face(
    image_bgr: np.ndarray,
    bbox: list[int],
    scale: Optional[float],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop face from image using bbox with padding (matches CropImage)."""
    if scale is None:
        return cv2.resize(image_bgr, (out_w, out_h))

    src_h, src_w = image_bgr.shape[:2]
    x, y, box_w, box_h = bbox

    s = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_w = box_w * s
    new_h = box_h * s
    cx, cy = box_w / 2 + x, box_h / 2 + y

    ltx = int(cx - new_w / 2)
    lty = int(cy - new_h / 2)
    rbx = int(cx + new_w / 2)
    rby = int(cy + new_h / 2)

    if ltx < 0:
        rbx -= ltx
        ltx = 0
    if lty < 0:
        rby -= lty
        lty = 0
    if rbx > src_w - 1:
        ltx -= rbx - src_w + 1
        rbx = src_w - 1
    if rby > src_h - 1:
        lty -= rby - src_h + 1
        rby = src_h - 1

    crop = image_bgr[lty : rby + 1, ltx : rbx + 1]
    return cv2.resize(crop, (out_w, out_h))


class LivenessEngine:
    """Loads both MiniFASNet models once, runs combined inference."""

    def __init__(
        self,
        model_dir: Path,
        device: str,
        print_pattern_override_enabled: bool = True,
    ) -> None:
        self._device = torch.device(device)
        self._models: list[tuple[torch.nn.Module, int, int, Optional[float]]] = []
        # 2026-07-22 print-pattern override (see _fuse docstring) — threaded
        # from app.config.Settings.PRINT_PATTERN_OVERRIDE_ENABLED so the fix
        # can be reverted via an env-var flip, not a code rollback.
        self._print_pattern_override_enabled = print_pattern_override_enabled
        self._load_models(model_dir)

    def _load_models(self, model_dir: Path) -> None:
        for fname in sorted(os.listdir(model_dir)):
            if not fname.endswith(".pth"):
                continue
            h, w, mtype, scale = _parse_model_name(fname)
            kernel = _get_kernel(h, w)
            model_cls = MODEL_MAP[mtype]
            model = model_cls(conv6_kernel=kernel).to(self._device)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                state = torch.load(model_dir / fname, map_location=self._device, weights_only=False)
            # Strip 'module.' prefix from DataParallel weights
            keys = list(state.keys())
            if keys and keys[0].startswith("module."):
                new_sd = OrderedDict()
                for k, v in state.items():
                    new_sd[k[7:]] = v
                state = new_sd

            model.load_state_dict(state)
            model.eval()
            self._models.append((model, h, w, scale))

    @torch.inference_mode()
    def predict(
        self, image_bgr: np.ndarray, bbox: Optional[list[int]] = None
    ) -> tuple[str, float, bool, dict]:
        """Run liveness prediction for a single image.

        Returns (label, score, face_detected, signal_info).
        label is "real", "spoof", or "no_face".
        signal_info contains multi-signal analysis breakdown.
        """
        if bbox is None:
            return "no_face", 0.0, False, {}

        # Neural network prediction
        prediction = np.zeros((1, 3))
        for model, h, w, scale in self._models:
            crop = _crop_face(image_bgr, bbox, scale, w, h)
            tensor = _to_tensor(crop).unsqueeze(0).to(self._device)
            out = model(tensor)
            prob = F.softmax(out, dim=1).cpu().numpy()
            prediction += prob

        nn_label = int(np.argmax(prediction))
        nn_score = float(prediction[0][nn_label]) / 2.0

        # Multi-signal analysis on the NATIVE-resolution face crop (224px).
        # face_px = native bbox area, feeds the resolution cue in `recapture`.
        face_px = max(0, bbox[2]) * max(0, bbox[3])
        face_crop = _crop_face(image_bgr, bbox, self._models[0][3], 224, 224)
        signal_info = analyze_face(face_crop, face_px)

        combined_label, combined_score = _fuse(
            nn_label, nn_score, signal_info, self._print_pattern_override_enabled
        )

        signal_info["nn_label"] = "real" if nn_label == 1 else "spoof"
        signal_info["nn_score"] = round(nn_score, 4)
        signal_info["combined_label"] = combined_label
        signal_info["combined_score"] = round(combined_score, 4)

        return combined_label, combined_score, True, signal_info

    @torch.inference_mode()
    def predict_batch(
        self,
        crops: list[np.ndarray],
        face_px_list: Optional[list[int]] = None,
    ) -> list[tuple[str, float, bool, dict]]:
        """Run liveness prediction on a batch of pre-cropped face images.

        `crops` must be native-resolution face crops (>=224px recommended) so the
        `recapture` signal stays valid. `face_px_list` carries the native bbox
        area per crop for the resolution cue.

        GPU batch for NN inference + per-image signal analysis.
        """
        if not crops:
            return []
        if face_px_list is None:
            face_px_list = [None] * len(crops)

        # GPU batch: stack all crops and forward through each model
        batch_preds = [np.zeros(3) for _ in crops]
        for model, h, w, scale in self._models:
            tensors = []
            for crop in crops:
                resized = cv2.resize(crop, (w, h))
                tensors.append(_to_tensor(resized))
            batch = torch.stack(tensors).to(self._device)
            out = model(batch)
            probs = F.softmax(out, dim=1).cpu().numpy()
            for i, prob in enumerate(probs):
                batch_preds[i] += prob

        # Combine NN output with signal analysis per image
        results: list[tuple[str, float, bool, dict]] = []
        for i, pred in enumerate(batch_preds):
            nn_label = int(np.argmax(pred))
            nn_score = float(pred[nn_label]) / 2.0

            # Signal analysis on the native crop (recapture needs full detail).
            signal_info = analyze_face(crops[i], face_px_list[i])

            label, score = _fuse(
                nn_label, nn_score, signal_info, self._print_pattern_override_enabled
            )

            signal_info["nn_label"] = "real" if nn_label == 1 else "spoof"
            signal_info["nn_score"] = round(nn_score, 4)
            signal_info["combined_label"] = label
            signal_info["combined_score"] = round(score, 4)
            results.append((label, score, True, signal_info))

        return results
