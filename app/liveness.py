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

from app.multisignal import analyze_face, SIGNAL_WEIGHTS, RECAPTURE_SPOOF_THRESHOLD

# Above this NN real-confidence, trust the trained CNN over the resolution-
# sensitive `recapture` cue. Rationale from testing: genuine photo/screen fakes
# never pushed the CNN past ~0.85 real-confidence, while a live (even low-res)
# webcam face scores ~0.99. Gating the recapture override on this stops
# false-rejecting real low-res captures without letting file-based fakes pass.
NN_TRUST_REAL = 0.95


def _fuse(nn_label: int, nn_score: float, signal_info: dict) -> tuple[str, float]:
    """Combine NN output with multi-signal analysis into a final verdict.

    A high `recapture` score is a physical, upscale-robust cue that the frame is
    a low-detail screen/print recapture; it overrides the NN "real" verdict
    (this is exactly the class of fake the NN misses) — UNLESS the CNN is very
    confident the face is real, which genuine fakes never achieve. Otherwise
    fall back to the NN, letting weaker signals tip borderline cases to spoof.
    """
    recap = signal_info["signal_scores"].get("recapture", 0.0)
    spoof_prob = signal_info["spoof_probability"]

    nn_very_confident_real = nn_label == 1 and nn_score >= NN_TRUST_REAL
    if recap >= RECAPTURE_SPOOF_THRESHOLD and not nn_very_confident_real:
        return "spoof", max(recap, spoof_prob)
    if nn_label != 1:
        return "spoof", max(nn_score, spoof_prob)
    if spoof_prob > 0.6:
        return "spoof", spoof_prob
    if spoof_prob > 0.35 and nn_score < 0.7:
        return "spoof", max(nn_score * 0.5, spoof_prob)
    if spoof_prob > 0.1 and nn_score < 0.6:
        return "spoof", max(nn_score * 0.4, spoof_prob)
    return "real", nn_score

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

    def __init__(self, model_dir: Path, device: str) -> None:
        self._device = torch.device(device)
        self._models: list[tuple[torch.nn.Module, int, int, Optional[float]]] = []
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

        combined_label, combined_score = _fuse(nn_label, nn_score, signal_info)

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

            label, score = _fuse(nn_label, nn_score, signal_info)

            signal_info["nn_label"] = "real" if nn_label == 1 else "spoof"
            signal_info["nn_score"] = round(nn_score, 4)
            signal_info["combined_label"] = label
            signal_info["combined_score"] = round(score, 4)
            results.append((label, score, True, signal_info))

        return results
