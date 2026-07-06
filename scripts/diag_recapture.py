#!/usr/bin/env python3
"""Diagnose recapture signal + full pipeline across calibration set + the FR photo."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from app.config import Settings, resolve_device
from app.face_detect import FaceDetector
from app.liveness import LivenessEngine
from app.multisignal import recapture_spoof_score, _ramp

settings = Settings()
device = resolve_device(settings.DEVICE)
detector = FaceDetector(settings.MODEL_DIR)
engine = LivenessEngine(settings.MODEL_DIR, device)

CAL = Path("/home/mrnurali/E-GAZ/docs/plans/calibration/incident_urgut")
sets = {
    "REAL(original)": sorted((CAL / "original").glob("*.jpg")),
    "SPOOF(urgut)": sorted((CAL / "urgut").glob("*.jpg")),
}

rows = []
for label, files in sets.items():
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            rows.append((label, f.name, "READ_ERROR", None, None, None, None))
            continue
        bbox = detector.detect(img)
        if bbox is None:
            rows.append((label, f.name, "no_face", None, None, None, None))
            continue
        combined_label, combined_score, _, sig = engine.predict(img, bbox)
        recap = sig["signal_scores"].get("recapture")
        face_px = max(0, bbox[2]) * max(0, bbox[3])
        rows.append((label, f.name, combined_label, sig["nn_label"], sig["nn_score"], recap, sig["spoof_probability"], face_px, sig["signal_scores"]))

print(f"{'SET':<16}{'file':<32}{'combined':<9}{'nn_lbl':<7}{'nn_sc':<7}{'recap':<8}{'spoof_p':<8}{'face_px':<9}")
for r in rows:
    if len(r) == 4:
        print(f"{r[0]:<16}{r[1]:<32}{r[2]:<9}")
        continue
    label, name, combined, nn_lbl, nn_sc, recap, spoof_p, face_px, sig = r
    print(f"{label:<16}{name:<32}{combined:<9}{nn_lbl:<7}{nn_sc:<7.3f}{recap:<8.3f}{spoof_p:<8.3f}{face_px:<9}")

print("\n--- recapture distribution ---")
import statistics
for label in ["REAL(original)", "SPOOF(urgut)"]:
    vals = [r[5] for r in rows if r[0]==label and len(r)>4 and r[5] is not None]
    if vals:
        print(f"{label}: n={len(vals)} min={min(vals):.3f} max={max(vals):.3f} mean={statistics.mean(vals):.3f}")

print("\n--- full signal breakdown for NEW_FR_PHOTO and worst-case REAL ---")
for r in rows:
    if r[0] == "NEW_FR_PHOTO" and len(r) > 4:
        print("NEW_FR_PHOTO signals:", json.dumps(r[8], indent=2))
