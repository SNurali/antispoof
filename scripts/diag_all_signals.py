#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from app.config import Settings, resolve_device
from app.face_detect import FaceDetector
from app.liveness import LivenessEngine

settings = Settings()
device = resolve_device(settings.DEVICE)
detector = FaceDetector(settings.MODEL_DIR)
engine = LivenessEngine(settings.MODEL_DIR, device)

CAL = Path("/home/mrnurali/E-GAZ/docs/plans/calibration/incident_urgut")
sets = {
    "REAL": sorted((CAL/"original").glob("*.jpg")),
    "SPOOF": sorted((CAL/"urgut").glob("*.jpg")),
    "NEWFR": [Path("/home/mrnurali/E-GAZ/docs/photo_2026-07-06_10-40-40.jpg")],
}
print(f"{'SET':<7}{'file':<32}{'recap':>7}{'fft':>6}{'lbp':>6}{'color':>7}{'moire':>7}{'sharp':>7}{'jpeg':>7}{'spoof_p':>8}")
for label, files in sets.items():
    for f in files:
        img = cv2.imread(str(f))
        bbox = detector.detect(img)
        if bbox is None:
            print(f"{label:<7}{f.name:<32} no_face"); continue
        _,_,_, sig = engine.predict(img, bbox)
        s = sig["signal_scores"]
        print(f"{label:<7}{f.name:<32}{s.get('recapture',0):>7.3f}{s.get('fft',0):>6.2f}{s.get('lbp',0):>6.2f}{s.get('color',0):>7.2f}{s.get('moire',0):>7.2f}{s.get('sharpness',0):>7.2f}{s.get('jpeg',0):>7.2f}{sig['spoof_probability']:>8.3f}")
