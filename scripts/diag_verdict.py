#!/usr/bin/env python3
import sys
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
sets = [
    ("REAL", sorted((CAL/"original").glob("*.jpg")), "real"),
    ("SPOOF", sorted((CAL/"urgut").glob("*.jpg")), "spoof"),
]
total_real=0; correct_real=0; total_spoof=0; correct_spoof=0
for label, files, expected in sets:
    for f in files:
        img = cv2.imread(str(f))
        bbox = detector.detect(img)
        if bbox is None:
            print(f"{label:<7}{f.name:<32} no_face"); continue
        combined_label, score, _, sig = engine.predict(img, bbox)
        ok = "OK" if combined_label == expected else "MISMATCH"
        if expected=="real":
            total_real+=1; correct_real += (combined_label=="real")
        else:
            total_spoof+=1; correct_spoof += (combined_label=="spoof")
        print(f"{label:<7}{f.name:<32}{combined_label:<7}{score:<7.3f}{ok}")
print(f"\nREAL correct: {correct_real}/{total_real}  SPOOF correct: {correct_spoof}/{total_spoof}")
