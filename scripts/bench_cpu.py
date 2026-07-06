#!/usr/bin/env python3
"""Measure current MiniFASNet ensemble latency on CPU (prod has no GPU)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from app.config import Settings
from app.face_detect import FaceDetector
from app.liveness import LivenessEngine

settings = Settings()
detector = FaceDetector(settings.MODEL_DIR)
engine = LivenessEngine(settings.MODEL_DIR, "cpu")  # force CPU regardless of DEVICE=auto

img = cv2.imread("/home/mrnurali/E-GAZ/docs/photo_2026-07-06_10-40-40.jpg")
bbox = detector.detect(img)

# warmup
for _ in range(3):
    engine.predict(img, bbox)

N = 20
t0 = time.perf_counter()
for _ in range(N):
    engine.predict(img, bbox)
elapsed = time.perf_counter() - t0
print(f"MiniFASNet ensemble (2 models) on CPU: {elapsed/N*1000:.1f} ms/frame avg over {N} runs")

import torch
print("torch threads:", torch.get_num_threads())
