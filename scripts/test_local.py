#!/usr/bin/env python3
"""Run liveness check on all images in a folder."""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2

# Allow importing from parent dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, resolve_device
from app.face_detect import FaceDetector
from app.liveness import LivenessEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch liveness test")
    parser.add_argument("folder", help="Folder with images to test")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    settings = Settings()
    device = resolve_device(settings.DEVICE)
    detector = FaceDetector(settings.MODEL_DIR)
    engine = LivenessEngine(settings.MODEL_DIR, device)

    folder = Path(args.folder)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)

    if not files:
        print(f"No images found in {folder}")
        return

    print(f"{'File':<35} {'Label':<10} {'Score':>6} {'ms':>7}")
    print("-" * 65)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name:<35} {'ERROR':<10} {'N/A':>6} {'N/A':>7}")
            continue

        t0 = time.perf_counter()
        bbox = detector.detect(img)
        if bbox is None:
            ms = (time.perf_counter() - t0) * 1000
            print(f"{f.name:<35} {'no_face':<10} {'0.00':>6} {ms:>6.1f}")
            continue

        label, score, _ = engine.predict(img, bbox)
        ms = (time.perf_counter() - t0) * 1000
        verdict = "REAL" if (label == "real" and score >= args.threshold) else "SPOOF"
        print(f"{f.name:<35} {verdict:<10} {score:>6.2f} {ms:>6.1f}")


if __name__ == "__main__":
    main()
