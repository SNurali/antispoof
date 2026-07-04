"""Face detection using RetinaFace Caffe model (bundled)."""

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FaceDetector:
    """RetinaFace-based face detector."""

    # Internal detection input size. 192 (Silent-Face default) is too small for
    # phone photos held at arm's length: a genuine face can drop below the 0.6
    # confidence gate and get rejected as no_face. 320 recovers those without
    # meaningfully raising cost, and never changes returned bbox coordinates
    # (they are scaled from the original width/height).
    DET_SIZE = 320

    def __init__(self, model_dir: Path, confidence: float = 0.6) -> None:
        deploy = model_dir / "detection_model" / "deploy.prototxt"
        weights = model_dir / "detection_model" / "Widerface-RetinaFace.caffemodel"
        self._net = cv2.dnn.readNetFromCaffe(str(deploy), str(weights))
        self._confidence = confidence

    def detect(self, image_bgr: np.ndarray) -> Optional[list[int]]:
        """Return face bbox as [x, y, w, h] or None if no face found."""
        h, w = image_bgr.shape[:2]
        aspect = w / h
        scale_img = image_bgr.copy()
        if w * h >= self.DET_SIZE * self.DET_SIZE:
            scale_img = cv2.resize(
                scale_img,
                (int(self.DET_SIZE * math.sqrt(aspect)), int(self.DET_SIZE / math.sqrt(aspect))),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(scale_img, 1, mean=(104, 117, 123))
        self._net.setInput(blob, "data")
        out = self._net.forward("detection_out").squeeze()

        max_idx = np.argmax(out[:, 2])
        if out[max_idx, 2] < self._confidence:
            return None

        left = int(out[max_idx, 3] * w)
        top = int(out[max_idx, 4] * h)
        right = int(out[max_idx, 5] * w)
        bottom = int(out[max_idx, 6] * h)
        return [left, top, right - left + 1, bottom - top + 1]
