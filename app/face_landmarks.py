"""SCRFD detection + landmark_3d_68 pose — shared per-frame analysis for the
active-liveness pipeline (Layer 0 QC / Layer 2 active challenge / Layer 3
cross-frame identity all consume the SAME single detection pass per frame;
none of them re-detect).

Uses insightface's buffalo_l bundle but restricted to `allowed_modules=
["detection", "landmark_3d_68"]` — this loads ONLY the SCRFD detector
(det_10g.onnx) and the 3D pose/landmark model (1k3d68.onnx), skipping
landmark_2d_106, genderage, and the bundled w600k_r50 recognition model
(unused — Layer 3 embeds with AdaFace, not buffalo_l's own recognizer). This
is a deliberately different detector from the RetinaFace Caffe model
`app/face_detect.py` uses for the existing /verify, /pad/check, etc.
endpoints — NOT a replacement for it, those endpoints are untouched. Running
two different face detectors in the same service is a real, accepted cost
of reusing the existing passive-PAD gate (Layer 1, RetinaFace-based)
unmodified inside the new multi-frame pipeline alongside this SCRFD-based
Layer 0/2/3 pass, rather than rewriting Layer 1 to share a detector.

Measured (2026-07-17, i5-11400, 12 threads, CPUExecutionProvider, det_size
320x320, docs/plans/calibration/incident_urgut, single photos not session
frames): ~100-160ms/frame after model load (~1.9s one-time). See
app/config.py::LIVENESS_DET_SIZE.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameFace:
    """One frame's face-analysis result. `pose` is (pitch, yaw, roll) degrees,
    matching insightface's raw `face.pose` layout (index 1 = yaw) — same
    convention already used by the sibling face_id/tracker project
    (app/recognizer.py::face_pose)."""

    bbox_xyxy: tuple[float, float, float, float]
    kps: np.ndarray  # (5, 2) float32 — for face_align.norm_crop alignment
    pose_pitch: float
    pose_yaw: float
    pose_roll: float
    det_score: float
    n_faces_detected: int  # total faces found in this frame (>1 = reject upstream)

    # Raw (68, 3) points from the SAME landmark_3d_68 model already loaded
    # for pose (no extra inference — `best.landmark_3d_68` was already being
    # computed and discarded except for `.pose` before 2026-07-17). insightface
    # fits this to `meanshape_68.pkl`, i.e. the STANDARD dlib/iBUG-300W
    # 68-point layout, so indices 36-41 = right eye, 42-47 = left eye — the
    # same convention every dlib/EAR blink-detection tutorial uses. VERIFIED
    # (not just recalled) 2026-07-17: rendered all 68 indices on a real photo
    # (Загрузки/face_id/Flask/sherzod111.jpg) and confirmed 36-41/42-47 land
    # exactly on the eye contours before wiring `eye_aspect_ratios()` below to
    # them. `None` only if the model returned nothing for an otherwise-detected
    # face (should not happen in practice, kept Optional defensively).
    landmark_68: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# BLINK support (RZA, 2026-07-17) — EAR (eye aspect ratio) from landmark_68's
# eye-contour points, per Soukupová & Čech 2016. Reuses the already-loaded
# landmark_3d_68 model (see FrameFace.landmark_68 docstring for the index
# verification) — no new model, no new dependency, no landmark_2d_106 needed.
# See app/active_challenge.py for how this feeds Layer 2's BLINK step and
# app/config.py::LIVENESS_EAR_BLINK_MAX for why the CLOSED-eye threshold is
# still an unverified literature placeholder despite the index mapping being
# verified — index correctness and threshold calibration are separate
# questions, do not conflate "this measures the right thing" with "this
# threshold is right".
# ---------------------------------------------------------------------------
RIGHT_EYE_IDX: tuple[int, int, int, int, int, int] = (36, 37, 38, 39, 40, 41)
LEFT_EYE_IDX: tuple[int, int, int, int, int, int] = (42, 43, 44, 45, 46, 47)


def _single_eye_ratio(landmark_68: np.ndarray, idx: tuple[int, int, int, int, int, int]) -> float:
    """EAR for one eye = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||), p1/p4 the
    two corners, p2/p3 upper lid, p5/p6 lower lid — in the index order given
    by `idx` (RIGHT_EYE_IDX / LEFT_EYE_IDX)."""
    p1, p2, p3, p4, p5, p6 = (landmark_68[i][:2] for i in idx)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def eye_aspect_ratios(landmark_68: np.ndarray) -> tuple[float, float]:
    """(right_ear, left_ear) from a (68, 2-or-3) landmark_68 array. Sanity
    check on 3 real bonafide frontal selfies (2026-07-17, no closed-eye pairs
    available — see app/config.py::LIVENESS_EAR_BLINK_MAX): open-eye EAR from
    THIS model ranged 0.214-0.317, roughly matching the published open-eye
    range but with one sample uncomfortably close to a naive 0.20 cutoff —
    a real reason the threshold needs its own calibration pass, not proof
    the index mapping is wrong (the same 3 frames visually have clearly open
    eyes)."""
    return (
        _single_eye_ratio(landmark_68, RIGHT_EYE_IDX),
        _single_eye_ratio(landmark_68, LEFT_EYE_IDX),
    )


def min_eye_aspect_ratio(landmark_68: np.ndarray) -> float:
    """min(right_ear, left_ear) — a real blink closes both eyes together, but
    `min()` (rather than requiring BOTH below threshold) is used to keep this
    consistent with the same permissive "any evidence in the series" pattern
    app/active_challenge.py already uses for TURN_LEFT/TURN_RIGHT. A single
    noisy low reading on one eye while the other stays open could in
    principle false-trigger this — a known limitation, same class as the
    TURN steps' low-entropy caveat, not unique to BLINK."""
    right_ear, left_ear = eye_aspect_ratios(landmark_68)
    return min(right_ear, left_ear)


# ---------------------------------------------------------------------------
# SMILE support (RZA, 2026-07-20, CHALLENGE_ENTROPY_SPRINT_v1.md §4) — a
# MAR-like (mouth aspect ratio) coefficient from landmark_68's OUTER lip
# contour points, reusing the same already-loaded landmark_3d_68 model that
# already backs pose (TURN) and EAR (BLINK). No new model, no new
# dependency — same pattern as the BLINK section above.
#
# INDEX VERIFICATION (done, not assumed) — 2026-07-20, SAME METHOD already
# used for the eye indices above (see FrameFace.landmark_68 docstring):
# ran the real buffalo_l landmark_3d_68 model on the SAME real photo
# (Загрузки/face_id/Flask/sherzod111.jpg) already used for the eye-index
# check, rendered all 68 points with per-index labels, then rendered a
# zoomed crop of only indices 48-67. Visually confirmed against the
# standard dlib/iBUG-300W 68-point layout:
#   - 48 = left mouth corner, 54 = right mouth corner (outer contour)
#   - 49-53 = upper outer lip, left-to-right (51 = cupid's-bow apex)
#   - 55-59 = lower outer lip, right-to-left (57 = lower-lip apex)
#   - 60-67 = inner lip contour (not used by mouth_aspect_ratio() below)
# All 20 mouth points (48-67) land exactly on the lip contour in the
# rendered crop, and a geometric sanity check on the SAME photo confirms
# the mouth region sits below the eyes/nose and spans ~37% of the face
# bbox width — consistent with a real mouth, not a mislabeled region.
# What is NOT verified: a second/third face or a non-frontal pose — this
# is a single confirmation on a single frontal photo, same scope limitation
# already disclosed for the eye-index check.
#
# THRESHOLD CALIBRATION — separate question from index correctness, and
# WEAKER than even BLINK's placeholder: EAR's 0.20 cutoff at least traces
# to a cited literature source (Soukupová & Čech 2016) for a well-studied
# blink signal. There is no equivalent widely-cited "MAR value = smiling"
# constant for THIS width/height formulation — smile-detection thresholds
# in the wild vary by formula and dataset. The ONE real data point measured
# here is a NEUTRAL (non-smiling) frontal mouth on the same photo above:
# mouth_aspect_ratio() = 2.56 (width=142.4px corner-to-corner, height=55.6px
# averaged vertical gap). No real smiling photo was available in this
# environment to measure the other end of the range. See
# app/config.py::LIVENESS_MAR_SMILE_MIN for how this single baseline shapes
# (and limits) the placeholder value.
# ---------------------------------------------------------------------------
MOUTH_LEFT_CORNER_IDX: int = 48
MOUTH_RIGHT_CORNER_IDX: int = 54
MOUTH_TOP_IDX: tuple[int, int] = (50, 52)     # upper outer lip, left/right of apex
MOUTH_BOTTOM_IDX: tuple[int, int] = (58, 56)  # lower outer lip, mirrors MOUTH_TOP_IDX


def mouth_aspect_ratio(landmark_68: np.ndarray) -> float:
    """Width/height ratio of the mouth's OUTER contour (48-59) — "ширина/
    высота рта" per CHALLENGE_ENTROPY_SPRINT_v1.md §4.1's SMILE row.
    Structurally the same EAR formula (corner-to-corner horizontal distance
    vs. two averaged vertical gaps), just built from 6 mouth points (48,
    50, 52, 54, 56, 58) instead of 6 eye points, and with numerator/
    denominator SWAPPED relative to EAR (EAR = vertical/horizontal so an
    OPEN eye reads high; here width/height so a WIDENED, flattened smiling
    mouth is expected to read HIGH, not low — a relaxed/neutral mouth is
    already fairly wide-flat at rest, see the module docstring above for
    why that makes calibration harder than BLINK's, not why the formula is
    wrong).

    width  = ||p48 - p54||           (corner-to-corner)
    height = mean(||p50-p58||, ||p52-p56||)   (two vertical gaps, mirrored
             left/right of the cupid's-bow apex, avoiding a single noisy
             midline measurement)

    Returns 0.0 for a degenerate (near-zero-height) mouth rather than
    dividing by ~0, same defensive pattern as _single_eye_ratio above."""
    p48, p50, p52, p54, p56, p58 = (
        landmark_68[i][:2] for i in (
            MOUTH_LEFT_CORNER_IDX, MOUTH_TOP_IDX[0], MOUTH_TOP_IDX[1],
            MOUTH_RIGHT_CORNER_IDX, MOUTH_BOTTOM_IDX[1], MOUTH_BOTTOM_IDX[0],
        )
    )
    width = np.linalg.norm(p48 - p54)
    height = (np.linalg.norm(p50 - p58) + np.linalg.norm(p52 - p56)) / 2.0
    if height < 1e-6:
        return 0.0
    return float(width / height)


class LandmarkDetector:
    """Loads buffalo_l detection+landmark_3d_68 once; analyzes frames on demand."""

    def __init__(self, det_size: int = 320, device: str = "cpu") -> None:
        """`device` is a RESOLVED device string ("cpu"/"cuda", see
        app/config.py::resolve_device) — same DEVICE knob app/liveness.py's
        LivenessEngine and app/adaface.py::AdaFaceEmbedder already take
        (2026-07-17 GPU dual-mode work). Measured on this repo's dev RTX
        3080: SCRFD det_10g + landmark_3d_68 combined ~6ms/frame warm on
        CUDAExecutionProvider vs ~34ms/frame CPUExecutionProvider on this
        same 12-thread dev box (see module docstring for the ~100-160ms/frame
        figure on the i5-11400 prod-like box). See
        scripts/bench_identity_layer.py for the reproducible benchmark.

        insightface's own `FaceAnalysis.prepare(ctx_id, ...)` FORCES
        CPUExecutionProvider internally whenever ctx_id<0 (see
        insightface.model_zoo.scrfd.SCRFD.prepare /
        insightface.model_zoo.landmark.Landmark.prepare — both call
        `self.session.set_providers(['CPUExecutionProvider'])` when
        ctx_id<0, silently overriding whatever `providers=` was passed at
        construction). So `ctx_id` and `providers` must agree here — ctx_id
        is derived FROM the resolved provider list, not hardcoded.

        Falls back to CPU-only in two independent ways, same pattern as
        AdaFaceEmbedder: (1) app.config.onnx_providers() only returns a CUDA
        provider if onnxruntime itself reports "CUDAExecutionProvider" in
        get_available_providers(); (2) session/model construction below is
        wrapped in try/except — a CUDA/cuDNN runtime mismatch also falls
        back to CPU-only rather than crashing.
        """
        # Imported lazily inside __init__, not at module top, so that a
        # deploy with LIVENESS_ENDPOINTS_ENABLED=False never needs
        # insightface installed at all (app/main.py only constructs this
        # class when the flag is on).
        from insightface.app import FaceAnalysis

        from app.config import onnx_providers

        providers = onnx_providers(device)
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1

        try:
            self._app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "landmark_3d_68"],
                providers=providers,
            )
            self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        except Exception:
            if providers != ["CPUExecutionProvider"]:
                logger.exception(
                    "SCRFD/landmark_3d_68 failed to init with providers=%s "
                    "(requested device=%s, ctx_id=%d) — falling back to "
                    "CPUExecutionProvider only. Common cause: onnxruntime-gpu "
                    "is installed but the CUDA/cuDNN runtime it needs is "
                    "missing or a version mismatch (see requirements.txt GPU "
                    "section).",
                    providers, device, ctx_id,
                )
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
                self._app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "landmark_3d_68"],
                    providers=providers,
                )
                self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
            else:
                raise

        logger.info(
            "LandmarkDetector ready (buffalo_l detection+landmark_3d_68, det_size=%d, "
            "requested device=%s, providers=%s, ctx_id=%d)",
            det_size, device, providers, ctx_id,
        )

    def analyze(self, image_bgr: np.ndarray) -> Optional[FrameFace]:
        """Returns the highest-det_score face's analysis, or None if no face
        found. Caller must check `n_faces_detected` to reject multi-face
        frames (potential coaching/substitution attempt) — this method does
        NOT reject on its own, it only reports the count."""
        faces = self._app.get(image_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        pose = getattr(best, "pose", None)
        pitch, yaw, roll = (0.0, 0.0, 0.0) if pose is None else (
            float(pose[0]), float(pose[1]), float(pose[2])
        )
        raw_lmk68 = getattr(best, "landmark_3d_68", None)
        landmark_68 = None if raw_lmk68 is None else np.asarray(raw_lmk68, dtype=np.float32)
        return FrameFace(
            bbox_xyxy=tuple(float(v) for v in best.bbox),
            kps=np.asarray(best.kps, dtype=np.float32),
            pose_pitch=pitch,
            pose_yaw=yaw,
            pose_roll=roll,
            det_score=float(best.det_score),
            n_faces_detected=len(faces),
            landmark_68=landmark_68,
        )

    @staticmethod
    def align_112(image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Landmark-aligned 112x112 crop for AdaFace, same alignment template
        insightface/AdaFace expect (arcface 5-point norm_crop)."""
        from insightface.utils import face_align

        return face_align.norm_crop(image_bgr, landmark=kps, image_size=112)
