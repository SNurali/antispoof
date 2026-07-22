"""Multi-signal spoof detection: frequency + texture + color + moire + sharpness.

These are heuristic signals that catch what neural networks miss:
- Screens produce periodic frequency patterns (FFT peaks)
- Printed photos have uniform texture (low LBP variance)
- Screens have narrow color gamut in YCbCr
- Moire patterns from screen pixel grids
- Real faces have natural depth-of-field variation (non-uniform sharpness)
"""

import cv2
import numpy as np
from typing import Optional


def _ensure_gray(face_bgr: np.ndarray) -> np.ndarray:
    if len(face_bgr.shape) == 3:
        return cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return face_bgr


# ---------------------------------------------------------------------------
# 1. FFT — frequency domain analysis
# ---------------------------------------------------------------------------
def fft_spoof_score(face_bgr: np.ndarray) -> float:
    """Detect periodic patterns from screens/prints via FFT.

    Real faces have smooth, broadband spectra.
    Screens/prints show peaks at specific frequencies (pixel grid, halftone dots).
    Returns spoof probability 0..1.
    """
    gray = _ensure_gray(face_bgr).astype(np.float32)
    h, w = gray.shape

    # Apply Hanning window to reduce edge artifacts
    win_h = np.hanning(h).reshape(-1, 1)
    win_w = np.hanning(w).reshape(1, -1)
    windowed = gray * win_h * win_w

    fft = np.fft.fft2(windowed)
    magnitude = np.abs(np.fft.fftshift(fft))

    # Mask out DC component (center)
    cy, cx = h // 2, w // 2
    mask = np.ones_like(magnitude, dtype=bool)
    r = max(3, min(h, w) // 10)
    y_lo, y_hi = max(0, cy - r), min(h, cy + r)
    x_lo, x_hi = max(0, cx - r), min(w, cx + r)
    mask[y_lo:y_hi, x_lo:x_hi] = False

    # Energy in high-frequency band (where screen patterns live)
    mag_masked = magnitude.copy()
    mag_masked[~mask] = 0
    total_energy = magnitude[mask].sum()
    if total_energy < 1e-10:
        return 0.0

    hf_energy = mag_masked[mask].sum()
    hf_ratio = hf_energy / total_energy

    # Find peaks: ratio of max to mean in annular band (vectorized)
    band_inner = max(5, min(h, w) // 6)
    band_outer = min(h, w) // 2 - 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    annular_mask = (dist >= band_inner) & (dist <= band_outer)

    annular_vals = magnitude[annular_mask]
    if len(annular_vals) == 0:
        return 0.0

    peak_ratio = annular_vals.max() / (annular_vals.mean() + 1e-10)

    # Score: high peak_ratio + high hf_ratio = likely spoof
    # Thresholds calibrated to avoid false positives on real faces
    score = 0.0
    if peak_ratio > 15.0:
        score += 0.5
    elif peak_ratio > 10.0:
        score += 0.3

    if hf_ratio > 0.4:
        score += 0.3
    elif hf_ratio > 0.3:
        score += 0.15

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# 2. LBP — texture analysis
# ---------------------------------------------------------------------------
def lbp_spoof_score(face_bgr: np.ndarray) -> float:
    """Local Binary Pattern texture analysis.

    Real skin has rich, varied micro-texture.
    Printed photos / screen photos have flatter, more uniform texture.
    Returns spoof probability 0..1.
    """
    gray = _ensure_gray(face_bgr)
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0

    # Compute LBP (vectorized)
    center = gray[1:-1, 1:-1]
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    lbp |= (gray[0:-2, 0:-2] >= center).astype(np.uint8) << 7
    lbp |= (gray[0:-2, 1:-1] >= center).astype(np.uint8) << 6
    lbp |= (gray[0:-2, 2:] >= center).astype(np.uint8) << 5
    lbp |= (gray[1:-1, 2:] >= center).astype(np.uint8) << 4
    lbp |= (gray[2:, 2:] >= center).astype(np.uint8) << 3
    lbp |= (gray[2:, 1:-1] >= center).astype(np.uint8) << 2
    lbp |= (gray[2:, 0:-2] >= center).astype(np.uint8) << 1
    lbp |= (gray[1:-1, 0:-2] >= center).astype(np.uint8) << 0

    # Histogram of LBP codes (uniform patterns)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-10)

    # Uniformity: real faces have more diverse patterns (higher entropy)
    nonzero = hist[hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    # Normalized entropy (max = log2(256) = 8)
    norm_entropy = entropy / 8.0

    # Low entropy = uniform texture = likely spoof
    score = 0.0
    if norm_entropy < 0.6:
        score = 0.6
    elif norm_entropy < 0.7:
        score = 0.3
    elif norm_entropy < 0.75:
        score = 0.15

    return score


# ---------------------------------------------------------------------------
# 3. Color space — YCbCr analysis
# ---------------------------------------------------------------------------
def color_spoof_score(face_bgr: np.ndarray) -> float:
    """Analyze color distribution in YCbCr space.

    Screens have limited color gamut and shifted chrominance.
    Printed photos have desaturated, narrow Cb/Cr distributions.
    Real faces have wider, more natural chrominance spread.
    Returns spoof probability 0..1.
    """
    ycbcr = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)

    # Standard deviation of chrominance channels
    cr_std = float(np.std(cr))
    cb_std = float(np.std(cb))

    # Real faces typically have cr_std > 8, cb_std > 6
    # Spoof faces are more uniform — require BOTH channels narrow to flag
    score = 0.0
    if cr_std < 4 and cb_std < 3:
        score = 0.6
    elif cr_std < 6 and cb_std < 5:
        score = 0.35
    elif cr_std < 8 and cb_std < 6:
        score = 0.15

    return score


# ---------------------------------------------------------------------------
# 4. Moire pattern detection
# ---------------------------------------------------------------------------
def moire_spoof_score(face_bgr: np.ndarray) -> float:
    """Detect moire/interference patterns from screen pixel grids.

    Uses high-pass filter + threshold to find periodic artifacts.
    Returns spoof probability 0..1.
    """
    gray = _ensure_gray(face_bgr).astype(np.float32)
    h, w = gray.shape

    # High-pass filter: subtract Gaussian blur
    blurred = cv2.GaussianBlur(gray, (11, 11), 3.0)
    hp = gray - blurred

    # Threshold to isolate strong periodic patterns
    thresh_val = np.percentile(np.abs(hp), 95)
    if thresh_val < 1.0:
        return 0.0

    binary = (np.abs(hp) > thresh_val).astype(np.uint8)

    # Measure density of high-frequency artifacts
    density = binary.sum() / (h * w)

    # Count connected components (moire creates many small blobs)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0

    # Filter small components (noise) vs medium (moire pattern)
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    moire_components = np.sum((areas > 5) & (areas < (h * w) // 20))

    score = 0.0
    if density > 0.15 and moire_components > 10:
        score = 0.5
    elif density > 0.10 and moire_components > 5:
        score = 0.3
    elif density > 0.05 and moire_components > 3:
        score = 0.15

    return score


# ---------------------------------------------------------------------------
# 5. Sharpness / depth-of-field analysis
# ---------------------------------------------------------------------------
def sharpness_spoof_score(face_bgr: np.ndarray) -> float:
    """Analyze sharpness variation across the face.

    Real faces have natural DOF: ears/hair slightly blurred, eyes sharp.
    Photos-of-photos tend to be uniformly sharp or uniformly blurry.
    Returns spoof probability 0..1.
    """
    gray = _ensure_gray(face_bgr)
    h, w = gray.shape

    # Laplacian variance as sharpness measure
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(laplacian.var())

    # Divide face into grid and measure local sharpness variation
    grid = 4
    cell_h, cell_w = h // grid, w // grid
    local_sharpness = []
    for gy in range(grid):
        for gx in range(grid):
            cell = gray[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            local_var = float(cv2.Laplacian(cell, cv2.CV_32F).var())
            local_sharpness.append(local_var)

    local_sharpness = np.array(local_sharpness)
    sharpness_cv = local_sharpness.std() / (local_sharpness.mean() + 1e-10)

    # Real faces: moderate sharpness_cv (some DOF variation)
    # Spoof (photo-of-photo): very low CV (uniform) or very high (artifacts)
    score = 0.0
    if sharpness_cv < 0.1:
        score = 0.4  # suspiciously uniform
    elif sharpness_cv < 0.2:
        score = 0.2
    elif sharpness_cv > 1.5:
        score = 0.25  # suspiciously noisy

    return score


# ---------------------------------------------------------------------------
# 6. JPEG compression artifact analysis
# ---------------------------------------------------------------------------
def jpeg_artifact_score(face_bgr: np.ndarray) -> float:
    """Detect double-compression artifacts.

    A photo of a screen/photo undergoes multiple JPEG compression cycles,
    creating characteristic block-boundary artifacts.
    Returns spoof probability 0..1.
    """
    gray = _ensure_gray(face_bgr).astype(np.float32)
    h, w = gray.shape

    # Detect 8x8 block boundaries (JPEG artifact signature)
    # Sum absolute differences at block boundaries vs interior
    block_diffs = []
    interior_diffs = []

    for y in range(8, h - 8, 8):
        # Boundary: difference between row y-1 and row y
        block_diffs.append(float(np.mean(np.abs(gray[y, :] - gray[y - 1, :]))))
        # Interior: difference between row y+3 and row y+4
        if y + 4 < h:
            interior_diffs.append(float(np.mean(np.abs(gray[y + 3, :] - gray[y + 4, :]))))

    for x in range(8, w - 8, 8):
        block_diffs.append(float(np.mean(np.abs(gray[:, x] - gray[:, x - 1]))))
        if x + 4 < w:
            interior_diffs.append(float(np.mean(np.abs(gray[:, x + 3] - gray[:, x + 4]))))

    if not block_diffs or not interior_diffs:
        return 0.0

    bd_mean = np.mean(block_diffs)
    id_mean = np.mean(interior_diffs)

    if id_mean < 1e-10:
        return 0.0

    ratio = bd_mean / id_mean

    # Real images: ratio ~1.0 (no block artifacts)
    # Re-compressed spoofs: ratio > 1.2 (block boundaries more pronounced)
    score = 0.0
    if ratio > 1.5:
        score = 0.4
    elif ratio > 1.3:
        score = 0.25
    elif ratio > 1.15:
        score = 0.1

    return score


# ---------------------------------------------------------------------------
# 7. Recapture / detail-loss analysis (THE dominant signal)
# ---------------------------------------------------------------------------
# Rationale: the fakes that slip past the NN are low-resolution recaptures —
# a phone photographing a screen/print, or a compressed re-forward. They lose
# high-frequency facial detail and camera sensor noise. Crucially this MUST be
# measured on the NATIVE-resolution face crop: if you first downscale to
# 160x160 (as the legacy signals do) the resolution gap vanishes and every
# signal goes blind. Measuring at a fixed 224x224 keeps this upscale-robust —
# an attacker who upscales a blurry recapture cannot restore detail that was
# never captured.

RECAPTURE_SPOOF_THRESHOLD = 0.5  # >= this => treat as physical recapture spoof

# ---------------------------------------------------------------------------
# Print-pattern override thresholds (RZA, 2026-07-22 incident — printed
# passport-page photo scored nn_score=0.5671/combined_label=real on
# /pad/check; see app/liveness.py::_fuse for the override this feeds and the
# full root-cause writeup).
#
# WHY `recapture` MISSED THIS ATTACK: the passport photo was a SHARP,
# high-resolution close-up shot of a physical paper document — it is NOT a
# low-detail screen/print recapture (recapture's own design target). Measured
# on the real incident file: recapture=0.003 (near-zero, i.e. "high detail,
# real-like") because the paper's own print-halftone dots, MRZ text and
# fibre texture ARE high-frequency detail — recapture cannot tell "detailed
# because it's real skin" from "detailed because it's a sharp photo of a
# highly-textured printed page". This is a genuine BLIND SPOT of that
# signal for this specific attack sub-class, not a threshold-tuning bug.
#
# WHAT DID fire correctly on the same frame: `fft`=0.6 (the print's
# halftone-dot grid is exactly the periodic high-frequency pattern this
# signal targets) and `color`=0.6 (a sepia/monochrome document scan has
# near-zero chrominance spread — cr_std<4 and cb_std<3, the tightest color()
# bucket). Both are RIGHT, but each is individually under-weighted in the
# combined ensemble (fft=0.05, color=0.10 of SIGNAL_WEIGHTS) so together they
# only contributed spoof_probability=0.0914 to the weighted sum — just under
# every soft threshold in `_fuse()` (nearest is `> 0.1`).
#
# CALIBRATION (RZA, 2026-07-22), same corpus discipline as RECAPTURE_SPOOF_
# THRESHOLD above — full pipeline (real FaceDetector + LivenessEngine,
# phash-deduplicated) run on:
#   - BONAFIDE_urgut_orig (n=12, confirmed real):      0/12  hits
#   - BONAFIDE_faces_real (n=42 after phash-dedup of
#     faces-dataset/real, unverified Telegram-scrape
#     provenance, used ONLY as an FRR check, not FAR):  0/42  hits
#   - UNVERIFIED_faces_fake (n=25 after dedup of
#     faces-dataset/fake, provenance/ground-truth
#     unverified — see app/resolution_check.py's own
#     caveat on this folder — informational only):      0/25  hits
#   - SPOOF_urgut_recapture (n=11, confirmed spoof):    0/11  hits (already
#     caught via the recapture override instead — no overlap needed)
#   - SPOOF_passport_tightcrop (n=1, urgut_v2_passport/
#     passport_style_spoof_01.jpg, confirmed spoof):     0/1   hit (already
#     caught via the recapture override — fft=0.6 but color=0.0 there, a
#     DIFFERENT sub-signature than this incident)
#   - SPOOF_passport_fullpage (n=1, THIS incident,
#     confirmed spoof):                                  1/1   hit
#
# `color >= 0.5` ALONE (without the `fft` co-condition) has ONE false
# positive in the 42-file faces-dataset/real corpus (fft=0.3, color=0.6,
# recapture=0.6735 — a plausible desaturated/grayscale-filtered photo in
# that unverified scrape) — this is exactly why the rule below requires
# BOTH `fft` AND `color` to independently clear their thresholds, not
# `color` alone. With the AND-composite, the false positive above does not
# fire (its fft=0.3 < PRINT_PATTERN_FFT_MIN) and the overall margin across
# all 79 bonafide+unverified samples (12 confirmed bonafide + 42 unverified
# real + 25 unverified fake — see the CALIBRATION list above; NOT 91, a prior
# version of this docstring miscounted the sum) is clean (0 hits).
#
# HONESTY CAVEAT: this is n=1 for the specific attack sub-class it targets
# (full-page passport photo) and n=2 counting the differently-signed
# tightcrop spoof it does NOT rely on — a real but thin positive-class
# sample, the same "not a statistically tight bound" caveat every other
# Layer 0/1 gate in this service already carries (see geometry_check.py).
# The negative-class margin (79 bonafide/unverified, 0 hits) is the
# strongest part of this calibration and the primary reason this override
# defaults ON in app/config.py::PRINT_PATTERN_OVERRIDE_ENABLED — see that
# flag's docstring for the rollback path AND for the raw corpus's actual
# on-disk location (/home/mrnurali/E-GAZ/faces-dataset/{real,fake}/ — it IS
# retained, just not at the /home/mrnurali/faces-dataset path 2PAC checked).
PRINT_PATTERN_FFT_MIN = 0.5
PRINT_PATTERN_COLOR_MIN = 0.5


def _ramp(x: float, lo: float, hi: float) -> float:
    """1.0 when x<=lo (spoof-like), 0.0 when x>=hi (real-like), linear between."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return (hi - x) / (hi - lo)


def recapture_spoof_score(face_native_bgr: np.ndarray, face_px: Optional[int]) -> float:
    """Detect low-detail screen/print recaptures on the NATIVE face crop.

    Combines four correlated detail cues (sharpness, gradient energy, sensor
    noise, HF spectral ratio) plus a lightly-weighted absolute face-resolution
    cue. Returns spoof probability 0..1. Thresholds calibrated on real vs
    recaptured captures with a clear margin (real <=0.24, recapture >=0.59).
    """
    g = cv2.cvtColor(cv2.resize(face_native_bgr, (224, 224)), cv2.COLOR_BGR2GRAY)
    gf = g.astype(np.float32)

    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())          # absolute sharpness
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    teng = float(np.sqrt(gx * gx + gy * gy).mean())          # gradient energy
    resid = float((gf - cv2.GaussianBlur(gf, (3, 3), 0)).std())  # sensor micro-noise

    spec = np.fft.fftshift(np.abs(np.fft.fft2(gf)))
    yy, xx = np.mgrid[0:224, 0:224]
    dist = np.sqrt((yy - 112) ** 2 + (xx - 112) ** 2)
    hf = float(spec[dist > 60].sum() / (spec.sum() + 1e-9))  # high-freq ratio

    s_lap = _ramp(lap, 800.0, 2600.0)
    s_teng = _ramp(teng, 25.0, 68.0)
    s_resid = _ramp(resid, 3.0, 10.0)
    s_hf = _ramp(hf, 0.15, 0.42)

    detail = 0.45 * s_lap + 0.20 * s_teng + 0.20 * s_resid + 0.15 * s_hf

    if face_px is not None:
        s_res = _ramp(float(face_px), 20000.0, 45000.0)
        return float(min(1.0, 0.80 * detail + 0.20 * s_res))
    return float(min(1.0, detail))


# ---------------------------------------------------------------------------
# Ensemble: combine all signals
# ---------------------------------------------------------------------------

# Weights for each signal. `recapture` dominates because it is the only signal
# computed on native resolution and it is what actually separates the classes.
SIGNAL_WEIGHTS = {
    "recapture": 0.45,
    "fft": 0.05,
    "lbp": 0.15,
    "color": 0.10,
    "moire": 0.10,
    "sharpness": 0.05,
    "jpeg": 0.10,
}


def analyze_face(face_bgr: np.ndarray, face_px: Optional[int] = None) -> dict:
    """Run all spoof signals on a face crop.

    `face_bgr` should be the highest-resolution face crop available (>=160px).
    When `face_px` (native bbox area in the source image) is provided, the
    dominant `recapture` signal is computed on `face_bgr` at native resolution;
    the legacy texture/color/frequency signals run on a 160x160 view.

    Returns dict with individual scores and combined spoof probability.
    """
    face160 = face_bgr
    if face_bgr.shape[0] != 160 or face_bgr.shape[1] != 160:
        face160 = cv2.resize(face_bgr, (160, 160))

    scores = {
        "fft": fft_spoof_score(face160),
        "lbp": lbp_spoof_score(face160),
        "color": color_spoof_score(face160),
        "moire": moire_spoof_score(face160),
        "sharpness": sharpness_spoof_score(face160),
        "jpeg": jpeg_artifact_score(face160),
    }
    if face_px is not None:
        scores["recapture"] = recapture_spoof_score(face_bgr, face_px)

    # Normalized weighted sum over the signals actually computed
    active = {k: SIGNAL_WEIGHTS[k] for k in scores}
    wsum = sum(active.values()) or 1.0
    combined = sum(scores[k] * active[k] for k in scores) / wsum

    return {
        "signal_scores": {k: round(v, 4) for k, v in scores.items()},
        "spoof_probability": round(combined, 4),
        "signals_triggered": [k for k, v in scores.items() if v > 0.1],
    }
