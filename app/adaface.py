"""AdaFace IR-101 (WebFace12M) embedding, ONNX-only — Layer 3 cross-frame identity.

Reused from the sibling face_id/tracker project (app/adaface.py there), trimmed
to the ONNX-inference path only: that project already exports the same
checkpoint to ONNX for a "CPU turbo" mode, and this service only ever needs
inference, never training or the torch model definition
(app/adaface_net.py in tracker is NOT copied here for that reason).

Weight provenance: models/liveness/adaface_ir101_webface12m.onnx is a copy of
face_id/tracker/weights/adaface_ir101_webface12m.onnx (2026-07-17). Same
model family already proven in that project for AdaFace-vs-AdaFace cosine
matching; NOT yet re-validated end-to-end for THIS service's specific
same-session-consistency use case (see config.py IDENTITY_MIN docstring for
why no real calibration exists yet).

IMPORTANT — this is IR-101 (~65M params), NOT the IR-18/IR-50 that
docs/plans/FACEID_LIVENESS_ML_CORE_v1.md §2.3/§7 recommends for this exact
role on CPU latency grounds. See app/config.py::ADAFACE_ONNX_PATH for the
measured latency numbers that confirm that concern. Kept as IR-101 in this
increment only because no lighter checkpoint exists anywhere in either
project yet — swapping the weight file (env var) is the intended fix path,
not a code change.
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AdaFaceEmbedder:
    """Loads the AdaFace IR-101 ONNX graph once, embeds aligned 112x112 crops.

    Preprocessing MUST match training: BGR, (x/255 - 0.5)/0.5, CHW, batch of 1.
    Output is L2-normalized so cosine similarity == dot product.
    """

    INPUT_SIZE = 112

    def __init__(
        self, onnx_path: Path, device: str = "cpu", intra_op_num_threads: int = 0
    ) -> None:
        """`device` is a RESOLVED device string ("cpu"/"cuda", see
        app/config.py::resolve_device) — same DEVICE knob app/liveness.py's
        LivenessEngine already takes, now wired through here too (2026-07-17
        GPU dual-mode work). Measured on this repo's dev RTX 3080
        (onnxruntime-gpu 1.20.1, CUDA 12.1/cuDNN 9.1 via the same pip
        nvidia-*-cu12 wheels torch already depends on): ~4ms/frame warm on
        CUDAExecutionProvider vs ~250ms/frame CPUExecutionProvider on this
        same 12-thread dev box, and 342-524ms/frame on the i5-11400 prod-like
        box this file's module docstring cites for CPU. See
        scripts/bench_identity_layer.py for the reproducible benchmark.

        Falls back to CPU-only in two independent ways, neither of which
        requires onnxruntime-gpu to be installed or raises if it is not:
        (1) app.config.onnx_providers() only returns a CUDA provider if
        onnxruntime itself reports "CUDAExecutionProvider" in
        get_available_providers() (i.e. onnxruntime-gpu is actually the
        package installed); (2) even if that check passes, session creation
        below is wrapped in try/except — a CUDA/cuDNN runtime version
        mismatch on the box (not knowable without actually trying to init)
        also falls back to CPU-only rather than crashing.
        """
        # Imported lazily so a deploy with LIVENESS_ENDPOINTS_ENABLED=False
        # (app/config.py) never needs onnxruntime installed at all — this
        # module is otherwise imported unconditionally by app/main.py via
        # app/identity_consistency.py.
        import onnxruntime as ort

        from app.config import onnx_providers

        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"AdaFace ONNX weights not found at {onnx_path} — "
                "copy models/liveness/adaface_ir101_webface12m.onnx before "
                "enabling LIVENESS_ENDPOINTS_ENABLED."
            )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            so.intra_op_num_threads = intra_op_num_threads

        providers = onnx_providers(device)
        try:
            self._session = ort.InferenceSession(str(onnx_path), so, providers=providers)
        except Exception:
            if providers != ["CPUExecutionProvider"]:
                logger.exception(
                    "AdaFace ONNX session failed to init with providers=%s "
                    "(requested device=%s) — falling back to CPUExecutionProvider "
                    "only. Common cause: onnxruntime-gpu is installed but the "
                    "CUDA/cuDNN runtime it needs is missing or a version "
                    "mismatch (see requirements.txt GPU section).",
                    providers, device,
                )
                self._session = ort.InferenceSession(
                    str(onnx_path), so, providers=["CPUExecutionProvider"]
                )
            else:
                raise

        self._input_name = self._session.get_inputs()[0].name
        active_providers = self._session.get_providers()
        logger.info(
            "AdaFace IR-101 ONNX loaded from %s (requested device=%s, active providers=%s)",
            onnx_path, device, active_providers,
        )

    def embed_aligned(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        """aligned_bgr_112: 112x112x3 BGR crop, ALREADY landmark-aligned
        (insightface.utils.face_align.norm_crop). Returns L2-normalized
        512-d float32 embedding."""
        x = ((aligned_bgr_112.astype(np.float32) / 255.0) - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]  # (1, 3, 112, 112)
        v = self._session.run(None, {self._input_name: x})[0][0].astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return v
        return v / norm

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Both inputs already L2-normalized -> dot product == cosine similarity."""
        return float(np.dot(a, b))
