"""Layer 3 — cross-frame identity consistency (PRIORITY #1 of this increment).

Per docs/plans/FACEID_LIVENESS_ML_CORE_v1.md §2.3: this is NOT identity
verification against Adliya (that stays external, unchanged, downstream of
this service) — its ONLY job is to confirm that every key frame in a single
liveness session shows the SAME physical person. This is the concrete
defense against the "moved for the blink step, swapped in an accomplice's
photo for the frame that goes to Adliya" splice attack that a purely
per-frame passive-PAD + per-frame active-challenge check cannot catch on
their own (each frame can individually look fine).

Reference frame selection: the FIRST frame that passed Layer 0 QC
(app/frame_qc.py) is used as the identity reference, all other valid frames
are compared against it. Deterministic and re-derivable from the SAME
`frames` list the caller already has (per session_id) — no re-sampling, no
separate fetch, satisfying the R2 requirement that the frames judged for
liveness are the exact frames that flow onward (see
app/main.py::_run_liveness_verdict for how `best_frame_seq` is chosen from
this same set).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.adaface import AdaFaceEmbedder


@dataclass(frozen=True)
class IdentityConsistencyResult:
    passed: bool
    min_similarity: float  # -1.0 if fewer than 2 embeddings (nothing to compare)
    reference_seq: int
    pairwise: dict = field(default_factory=dict)  # {seq: cosine_to_reference}


def compute_identity_consistency(
    embedder: AdaFaceEmbedder,
    frames_with_seq: list[tuple[int, np.ndarray]],  # (seq, aligned_112_bgr) IN ORDER
    identity_min: float,
) -> IdentityConsistencyResult:
    """`frames_with_seq` must already be Layer-0-QC-passed, aligned 112x112
    crops, in the order they should be considered (first = candidate
    reference). Returns passed=True trivially if fewer than 2 frames survive
    (nothing to compare — Layer 0's MIN_VALID_FRAMES gate is what should
    catch a too-short session, not this layer)."""
    if not frames_with_seq:
        return IdentityConsistencyResult(passed=False, min_similarity=-1.0, reference_seq=-1)

    embeddings = [(seq, embedder.embed_aligned(crop)) for seq, crop in frames_with_seq]
    ref_seq, ref_emb = embeddings[0]

    if len(embeddings) < 2:
        return IdentityConsistencyResult(passed=True, min_similarity=1.0, reference_seq=ref_seq)

    pairwise = {}
    sims = []
    for seq, emb in embeddings[1:]:
        sim = AdaFaceEmbedder.cosine(ref_emb, emb)
        pairwise[seq] = round(sim, 4)
        sims.append(sim)

    min_sim = min(sims)
    return IdentityConsistencyResult(
        passed=min_sim >= identity_min,
        min_similarity=round(min_sim, 4),
        reference_seq=ref_seq,
        pairwise=pairwise,
    )
