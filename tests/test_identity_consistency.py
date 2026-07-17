"""Tests for app/identity_consistency.py — Layer 3 cross-frame identity (top
priority of the 2026-07-17 increment). Uses a fake embedder (no real ONNX
weights needed) — AdaFaceEmbedder.cosine() is a pure static function reused
directly, so these tests exercise the SAME similarity math the real pipeline
uses without requiring the 260MB weight file in CI.
"""
import numpy as np
import pytest

from app.identity_consistency import compute_identity_consistency


class _FakeEmbedder:
    """embed_aligned() returns whatever vector the caller pre-registered for
    that crop object (matched by identity), so tests can control embeddings
    directly without running a real model."""

    def __init__(self, mapping: dict):
        self._mapping = mapping  # id(crop) -> np.ndarray

    def embed_aligned(self, crop):
        return self._mapping[id(crop)]

    @staticmethod
    def cosine(a, b):
        from app.adaface import AdaFaceEmbedder
        return AdaFaceEmbedder.cosine(a, b)


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestComputeIdentityConsistency:
    def test_empty_frames_fails(self):
        result = compute_identity_consistency(_FakeEmbedder({}), [], identity_min=0.4)
        assert result.passed is False
        assert result.min_similarity == -1.0

    def test_single_frame_trivially_passes(self):
        crop = object()
        embedder = _FakeEmbedder({id(crop): _unit([1, 0, 0])})
        result = compute_identity_consistency(embedder, [(0, crop)], identity_min=0.4)
        assert result.passed is True
        assert result.min_similarity == 1.0

    def test_identical_identity_across_frames_passes(self):
        c0, c1, c2 = object(), object(), object()
        v = _unit([1, 0, 0])
        embedder = _FakeEmbedder({id(c0): v, id(c1): v, id(c2): v})
        result = compute_identity_consistency(
            embedder, [(0, c0), (1, c1), (2, c2)], identity_min=0.4,
        )
        assert result.passed is True
        assert result.min_similarity == pytest.approx(1.0)
        assert result.reference_seq == 0

    def test_swapped_identity_mid_session_fails(self):
        """The exact attack Layer 3 exists to catch: frames 0-1 are one
        person, frame 2 is a different person (near-orthogonal embedding)."""
        c0, c1, c2 = object(), object(), object()
        same = _unit([1, 0, 0])
        different = _unit([0, 1, 0])
        embedder = _FakeEmbedder({id(c0): same, id(c1): same, id(c2): different})
        result = compute_identity_consistency(
            embedder, [(0, c0), (1, c1), (2, c2)], identity_min=0.4,
        )
        assert result.passed is False
        assert result.min_similarity < 0.4
        assert result.pairwise[2] < 0.4

    def test_min_similarity_is_the_worst_pairwise_not_the_average(self):
        """One bad pair must fail the whole session even if other pairs are
        strong — averaging would let a single swapped frame hide behind
        good matches on the rest (exactly the compensation ML_CORE §3
        forbids for the active-challenge gate; the same principle applies
        here: identity consistency is a floor, not a mean)."""
        c0, c1, c2 = object(), object(), object()
        ref = _unit([1, 0, 0])
        close = _unit([0.95, 0.05, 0.0])
        far = _unit([0.1, 0.9, 0.0])
        embedder = _FakeEmbedder({id(c0): ref, id(c1): close, id(c2): far})
        result = compute_identity_consistency(
            embedder, [(0, c0), (1, c1), (2, c2)], identity_min=0.4,
        )
        assert result.min_similarity == result.pairwise[2]
        assert result.passed is False

    def test_threshold_boundary_is_inclusive(self):
        c0, c1 = object(), object()
        ref = np.array([1.0, 0.0], dtype=np.float32)
        # construct a vector at EXACTLY cosine=0.4 from ref
        angle = np.arccos(0.4)
        other = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        embedder = _FakeEmbedder({id(c0): ref, id(c1): other})
        result = compute_identity_consistency(embedder, [(0, c0), (1, c1)], identity_min=0.4)
        assert result.passed is True
        assert result.min_similarity == pytest.approx(0.4, abs=1e-3)
