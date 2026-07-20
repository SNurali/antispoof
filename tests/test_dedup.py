"""Tests for app/dedup_store.py and its wiring into POST /pad/check.

Covers the three mechanisms built in response to the 2026-07-20 fraud
incident (same photo accepted for two different abonents, 46s apart, one
inspector) — see app/dedup_store.py module docstring for the full design:

1. pHash frame-reuse dedup — HARD BLOCK (verdict=spoof, reason=DUPLICATE_PHOTO)
2. AdaFace-embedding dedup — ALERT ONLY, never blocks
3. Inspector/abonent fraud-pattern heuristic — SOFT, log-only

Unit tests exercise app.dedup_store.DedupStore directly (fast, no FastAPI/
model dependency, full control over the `now` clock for TTL/window tests).
HTTP tests exercise the real /pad/check wiring end-to-end with a
per-test-isolated in-memory DedupStore (":memory:") swapped into
app.main.dedup_store — never the shared module-level singleton, so these
tests cannot pollute (or be polluted by) any other test file that imports
app.main.
"""
import base64
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

os.environ.setdefault("SERVICE_TOKEN", "")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("RATE_LIMIT_BURST", "1000")
os.environ.setdefault("RATE_LIMIT_SUSTAINED", "1000.0")
os.environ["ANTISPOOF_SKIP_MODELS"] = "1"

from app.dedup_store import DedupStore, compute_phash, phash_hamming_distance  # noqa: E402


def _make_test_image(width: int = 200, height: int = 200, seed: int = 0) -> np.ndarray:
    """Deterministic-but-distinguishable BGR image. `seed` changes the
    circle color/position enough to produce a genuinely different pHash —
    NOT just JPEG re-noise of the same content."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx = width // 2 + (seed * 17) % 40 - 20
    cy = height // 2 + (seed * 29) % 40 - 20
    color = (200 - seed * 5 % 100, 180, 160 - seed * 3 % 80)
    cv2.circle(img, (cx, cy), min(width, height) // 3, color, -1)
    return img


def _encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_base64_image(seed: int = 0) -> str:
    return base64.b64encode(_encode_jpeg(_make_test_image(seed=seed))).decode()


# ---------------------------------------------------------------------------
# Unit tests — compute_phash / phash_hamming_distance
# ---------------------------------------------------------------------------

class TestPHash:
    def test_identical_image_has_zero_hamming_distance(self):
        img = _make_test_image(seed=1)
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert h1 == h2
        assert phash_hamming_distance(h1, h2) == 0

    def test_jpeg_recompression_keeps_hash_close(self):
        """Re-encoding the SAME image at a different JPEG quality (what a
        client retry / re-upload would produce) must stay within the
        DEDUP_PHASH_HAMMING_MAX default (4) — this is the exact scenario
        the hard-block gate must catch."""
        img = _make_test_image(seed=2)
        h_original = compute_phash(img)
        _, buf_q50 = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        recompressed = cv2.imdecode(np.frombuffer(buf_q50.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        h_recompressed = compute_phash(recompressed)
        assert phash_hamming_distance(h_original, h_recompressed) <= 4

    def test_different_images_have_large_hamming_distance(self):
        """Two genuinely different photos must NOT collide within the
        default Hamming threshold — this is the FRR-safety property the
        whole hard-block design depends on."""
        img_a = _make_test_image(seed=1)
        img_b = _make_test_image(seed=9)
        h_a = compute_phash(img_a)
        h_b = compute_phash(img_b)
        assert phash_hamming_distance(h_a, h_b) > 4

    def test_hex_string_is_64_bits(self):
        h = compute_phash(_make_test_image())
        assert len(h) == 16  # 64 bits / 4 bits-per-hex-char
        int(h, 16)  # must parse as hex without raising


# ---------------------------------------------------------------------------
# Unit tests — DedupStore.check_and_record_phash (hard block)
# ---------------------------------------------------------------------------

class TestDedupStorePhashBlock:
    @pytest.fixture
    def store(self):
        return DedupStore(Path(":memory:"), ttl_days=90.0)

    def test_first_frame_never_matches(self, store):
        h = compute_phash(_make_test_image(seed=1))
        match = store.check_and_record_phash(h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1")
        assert match is None

    def test_same_photo_different_transaction_ref_is_hard_blocked(self, store):
        """THE core incident scenario: same photo, two different sales."""
        h = compute_phash(_make_test_image(seed=1))
        store.check_and_record_phash(h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1")
        match = store.check_and_record_phash(h, hamming_max=4, correlation_id="c2", transaction_ref="req2:bal2")
        assert match is not None
        assert match.correlation_id == "c1"
        assert match.transaction_ref == "req1:bal1"
        assert match.hamming_distance == 0

    def test_same_photo_same_transaction_ref_is_allowed(self):
        """Legitimate client retry (network retry resending the identical
        base64 payload for the SAME sale) must NOT be flagged as fraud —
        see module docstring / docs/BACKEND_REQUIREMENTS_2026-07-06_otvet_final.md
        item 3 (Idempotency-Key retry contract)."""
        store = DedupStore(Path(":memory:"), ttl_days=90.0)
        h = compute_phash(_make_test_image(seed=1))
        store.check_and_record_phash(h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1")
        match = store.check_and_record_phash(h, hamming_max=4, correlation_id="c1-retry", transaction_ref="req1:bal1")
        assert match is None

    def test_different_photos_not_flagged(self, store):
        h1 = compute_phash(_make_test_image(seed=1))
        h2 = compute_phash(_make_test_image(seed=9))
        store.check_and_record_phash(h1, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1")
        match = store.check_and_record_phash(h2, hamming_max=4, correlation_id="c2", transaction_ref="req2:bal2")
        assert match is None

    def test_ttl_expiry_stops_matching(self, store):
        """A duplicate older than the TTL window must no longer be
        considered a match (retention == 90 days by default, tested here
        with an injected `now`)."""
        h = compute_phash(_make_test_image(seed=1))
        t_old = 1_000_000.0
        store.check_and_record_phash(
            h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1", now=t_old,
        )
        ttl_s = 90.0 * 86400.0
        t_after_ttl = t_old + ttl_s + 3600.0  # 1h past the TTL window
        match = store.check_and_record_phash(
            h, hamming_max=4, correlation_id="c2", transaction_ref="req2:bal2", now=t_after_ttl,
        )
        assert match is None

    def test_within_ttl_still_matches(self, store):
        h = compute_phash(_make_test_image(seed=1))
        t_old = 1_000_000.0
        store.check_and_record_phash(
            h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1", now=t_old,
        )
        t_within = t_old + 86400.0 * 89  # 89 days later, still inside 90-day TTL
        match = store.check_and_record_phash(
            h, hamming_max=4, correlation_id="c2", transaction_ref="req2:bal2", now=t_within,
        )
        assert match is not None

    def test_repeated_reuse_recorded_as_separate_rows(self, store):
        """Every attempt (including duplicates) is recorded — a third reuse
        of the same photo must still be caught, and the audit trail should
        reflect every attempt, not just the first."""
        h = compute_phash(_make_test_image(seed=1))
        store.check_and_record_phash(h, hamming_max=4, correlation_id="c1", transaction_ref="req1:bal1")
        store.check_and_record_phash(h, hamming_max=4, correlation_id="c2", transaction_ref="req2:bal2")
        match3 = store.check_and_record_phash(h, hamming_max=4, correlation_id="c3", transaction_ref="req3:bal3")
        assert match3 is not None  # matches SOMETHING (c1 or c2), still blocked


# ---------------------------------------------------------------------------
# Unit tests — DedupStore embedding alert (never blocks)
# ---------------------------------------------------------------------------

class TestDedupStoreEmbeddingAlert:
    @pytest.fixture
    def store(self):
        return DedupStore(Path(":memory:"), ttl_days=90.0)

    def _record_with_embedding(self, store, correlation_id, transaction_ref, embedding, abonent_id=None, inspector_id=None):
        h = compute_phash(_make_test_image(seed=hash(correlation_id) % 10))
        store.check_and_record_phash(
            h, hamming_max=4, correlation_id=correlation_id, transaction_ref=transaction_ref,
            abonent_id=abonent_id, inspector_id=inspector_id,
        )
        store.record_embedding(correlation_id, embedding)

    def test_same_person_different_transaction_flagged_as_alert(self, store):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._record_with_embedding(store, "c1", "req1:bal1", emb, abonent_id=None)
        matches = store.check_embedding_alert(emb, cosine_min=0.4, transaction_ref="req2:bal2")
        assert len(matches) == 1
        assert matches[0].correlation_id == "c1"
        assert matches[0].cosine_similarity == pytest.approx(1.0)

    def test_same_abonent_excluded_repeat_customer_not_alerted(self, store):
        """A repeat customer (same abonent_id) buying again is the expected
        case, not fraud — must be excluded from the alert."""
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._record_with_embedding(store, "c1", "req1:bal1", emb, abonent_id="abonent-42")
        matches = store.check_embedding_alert(
            emb, cosine_min=0.4, transaction_ref="req2:bal2", exclude_abonent_id="abonent-42",
        )
        assert matches == []

    def test_dissimilar_embedding_not_flagged(self, store):
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._record_with_embedding(store, "c1", "req1:bal1", emb_a)
        matches = store.check_embedding_alert(emb_b, cosine_min=0.4, transaction_ref="req2:bal2")
        assert matches == []

    def test_same_transaction_ref_excluded(self, store):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._record_with_embedding(store, "c1", "req1:bal1", emb)
        matches = store.check_embedding_alert(emb, cosine_min=0.4, transaction_ref="req1:bal1")
        assert matches == []


# ---------------------------------------------------------------------------
# Unit tests — inspector/abonent fraud-pattern heuristic
# ---------------------------------------------------------------------------

class TestInspectorFraudAlert:
    @pytest.fixture
    def store(self):
        return DedupStore(Path(":memory:"), ttl_days=90.0)

    def test_below_threshold_no_alert(self, store):
        store.record_inspector_activity("insp-1", "abonent-A", "c1", "req1:bal1")
        store.record_inspector_activity("insp-1", "abonent-B", "c2", "req2:bal2")
        alert = store.check_inspector_fraud_alert("insp-1", window_s=300.0, distinct_abonent_max=3)
        assert alert is None

    def test_reaching_threshold_triggers_alert(self, store):
        """Mirrors the incident shape: one inspector, several different
        abonents in a short window."""
        store.record_inspector_activity("insp-1", "abonent-A", "c1", "req1:bal1")
        store.record_inspector_activity("insp-1", "abonent-B", "c2", "req2:bal2")
        store.record_inspector_activity("insp-1", "abonent-C", "c3", "req3:bal3")
        alert = store.check_inspector_fraud_alert("insp-1", window_s=300.0, distinct_abonent_max=3)
        assert alert is not None
        assert alert.distinct_abonent_count == 3
        assert set(alert.abonent_ids) == {"abonent-A", "abonent-B", "abonent-C"}

    def test_same_abonent_repeated_does_not_count_twice(self, store):
        store.record_inspector_activity("insp-1", "abonent-A", "c1", "req1:bal1")
        store.record_inspector_activity("insp-1", "abonent-A", "c2", "req2:bal2")
        store.record_inspector_activity("insp-1", "abonent-A", "c3", "req3:bal3")
        alert = store.check_inspector_fraud_alert("insp-1", window_s=300.0, distinct_abonent_max=3)
        assert alert is None  # only 1 distinct abonent, no matter how many sales

    def test_outside_window_not_counted(self, store):
        t0 = 1_000_000.0
        store.record_inspector_activity("insp-1", "abonent-A", "c1", "req1:bal1", now=t0)
        store.record_inspector_activity("insp-1", "abonent-B", "c2", "req2:bal2", now=t0 + 100)
        store.record_inspector_activity("insp-1", "abonent-C", "c3", "req3:bal3", now=t0 + 1000)  # > 300s later
        alert = store.check_inspector_fraud_alert("insp-1", window_s=300.0, distinct_abonent_max=3, now=t0 + 1000)
        assert alert is None  # only abonent-C is within the last 300s

    def test_different_inspectors_isolated(self, store):
        store.record_inspector_activity("insp-1", "abonent-A", "c1", "req1:bal1")
        store.record_inspector_activity("insp-2", "abonent-B", "c2", "req2:bal2")
        store.record_inspector_activity("insp-2", "abonent-C", "c3", "req3:bal3")
        assert store.check_inspector_fraud_alert("insp-1", window_s=300.0, distinct_abonent_max=2) is None
        alert = store.check_inspector_fraud_alert("insp-2", window_s=300.0, distinct_abonent_max=2)
        assert alert is not None
        assert alert.inspector_id == "insp-2"


# ---------------------------------------------------------------------------
# HTTP-level integration — POST /pad/check wiring
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_startup():
    import app.main as m
    from app.main import app
    original_handlers = app.router.on_startup.copy()
    app.router.on_startup.clear()
    m._rate_limiter._windows.clear()
    m._rate_limiter._burst = 1000
    m._rate_limiter._sustained = 1000.0
    yield
    app.router.on_startup = original_handlers


@pytest.fixture
def client():
    """Same mocked-model pattern as tests/test_pad_check.py, PLUS a fresh
    per-test in-memory DedupStore swapped into app.main.dedup_store — never
    the module-level singleton, so these tests cannot leak state into (or
    be polluted by) any other test file that imports app.main."""
    import app.main as m

    mock_detector = MagicMock()
    mock_detector.detect.return_value = [50, 50, 100, 100]
    mock_engine = MagicMock()
    mock_engine.predict.return_value = ("real", 0.95, True, {
        "signal_scores": {"recapture": 0.1}, "spoof_probability": 0.05,
        "nn_label": "real", "nn_score": 0.95,
    })
    m.detector = mock_detector
    m.engine = mock_engine
    m._models_loaded = True

    old_store = m.dedup_store
    m.dedup_store = DedupStore(Path(":memory:"), ttl_days=90.0)

    old_dedup_enabled = m.settings.DEDUP_ENABLED
    old_fraud_enabled = m.settings.FRAUD_INSPECTOR_ALERT_ENABLED

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c

    m.dedup_store = old_store
    m.settings.DEDUP_ENABLED = old_dedup_enabled
    m.settings.FRAUD_INSPECTOR_ALERT_ENABLED = old_fraud_enabled


def _pad_check_body(correlation_id: str, transaction_ref: str, seed: int = 1, **extra) -> dict:
    body = {
        "correlation_id": correlation_id,
        "transaction_type": "sale",
        "transaction_ref": transaction_ref,
        "face_photo": _make_base64_image(seed=seed),
    }
    body.update(extra)
    return body


class TestPadCheckDedupDisabledByDefault:
    def test_default_disabled_duplicate_photo_not_blocked(self, client):
        """DEDUP_ENABLED defaults False — the SAME photo across two
        different transaction_refs must NOT be blocked until an operator
        explicitly enables it (see app/config.py::DEDUP_ENABLED docstring)."""
        import app.main as m
        assert m.settings.DEDUP_ENABLED is False

        resp1 = client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        resp2 = client.post("/pad/check", json=_pad_check_body("c2", "req2:bal2", seed=1))
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["verdict"] == "live"
        assert resp2.json()["verdict"] == "live"  # NOT flagged — flag is off


class TestPadCheckDedupEnabled:
    def test_duplicate_photo_across_different_sales_is_blocked(self, client):
        import app.main as m
        m.settings.DEDUP_ENABLED = True

        resp1 = client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        assert resp1.status_code == 200
        assert resp1.json()["verdict"] == "live"

        resp2 = client.post("/pad/check", json=_pad_check_body("c2", "req2:bal2", seed=1))
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["verdict"] == "spoof"
        assert data2["reason"] == "DUPLICATE_PHOTO"
        assert data2["save_frame"] is True
        assert data2["signals"]["dedup_check"]["matched_correlation_id"] == "c1"
        assert data2["signals"]["dedup_check"]["matched_transaction_ref"] == "req1:bal1"

    def test_retry_same_transaction_ref_not_blocked(self, client):
        """Legitimate client retry (same sale, same photo bytes resent)
        must not be treated as fraud."""
        import app.main as m
        m.settings.DEDUP_ENABLED = True

        resp1 = client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        resp2 = client.post("/pad/check", json=_pad_check_body("c1-retry", "req1:bal1", seed=1))
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["verdict"] == "live"
        assert resp2.json()["verdict"] == "live"

    def test_different_photos_different_sales_both_pass(self, client):
        import app.main as m
        m.settings.DEDUP_ENABLED = True

        resp1 = client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        resp2 = client.post("/pad/check", json=_pad_check_body("c2", "req2:bal2", seed=9))
        assert resp1.json()["verdict"] == "live"
        assert resp2.json()["verdict"] == "live"

    def test_duplicate_check_does_not_create_new_files_on_disk(self, client, tmp_path):
        """Same invariant tests/test_pad_check.py::TestNoFrameStorage
        enforces for the base /pad/check path — dedup must not write a raw
        frame to disk either (the in-memory `:memory:` store used by this
        fixture never touches disk at all, but this pins the property
        explicitly for the dedup code path)."""
        import app.main as m
        m.settings.DEDUP_ENABLED = True

        before = set(tmp_path.rglob("*"))
        client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        client.post("/pad/check", json=_pad_check_body("c2", "req2:bal2", seed=1))
        after = set(tmp_path.rglob("*"))
        assert before == after


class TestPadCheckFraudAlert:
    def test_optional_fields_absent_is_backward_compatible(self, client):
        """No abonent_id/inspector_id sent — request must behave exactly as
        before this feature existed."""
        resp = client.post("/pad/check", json=_pad_check_body("c1", "req1:bal1", seed=1))
        assert resp.status_code == 200
        assert "fraud_alert" not in resp.json()["signals"]

    def test_multi_abonent_inspector_pattern_flagged(self, client):
        import app.main as m
        assert m.settings.FRAUD_INSPECTOR_ALERT_ENABLED is True  # default on, see app/config.py

        for i, seed in enumerate([1, 2, 3], start=1):
            resp = client.post("/pad/check", json=_pad_check_body(
                f"c{i}", f"req{i}:bal{i}", seed=seed,
                abonent_id=f"abonent-{i}", inspector_id="insp-ODILOV",
            ))
            assert resp.status_code == 200

        data = resp.json()
        assert data["signals"]["fraud_alert"]["type"] == "INSPECTOR_MULTI_ABONENT"
        assert data["signals"]["fraud_alert"]["inspector_id"] == "insp-ODILOV"
        assert data["signals"]["fraud_alert"]["distinct_abonent_count"] == 3

    def test_single_abonent_repeated_not_flagged(self, client):
        """Same inspector, SAME abonent across multiple sales — not a
        fraud pattern (e.g. multiple cylinders in one visit)."""
        for i, seed in enumerate([1, 2, 3], start=1):
            resp = client.post("/pad/check", json=_pad_check_body(
                f"c{i}", f"req{i}:bal{i}", seed=seed,
                abonent_id="abonent-1", inspector_id="insp-1",
            ))
        assert "fraud_alert" not in resp.json()["signals"]

    def test_disabled_flag_never_alerts(self, client):
        import app.main as m
        m.settings.FRAUD_INSPECTOR_ALERT_ENABLED = False
        for i, seed in enumerate([1, 2, 3], start=1):
            resp = client.post("/pad/check", json=_pad_check_body(
                f"c{i}", f"req{i}:bal{i}", seed=seed,
                abonent_id=f"abonent-{i}", inspector_id="insp-1",
            ))
        assert "fraud_alert" not in resp.json()["signals"]
