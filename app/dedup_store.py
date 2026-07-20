"""Frame-reuse dedup + inspector/abonent fraud-pattern alerting for /pad/check.

Built in response to a real production fraud incident (2026-07-20): the SAME
photo (bit-identical or near-identical file) was accepted by `/pad/check` for
TWO DIFFERENT abonents on ONE sale request, submitted by the same inspector
(ODILOV) 46 seconds apart. The service is stateless today — nothing stops the
exact same JPEG bytes from being replayed for an unlimited number of
different sales. This module closes that specific gap.

Two independent mechanisms, deliberately different strictness:

1. **pHash dedup — HARD BLOCK** (`check_and_record_phash`). A perceptual hash
   of the full frame (not just the face crop — catches "same photo file
   reused", the exact incident shape) is compared against every OTHER
   frame's hash recorded in the last `DEDUP_TTL_DAYS` days. A near-identical
   match belonging to a DIFFERENT `transaction_ref` is a hard reject
   (`verdict=spoof`, `reason=DUPLICATE_PHOTO`) — the same photo cannot
   legitimately be the live capture for two different sales. A match
   belonging to the SAME `transaction_ref` is NOT flagged — that is an
   expected client retry (network retry re-sending the identical base64
   payload, `Idempotency-Key`-style), not fraud; see
   `docs/BACKEND_REQUIREMENTS_2026-07-06_otvet_final.md` item 3 for the
   retry contract this must stay compatible with.

2. **AdaFace embedding dedup — ALERT ONLY, NEVER A BLOCK**
   (`check_embedding_alert`). Deliberately weaker than the Kimi K2 review's
   proposed spec (which suggested embedding-match as a reject gate too) —
   see the correction called out by the owner: the SAME real customer
   legitimately buys gas cylinders on different days/requests, so a
   same-person embedding match across two different `transaction_ref`s is
   the EXPECTED common case, not fraud. Blocking on it would be a systemic
   false-positive that punishes repeat customers. This is surfaced as a soft
   `signals.dedup_check.embedding_alert` field for investigation only — it
   never changes `verdict`. See app/main.py's call site for why this is
   OFF by default (LIVENESS_ENDPOINTS_ENABLED + DEDUP_EMBEDDING_ALERT_ENABLED
   both required — no embedding is computed on the default /pad/check path,
   see app/config.py::DEDUP_EMBEDDING_ALERT_ENABLED docstring for the
   latency-budget reasoning).

A third, independent mechanism lives here too because it shares the same
storage/TTL machinery:

3. **Inspector/abonent fraud-pattern heuristic — SOFT, LOG-ONLY**
   (`record_inspector_activity` / `check_inspector_fraud_alert`). One
   `inspector_id` running sales against an unusually large number of DISTINCT
   `abonent_id`s within a short window is a pattern consistent with (but not
   proof of) the incident shape — flagged into the audit log and an
   additive, non-blocking `signals.fraud_alert` response field for
   downstream escalation (Laravel's own hard/soft fraud-signal contract, not
   decided by this service). Requires the CALLER to send the new optional
   `abonent_id`/`inspector_id` fields on `/pad/check` — a no-op today since no
   caller sends them yet (backward compatible, see PadCheckRequest).

Storage: SQLite, one file, persists across service restarts (unlike the
in-memory `SessionStore` in app/liveness_session.py — a 90-day dedup window
that resets on every restart/deploy would defeat the purpose). Single
writer-lock (`threading.Lock`) serializes access within one process, same
concurrency posture as `SessionStore`. NOT shared across multiple worker
processes/replicas — same limitation the existing `WEB_CONCURRENCY` startup
guard in app/main.py already documents for the in-memory session store; this
module does not (yet) have an equivalent guard because dedup/fraud-alert are
both DEFAULT DISABLED (see app/config.py), so there is nothing to guard
against yet. Revisit if/when DEDUP_ENABLED defaults on AND the service ever
runs with >1 worker.

Performance: dedup lookups are O(n) over all non-expired rows (fetched into
Python, compared via integer XOR + `int.bit_count()` for pHash, `np.dot` for
embeddings) — no bucketing/indexing beyond a `created_at` index. Fine for
this service's real traffic volume (a single inspection point, low
transactions/day); revisit with a proper approximate-nearest-neighbor index
if/when volume grows enough to make this measurably slow within the 2s
/pad/check budget.
"""
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pHash — 64-bit perceptual hash via DCT (same algorithm family as the
# widely-used `imagehash.phash` reference implementation), reimplemented
# here with cv2/numpy only so this module adds ZERO new dependencies
# (requirements.txt already has opencv-python + numpy for every other layer
# in this service).
# ---------------------------------------------------------------------------
def compute_phash(image_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Returns a 64-bit perceptual hash as a 16-char hex string.

    Algorithm: grayscale -> resize to (hash_size*highfreq_factor)^2 ->
    2D DCT -> keep the top-left hash_size x hash_size low-frequency block ->
    threshold each coefficient against the block's median -> 64 bits.
    Robust to JPEG re-compression, minor resize/crop noise, and small
    brightness/contrast shifts — deliberately NOT robust to genuinely
    different photo content (that is exactly what makes it useful as a
    "same photo, maybe re-encoded" detector rather than a general
    similar-image detector)."""
    img_size = hash_size * highfreq_factor
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    median = float(np.median(dct_low))
    bits = (dct_low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size * hash_size // 4}x")


def phash_hamming_distance(hex_a: str, hex_b: str) -> int:
    """Number of differing bits between two pHash hex strings."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DedupMatch:
    correlation_id: str
    transaction_ref: str
    abonent_id: Optional[str]
    inspector_id: Optional[str]
    hamming_distance: int
    age_s: float


@dataclass(frozen=True)
class EmbeddingAlertMatch:
    correlation_id: str
    transaction_ref: str
    abonent_id: Optional[str]
    inspector_id: Optional[str]
    cosine_similarity: float
    age_s: float


@dataclass(frozen=True)
class InspectorFraudAlert:
    inspector_id: str
    distinct_abonent_count: int
    window_s: float
    abonent_ids: list = field(default_factory=list)


class DedupStore:
    """SQLite-backed frame-reuse + inspector-activity store. See module
    docstring for the three mechanisms this backs."""

    def __init__(self, db_path: Path, ttl_days: float = 90.0) -> None:
        self._ttl_s = ttl_days * 86400.0
        self._lock = threading.Lock()
        db_str = str(db_path)
        if db_str != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI's sync route handlers (and the
        # asyncio.to_thread offload for the passive-PAD inference) may run
        # this connection from different worker threads within the SAME
        # process — access is still serialized by self._lock, same pattern
        # SessionStore uses for its in-memory dict.
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        # DELETE (default) journal mode, not WAL — avoids -wal/-shm sidecar
        # files existing on disk between requests, which would otherwise
        # complicate reasoning about "did this request create a new file"
        # (see tests/test_pad_check.py::TestNoFrameStorage, the same
        # invariant this module respects at the DB-file level too).
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dedup_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phash TEXT NOT NULL,
                    embedding BLOB,
                    correlation_id TEXT NOT NULL,
                    transaction_ref TEXT NOT NULL,
                    abonent_id TEXT,
                    inspector_id TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dedup_frames_created_at ON dedup_frames(created_at)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspector_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspector_id TEXT NOT NULL,
                    abonent_id TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    transaction_ref TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspector_activity_inspector_created "
                "ON inspector_activity(inspector_id, created_at)"
            )
            self._conn.commit()

    def _prune_expired_locked(self, now: float) -> None:
        cutoff = now - self._ttl_s
        self._conn.execute("DELETE FROM dedup_frames WHERE created_at < ?", (cutoff,))
        self._conn.execute("DELETE FROM inspector_activity WHERE created_at < ?", (cutoff,))

    # ------------------------------------------------------------------
    # 1. pHash dedup — hard block
    # ------------------------------------------------------------------
    def check_and_record_phash(
        self,
        phash_hex: str,
        hamming_max: int,
        correlation_id: str,
        transaction_ref: str,
        abonent_id: Optional[str] = None,
        inspector_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Optional[DedupMatch]:
        """Returns the closest `DedupMatch` if `phash_hex` is within
        `hamming_max` bits of a frame previously recorded under a DIFFERENT
        `transaction_ref` (same-transaction_ref matches are excluded — see
        module docstring, legitimate client retry). Always records the new
        frame afterward regardless of outcome, so every attempt (including
        a duplicate) is visible in the audit trail as its own row."""
        now = now if now is not None else time.time()
        best: Optional[DedupMatch] = None
        with self._lock:
            self._prune_expired_locked(now)
            cutoff = now - self._ttl_s
            rows = self._conn.execute(
                "SELECT phash, correlation_id, transaction_ref, abonent_id, inspector_id, created_at "
                "FROM dedup_frames WHERE created_at >= ? AND transaction_ref != ?",
                (cutoff, transaction_ref),
            ).fetchall()
            candidate = int(phash_hex, 16)
            best_distance: Optional[int] = None
            for row_phash, row_corr, row_ref, row_abon, row_insp, row_created in rows:
                dist = (candidate ^ int(row_phash, 16)).bit_count()
                if dist <= hamming_max and (best_distance is None or dist < best_distance):
                    best_distance = dist
                    best = DedupMatch(
                        correlation_id=row_corr,
                        transaction_ref=row_ref,
                        abonent_id=row_abon,
                        inspector_id=row_insp,
                        hamming_distance=dist,
                        age_s=round(now - row_created, 1),
                    )
            self._conn.execute(
                "INSERT INTO dedup_frames "
                "(phash, correlation_id, transaction_ref, abonent_id, inspector_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (phash_hex, correlation_id, transaction_ref, abonent_id, inspector_id, now),
            )
            self._conn.commit()
        return best

    def record_embedding(self, correlation_id: str, embedding: np.ndarray) -> None:
        """Attach an AdaFace embedding to the most recent `dedup_frames` row
        for this `correlation_id` (written by `check_and_record_phash` just
        before, in the SAME /pad/check request). No-op if no such row exists
        (defensive only — should not happen given the call order in
        app/main.py)."""
        with self._lock:
            self._conn.execute(
                "UPDATE dedup_frames SET embedding = ? WHERE id = ("
                "  SELECT id FROM dedup_frames WHERE correlation_id = ? ORDER BY id DESC LIMIT 1"
                ")",
                (embedding.astype(np.float32).tobytes(), correlation_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 2. AdaFace embedding dedup — alert only
    # ------------------------------------------------------------------
    def check_embedding_alert(
        self,
        embedding: np.ndarray,
        cosine_min: float,
        transaction_ref: str,
        exclude_abonent_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> list:
        """Returns EmbeddingAlertMatch entries for stored embeddings above
        `cosine_min`, excluding rows for the SAME `transaction_ref` and (if
        `exclude_abonent_id` is known on both sides) the SAME abonent — a
        repeat customer is the expected common case, not an alert-worthy
        pattern. NEVER used to reject a verdict — see module docstring."""
        now = now if now is not None else time.time()
        cutoff = now - self._ttl_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding, correlation_id, transaction_ref, abonent_id, inspector_id, created_at "
                "FROM dedup_frames WHERE created_at >= ? AND transaction_ref != ? AND embedding IS NOT NULL",
                (cutoff, transaction_ref),
            ).fetchall()
        matches: list = []
        for row_emb, row_corr, row_ref, row_abon, row_insp, row_created in rows:
            if exclude_abonent_id is not None and row_abon == exclude_abonent_id:
                continue
            stored = np.frombuffer(row_emb, dtype=np.float32)
            if stored.shape != embedding.shape:
                continue
            sim = float(np.dot(stored, embedding))
            if sim >= cosine_min:
                matches.append(
                    EmbeddingAlertMatch(
                        correlation_id=row_corr,
                        transaction_ref=row_ref,
                        abonent_id=row_abon,
                        inspector_id=row_insp,
                        cosine_similarity=round(sim, 4),
                        age_s=round(now - row_created, 1),
                    )
                )
        matches.sort(key=lambda m: -m.cosine_similarity)
        return matches

    # ------------------------------------------------------------------
    # 3. Inspector/abonent fraud-pattern heuristic — soft, log-only
    # ------------------------------------------------------------------
    def record_inspector_activity(
        self,
        inspector_id: str,
        abonent_id: str,
        correlation_id: str,
        transaction_ref: str,
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO inspector_activity "
                "(inspector_id, abonent_id, correlation_id, transaction_ref, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (inspector_id, abonent_id, correlation_id, transaction_ref, now),
            )
            self._conn.commit()

    def check_inspector_fraud_alert(
        self,
        inspector_id: str,
        window_s: float,
        distinct_abonent_max: int,
        now: Optional[float] = None,
    ) -> Optional[InspectorFraudAlert]:
        """Returns an alert if `inspector_id` has touched >= `distinct_abonent_max`
        DISTINCT `abonent_id`s within the last `window_s` seconds. Pure
        read — does not itself record anything (call `record_inspector_activity`
        first, same request, so the just-recorded row is included)."""
        now = now if now is not None else time.time()
        cutoff = now - window_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT abonent_id FROM inspector_activity "
                "WHERE inspector_id = ? AND created_at >= ?",
                (inspector_id, cutoff),
            ).fetchall()
        abonent_ids = sorted({r[0] for r in rows})
        if len(abonent_ids) >= distinct_abonent_max:
            return InspectorFraudAlert(
                inspector_id=inspector_id,
                distinct_abonent_count=len(abonent_ids),
                window_s=window_s,
                abonent_ids=abonent_ids,
            )
        return None


def build_dedup_store(settings) -> DedupStore:
    """Constructed UNCONDITIONALLY at app/main.py import time (same pattern
    as app/liveness_session.py::build_session_store) regardless of
    DEDUP_ENABLED/DEDUP_EMBEDDING_ALERT_ENABLED/FRAUD_INSPECTOR_ALERT_ENABLED
    — so flipping those flags at runtime does not require a restart, and so
    the SQLite file already exists before the first request in any test
    (required by tests/test_pad_check.py::TestNoFrameStorage, which asserts
    /pad/check does not create NEW files on disk — the DB file must already
    exist beforehand, only rows get appended during a request)."""
    return DedupStore(Path(settings.DEDUP_DB_PATH), ttl_days=settings.DEDUP_TTL_DAYS)
