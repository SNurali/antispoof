"""Challenge-session store + randomized challenge_spec generation.

Per docs/plans/FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md §1.2: this service
(not Laravel) owns challenge generation and issues `session_id` —
POST /liveness/challenge is called by Laravel when it handles the public
POST /liveness/start, and Laravel relays session_id/challenge_spec to the
client unchanged. POST /liveness/verdict then looks the session up by
session_id — the challenge_spec used for grading is ALWAYS the one this
service generated and stored, never a client- or Laravel-supplied spec, so
a compromised/relayed request cannot grade itself against an easier
challenge than the one actually shown to the user.

TWO BACKENDS, same public interface (`create`/`get`/`consume`) so
app/main.py only changes its ONE instantiation call site, not the call
sites that use the store:

- `SessionStore` — in-memory dict + threading.Lock. Single process only;
  does not survive a restart and is NOT shared across worker processes.
  Fine for a single-worker dev/smoke deploy (see app/config.py
  ::SESSION_STORE_BACKEND).
- `RedisSessionStore` — Redis-backed, shared across any number of worker
  processes/replicas. Required as soon as this service runs with more than
  one worker (see app/main.py's WEB_CONCURRENCY startup guard) — this is
  the shared store FACEID_PHASE1_PAD_GATE.md §2 item 10 and
  docs/LIVENESS_CONTRACT_v1.md §4 item 7 flagged as a known limitation.

Backend selection is explicit via `SESSION_STORE_BACKEND=memory|redis`
(app/config.py) — never a silent fallback. If `redis` is selected and the
Redis server is unreachable at startup, `build_session_store()` raises
rather than quietly degrading to in-memory (a silent fallback under
multi-worker would reintroduce exactly the cross-worker
SESSION_NOT_FOUND bug the Redis backend exists to close).
"""
import json
import logging
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChallengeSession:
    """`correlation_id` is the R2 SESSION-BINDING key (per backend
    confirmation, agent-mesh 2026-07-17, umid-agent msg 1784265688632-0):
    it is minted by Laravel in POST /liveness/start and must match on the
    later /liveness/verdict call for the same session_id — see
    app/main.py::_run_liveness_verdict. `transaction_ref` (and
    `transaction_type`) are stored for audit/logging passthrough only; the
    sale natural key can legitimately not be final yet when the challenge
    is issued, so it is intentionally NOT part of the binding check."""

    session_id: str
    steps: list[str]
    t_instruction_shown: float  # time.time() when issued
    expires_at: float
    correlation_id: str
    transaction_type: str
    transaction_ref: str
    used: bool = False


def generate_challenge_spec(steps_pool: list[str], step_count: int, rng: Optional[random.Random] = None) -> list[str]:
    """Randomized subset+order from the pool. See app/config.py
    ::LIVENESS_CHALLENGE_STEPS_POOL for why the pool (and therefore the
    entropy against video-replay) is small in this increment."""
    rng = rng or random
    pool = list(steps_pool)
    k = min(step_count, len(pool))
    chosen = rng.sample(pool, k)
    return chosen


class SessionStore:
    """Thread-safe in-memory session store with lazy expiry sweep."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, ChallengeSession] = {}

    def create(
        self,
        steps: list[str],
        ttl_s: float,
        correlation_id: str,
        transaction_type: str,
        transaction_ref: str,
    ) -> ChallengeSession:
        now = time.time()
        session = ChallengeSession(
            session_id=str(uuid.uuid4()),
            steps=steps,
            t_instruction_shown=now,
            expires_at=now + ttl_s,
            correlation_id=correlation_id,
            transaction_type=transaction_type,
            transaction_ref=transaction_ref,
        )
        with self._lock:
            self._sessions[session.session_id] = session
            self._sweep_expired_locked()
        return session

    def get(self, session_id: str) -> Optional[ChallengeSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def consume(self, session_id: str) -> tuple[Optional[ChallengeSession], Optional[str]]:
        """Atomically fetch + mark used. Returns (session, error_reason).
        error_reason in {None, "SESSION_NOT_FOUND", "SESSION_EXPIRED", "SESSION_ALREADY_USED"}."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None, "SESSION_NOT_FOUND"
            if session.used:
                return None, "SESSION_ALREADY_USED"
            if time.time() > session.expires_at:
                return None, "SESSION_EXPIRED"
            session.used = True
            return session, None

    def _sweep_expired_locked(self) -> None:
        now = time.time()
        # Keep a generous grace window past expiry (2x TTL) so a
        # late-but-still-informative /liveness/verdict call can still be
        # attributed to SESSION_EXPIRED rather than SESSION_NOT_FOUND in
        # the audit log, without the dict growing unbounded.
        stale = [sid for sid, s in self._sessions.items() if now > s.expires_at * 2 + 60]
        for sid in stale:
            del self._sessions[sid]


class RedisSessionStore:
    """Redis-backed challenge session store — SAME public interface as
    `SessionStore` (`create`/`get`/`consume`), so app/main.py's call sites
    do not change, only which class gets instantiated.

    Atomicity: the in-memory store used `threading.Lock` to make
    `consume()` (fetch-and-mark-used) atomic — that only protects a single
    process, so it does nothing for two different worker processes racing
    to consume the same `session_id`. Here the fetch-check-mark sequence
    runs as a single Lua script via `EVAL`, which Redis always executes as
    one atomic step (Redis is single-threaded for command/script
    execution) regardless of how many separate client processes call it
    concurrently — no client-side lock needed or possible across processes.

    TTL: the Redis key TTL is set to `ttl_s*2 + 60` — the SAME grace
    window `SessionStore._sweep_expired_locked` used above — so a
    late-but-still-informative `/liveness/verdict` call can still be told
    apart as `SESSION_EXPIRED` (checked via the `expires_at` field stored
    in the session payload) rather than Redis having already reaped the
    key at exactly `ttl_s` and collapsing that case into
    `SESSION_NOT_FOUND`. The storage TTL is only for eventual cleanup; the
    actual challenge validity window (`ttl_s`) is enforced by the
    `expires_at` check inside the Lua script below, mirroring the
    in-memory store's `time.time() > session.expires_at` check exactly.
    """

    _KEY_PREFIX = "antispoof:liveness:session:"

    # KEYS[1] = session key, ARGV[1] = current time.time() as a string.
    # Returns a 2-element array: [status, json_payload_or_nil].
    # status in {"OK", "NOT_FOUND", "ALREADY_USED", "EXPIRED"}.
    _CONSUME_SCRIPT = """
    local raw = redis.call('GET', KEYS[1])
    if raw == false then
      return {'NOT_FOUND', false}
    end
    local data = cjson.decode(raw)
    if data.used then
      return {'ALREADY_USED', false}
    end
    if tonumber(ARGV[1]) > data.expires_at then
      return {'EXPIRED', false}
    end
    data.used = true
    local newraw = cjson.encode(data)
    local ttl = redis.call('TTL', KEYS[1])
    if ttl and ttl > 0 then
      redis.call('SET', KEYS[1], newraw, 'EX', ttl)
    else
      redis.call('SET', KEYS[1], newraw)
    end
    return {'OK', newraw}
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._consume_script = self._redis.register_script(self._CONSUME_SCRIPT)

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    @staticmethod
    def _decode(value):
        return value.decode() if isinstance(value, bytes) else value

    @classmethod
    def _to_session(cls, data: dict) -> ChallengeSession:
        return ChallengeSession(**data)

    def create(
        self,
        steps: list[str],
        ttl_s: float,
        correlation_id: str,
        transaction_type: str,
        transaction_ref: str,
    ) -> ChallengeSession:
        now = time.time()
        session = ChallengeSession(
            session_id=str(uuid.uuid4()),
            steps=steps,
            t_instruction_shown=now,
            expires_at=now + ttl_s,
            correlation_id=correlation_id,
            transaction_type=transaction_type,
            transaction_ref=transaction_ref,
        )
        # Same 2x+60s grace window as SessionStore._sweep_expired_locked —
        # storage cleanup only, NOT the business-logic expiry check (see
        # class docstring above).
        storage_ttl_s = int(ttl_s * 2 + 60) + 1
        self._redis.set(self._key(session.session_id), json.dumps(asdict(session)), ex=storage_ttl_s)
        return session

    def get(self, session_id: str) -> Optional[ChallengeSession]:
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return self._to_session(json.loads(raw))

    def consume(self, session_id: str) -> tuple[Optional[ChallengeSession], Optional[str]]:
        """Atomically fetch + mark used via a Redis Lua script — see class
        docstring. Returns (session, error_reason), error_reason in
        {None, "SESSION_NOT_FOUND", "SESSION_EXPIRED", "SESSION_ALREADY_USED"},
        i.e. the exact same contract as SessionStore.consume()."""
        status, payload = self._consume_script(keys=[self._key(session_id)], args=[str(time.time())])
        status = self._decode(status)
        if status == "OK":
            return self._to_session(json.loads(self._decode(payload))), None
        error_map = {
            "NOT_FOUND": "SESSION_NOT_FOUND",
            "ALREADY_USED": "SESSION_ALREADY_USED",
            "EXPIRED": "SESSION_EXPIRED",
        }
        return None, error_map.get(status, "SESSION_NOT_FOUND")

    def ping(self) -> bool:
        """Used at startup to fail fast (not silently fall back to
        in-memory) if SESSION_STORE_BACKEND=redis but Redis is
        unreachable — see build_session_store() below."""
        return bool(self._redis.ping())


def build_session_store(settings):
    """Selects and constructs the session store backend from
    `settings.SESSION_STORE_BACKEND` ("memory" | "redis"). This is the
    ONLY place backend selection happens — app/main.py calls this instead
    of constructing SessionStore()/RedisSessionStore() directly, so
    switching backends is an env-var change (SESSION_STORE_BACKEND,
    REDIS_URL), not a code change.

    No silent fallback: an unreachable Redis with SESSION_STORE_BACKEND=redis
    raises RuntimeError at startup rather than quietly running in-memory —
    a silent fallback under a multi-worker deploy would reintroduce the
    exact cross-worker SESSION_NOT_FOUND bug this backend exists to close."""
    backend = (settings.SESSION_STORE_BACKEND or "memory").strip().lower()
    if backend == "memory":
        logger.info("liveness session store backend=memory (single-process only)")
        return SessionStore()
    if backend == "redis":
        import redis as redis_lib

        client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        store = RedisSessionStore(client)
        try:
            store.ping()
        except Exception as exc:  # noqa: BLE001 — re-raised with actionable context below
            raise RuntimeError(
                f"SESSION_STORE_BACKEND=redis but Redis at {settings.REDIS_URL!r} is "
                f"unreachable ({exc!r}) — refusing to silently fall back to the "
                "in-memory store. Start/fix Redis, or set SESSION_STORE_BACKEND=memory "
                "for a single-worker dev deploy."
            ) from exc
        logger.info("liveness session store backend=redis url=%s (shared across workers)", settings.REDIS_URL)
        return store
    raise ValueError(f"Unknown SESSION_STORE_BACKEND={backend!r} — expected 'memory' or 'redis'")
