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
import dataclasses
import json
import logging
import random
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)


class StepWindowDict(TypedDict):
    """LOW finding (MF DOOM code review, 2026-07-20): single source of truth
    for the `step_windows` dict shape (`{step, min_delay_ms, max_delay_ms}`)
    — previously this key set was duplicated informally across three call
    sites (`generate_step_windows`'s return value here,
    `ChallengeSession.step_windows`'s stored copy, and
    `app/main.py::_validate_step_windows`'s reads via `window["step"]` /
    `window["min_delay_ms"]` / `window["max_delay_ms"]`) with nothing tying
    them together — a key rename in one place would silently desync from the
    others with no type-checker signal. `TypedDict` costs nothing at runtime
    (still a plain dict), but gives every one of those three sites the same
    static contract."""

    step: str
    min_delay_ms: int
    max_delay_ms: int


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
    # Фаза 2 (docs/plans/CHALLENGE_ENTROPY_SPRINT_v1.md §5.3), НОВОЕ поле —
    # окна тайминга сэмплированы ОДИН раз при выдаче challenge и должны быть
    # той же самой копией, что и в LivenessChallengeResponse.challenge_spec
    # (не пересэмплированы заново на verdict) — сессия единственный источник
    # истины, тот же принцип, что уже действует для `steps`. Список dict'ов
    # с ключами step/min_delay_ms/max_delay_ms (см. generate_step_windows).
    # default_factory=list — старые вызовы .create()/ChallengeSession(...)
    # без этого параметра (напр. существующие тесты) продолжают работать.
    # Тип list[dict] (не list[StepWindowDict]) сознательно — RedisSessionStore
    # десериализует это поле из чистого json.loads() (см. _to_session), где
    # TypedDict не даёт рантайм-гарантий сверх обычного dict; StepWindowDict
    # используется как СТАТИЧЕСКИЙ контракт на самих функциях-генераторах
    # ниже и на _validate_step_windows (app/main.py), а не здесь.
    step_windows: list[dict] = field(default_factory=list)


def generate_challenge_spec(
    steps_pool: list[str],
    step_count_min: int,
    step_count_max: int,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Randomized subset+order from the pool.

    Фаза 0 (CHALLENGE_ENTROPY_SPRINT_v1.md §3): продовый дефолтный ГСЧ —
    `secrets.SystemRandom()` (криптографически стойкий, на `os.urandom`), а
    НЕ модуль `random` (предсказуемый Mersenne Twister, не годится как
    единственный барьер энтропии против подготовленной video-replay атаки).
    `secrets.SystemRandom` — drop-in подкласс `random.Random` (тот же
    интерфейс `.sample`/`.randint`), поэтому сигнатура не меняется и
    инъекция детерминированного `rng=random.Random(seed)` в тестах
    продолжает работать как раньше.

    Фаза 2 (§5.1): `step_count` стал диапазоном `[step_count_min,
    step_count_max]` — конкретное k выбирается случайно ПРИ КАЖДОЙ генерации
    (не фиксированное число, как раньше). Обе границы дополнительно КЛАМПЯТСЯ
    к текущему размеру пула — если пул МЕНЬШЕ step_count_min, диапазон
    честно ужимается до [len(pool), len(pool)], т.е. k=len(pool)
    детерминированно, а не падение `rng.sample`/`rng.randint` на
    невозможном диапазоне и не тихая порча поведения. Это было прод-
    поведением до 2026-07-21 включительно (пул = 2 шага: TURN_LEFT/
    TURN_RIGHT, MIN/MAX=3/4 клампился в k=2 детерминированно — см.
    tests/test_liveness_session.py::test_pool_of_two_stays_deterministic_
    with_min3_max4, оставлен как регрессионный тест старого поведения при
    искусственно урезанном пуле, не как описание сегодняшнего прода).
    С 2026-07-21 (RZA, NOD_UP/NOD_DOWN в app/config.py::
    LIVENESS_CHALLENGE_STEPS_POOL) пул = 4 шага, и клампинг больше НЕ
    схлопывает диапазон — прод реально сэмплирует k в {3, 4} из этих 4
    шагов при каждой генерации (см. tests/test_liveness_session.py::
    test_k_distribution_across_pool_of_four_or_more). Дальнейшее
    расширение пула (BLINK/SMILE) остаётся Фазой 5 (волновая калибровка),
    отдельно от этого изменения.
    """
    rng = rng or secrets.SystemRandom()
    pool = list(steps_pool)
    hi = min(step_count_max, len(pool))
    lo = min(step_count_min, hi)
    k = rng.randint(lo, hi)
    chosen = rng.sample(pool, k)
    return chosen


def generate_step_windows(
    steps: list[str],
    delay_min_ms: int,
    delay_max_ms: int,
    rng: Optional[random.Random] = None,
) -> list[StepWindowDict]:
    """Фаза 2 (§5.3): для каждого выбранного шага сэмплирует случайное окно
    задержки `[min_delay_ms, max_delay_ms]` ВНУТРИ сконфигурированного
    диапазона (`LIVENESS_STEP_DELAY_MIN_MS`/`_MAX_MS`, app/config.py) —
    т.е. рандомизируется не только САМ факт задержки, но и ширина/границы
    окна на каждый шаг, тем же `secrets`-ГСЧ, что и выбор шагов. Два
    независимых сэмпла из диапазона сортируются, чтобы `min_delay_ms <=
    max_delay_ms` было инвариантом всегда, а не "по счастливой случайности".

    Дефолт ГСЧ — `secrets.SystemRandom()`, тот же принцип, что и в
    `generate_challenge_spec` (см. её docstring про Фазу 0).

    MEDIUM finding (MF DOOM code review, 2026-07-20): `delay_min_ms`/
    `delay_max_ms` are CLAMPED to `(min(...), max(...))` BEFORE being passed
    to `rng.randint` — symmetric to `generate_challenge_spec`'s own `lo, hi =
    min(...), max(...)` clamp of `step_count_min`/`step_count_max` above.
    Without this, a misconfigured `LIVENESS_STEP_DELAY_MIN_MS >
    LIVENESS_STEP_DELAY_MAX_MS` in `app/config.py` (env var typo, swapped
    values) would make `rng.randint(delay_min_ms, delay_max_ms)` raise
    `ValueError: empty range for randrange()` — a 500 on every single
    `/liveness/challenge` call, not a graceful degrade. Clamping here is
    defense-in-depth alongside `app/config.py`'s own `Settings` validation
    (`LIVENESS_STEP_DELAY_MIN_MS`/`_MAX_MS` no longer accept an inverted
    range either) — this function does not trust the config layer alone."""
    rng = rng or secrets.SystemRandom()
    lo_bound, hi_bound = min(delay_min_ms, delay_max_ms), max(delay_min_ms, delay_max_ms)
    windows: list[StepWindowDict] = []
    for step in steps:
        a = rng.randint(lo_bound, hi_bound)
        b = rng.randint(lo_bound, hi_bound)
        lo, hi = (a, b) if a <= b else (b, a)
        windows.append({"step": step, "min_delay_ms": lo, "max_delay_ms": hi})
    return windows


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
        step_windows: Optional[list[dict]] = None,
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
            step_windows=step_windows or [],
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

    _KNOWN_FIELDS = {f.name for f in dataclasses.fields(ChallengeSession)}

    @classmethod
    def _to_session(cls, data: dict) -> ChallengeSession:
        """MEDIUM finding (2PAC code review, 2026-07-20): drop any key not on
        `ChallengeSession` before constructing it, instead of `ChallengeSession
        (**data)` unconditionally. A rolling deploy (new worker writes a
        session with a field an older worker's `ChallengeSession` dataclass
        does not know about yet, e.g. a future additive field) would otherwise
        make the OLD worker's `_to_session()` raise `TypeError: unexpected
        keyword argument` on `get()`/`consume()` for that session — a crash on
        read, not a graceful degrade, purely from version skew across
        instances sharing the same Redis store. Filtering to known fields
        means an old worker simply ignores a field it does not understand yet
        (same "ignore unknown, do not crash" principle Pydantic v1 legacy code
        elsewhere in this repo already follows for forward compatibility)."""
        return ChallengeSession(**{k: v for k, v in data.items() if k in cls._KNOWN_FIELDS})

    def create(
        self,
        steps: list[str],
        ttl_s: float,
        correlation_id: str,
        transaction_type: str,
        transaction_ref: str,
        step_windows: Optional[list[dict]] = None,
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
            step_windows=step_windows or [],
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
