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

IN-MEMORY ONLY in this increment (see app/config.py
::LIVENESS_SESSION_TTL_S docstring) — sessions do not survive a process
restart and are not shared across worker processes. A single-worker deploy
is fine for a smoke/dev rollout; horizontal scaling needs a shared store
(Redis) before this can run behind more than one process, same caveat
FACEID_PHASE1_PAD_GATE.md §2 item 10 already raised for /pad/check.
"""
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
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
