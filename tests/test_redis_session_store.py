"""Tests for app/liveness_session.py::RedisSessionStore + build_session_store.

Uses fakeredis (in-process fake Redis server, incl. Lua/EVAL support via the
`fakeredis[lua]` extra — see requirements-dev.txt) so the REAL
RedisSessionStore code — including the atomic consume() Lua script — is
exercised without needing a real `redis-server` process for CI/dev. Mirrors
tests/test_liveness_session.py's SessionStore scenarios 1:1 so both backends
are proven to honor the exact same public contract, plus a couple of
Redis-specific cases (native TTL expiry, cross-"process" atomicity, backend
selection/fail-fast).
"""
import threading
import time

import fakeredis
import pytest

from app.liveness_session import RedisSessionStore, SessionStore, build_session_store


@pytest.fixture()
def redis_client():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture()
def store(redis_client):
    return RedisSessionStore(redis_client)


class TestRedisSessionStoreParityWithSessionStore:
    """Same scenarios as tests/test_liveness_session.py::TestSessionStore —
    both backends must behave identically from the caller's point of view."""

    def test_create_returns_session_with_id(self, store):
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        assert session.session_id
        assert session.steps == ["TURN_LEFT"]
        assert session.used is False

    def test_consume_marks_used_and_second_consume_fails(self, store):
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        got, err = store.consume(session.session_id)
        assert err is None
        assert got.session_id == session.session_id

        got2, err2 = store.consume(session.session_id)
        assert got2 is None
        assert err2 == "SESSION_ALREADY_USED"

    def test_consume_unknown_session_id(self, store):
        got, err = store.consume("does-not-exist")
        assert got is None
        assert err == "SESSION_NOT_FOUND"

    def test_consume_expired_session(self, store):
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=0.01, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        time.sleep(0.02)
        got, err = store.consume(session.session_id)
        assert got is None
        assert err == "SESSION_EXPIRED"

    def test_get_does_not_consume(self, store):
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        fetched = store.get(session.session_id)
        assert fetched is not None
        assert fetched.used is False
        got, err = store.consume(session.session_id)
        assert err is None


class TestRedisSessionStoreSpecifics:
    def test_get_unknown_session_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_key_reaped_after_storage_ttl_reads_as_not_found(self, store, redis_client):
        """Past the grace window (2x ttl_s + 60s) Redis itself has expired
        the key — indistinguishable from SESSION_NOT_FOUND, same as the
        in-memory store's _sweep_expired_locked() eventually deleting a
        stale entry outright."""
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=1.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        key = store._key(session.session_id)
        assert redis_client.ttl(key) > 0
        redis_client.delete(key)  # simulate storage TTL having elapsed
        got, err = store.consume(session.session_id)
        assert got is None
        assert err == "SESSION_NOT_FOUND"

    def test_concurrent_consume_only_one_winner(self, store):
        """The scenario threading.Lock protected in-process for
        SessionStore, proven here across two THREADS sharing one
        RedisSessionStore/client — the atomicity comes from the Lua
        script running as a single Redis command, not from any
        client-side lock (there isn't one), which is exactly what makes
        it also safe across separate worker PROCESSES."""
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        results = []
        barrier = threading.Barrier(2)

        def _consume():
            barrier.wait()
            results.append(store.consume(session.session_id))

        threads = [threading.Thread(target=_consume) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [r for r in results if r[1] is None]
        fails = [r for r in results if r[1] == "SESSION_ALREADY_USED"]
        assert len(oks) == 1
        assert len(fails) == 1

    def test_session_survives_process_restart_simulation(self, redis_client):
        """A second RedisSessionStore instance (standing in for a second
        worker process sharing the same Redis) can see a session created
        by the first — the actual bug the Redis backend exists to fix."""
        store_a = RedisSessionStore(redis_client)
        store_b = RedisSessionStore(redis_client)
        session = store_a.create(
            steps=["TURN_RIGHT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        got, err = store_b.consume(session.session_id)
        assert err is None
        assert got.session_id == session.session_id


class _FakeSettings:
    def __init__(self, backend="memory", redis_url="redis://localhost:6379/0"):
        self.SESSION_STORE_BACKEND = backend
        self.REDIS_URL = redis_url


class TestBuildSessionStore:
    def test_memory_backend_returns_session_store(self):
        assert isinstance(build_session_store(_FakeSettings(backend="memory")), SessionStore)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            build_session_store(_FakeSettings(backend="bogus"))

    def test_redis_backend_unreachable_raises_no_silent_fallback(self):
        """Point at a port nothing is listening on — must raise, must NOT
        silently return an in-memory SessionStore."""
        settings = _FakeSettings(backend="redis", redis_url="redis://127.0.0.1:1/0")
        with pytest.raises(RuntimeError, match="unreachable"):
            build_session_store(settings)
