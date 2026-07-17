"""Tests for app/liveness_session.py — challenge generation + session store."""
import random
import time

from app.liveness_session import SessionStore, generate_challenge_spec


class TestGenerateChallengeSpec:
    def test_returns_subset_of_pool(self):
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        steps = generate_challenge_spec(pool, 2, rng=random.Random(42))
        assert set(steps).issubset(set(pool))
        assert len(steps) == 2

    def test_step_count_capped_at_pool_size(self):
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        steps = generate_challenge_spec(pool, 10, rng=random.Random(1))
        assert len(steps) == len(pool)

    def test_randomization_varies_order(self):
        """With a 2-element pool there are only 2 possible orderings — assert
        both are reachable (documents the known low-entropy limitation,
        does not assert a specific distribution)."""
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        seen = set()
        for seed in range(20):
            steps = generate_challenge_spec(pool, 2, rng=random.Random(seed))
            seen.add(tuple(steps))
        assert len(seen) == 2  # both orderings observed across seeds


class TestSessionStore:
    def test_create_returns_session_with_id(self):
        store = SessionStore()
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        assert session.session_id
        assert session.steps == ["TURN_LEFT"]
        assert session.used is False

    def test_consume_marks_used_and_second_consume_fails(self):
        store = SessionStore()
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

    def test_consume_unknown_session_id(self):
        store = SessionStore()
        got, err = store.consume("does-not-exist")
        assert got is None
        assert err == "SESSION_NOT_FOUND"

    def test_consume_expired_session(self):
        store = SessionStore()
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=0.01, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        time.sleep(0.02)
        got, err = store.consume(session.session_id)
        assert got is None
        assert err == "SESSION_EXPIRED"

    def test_get_does_not_consume(self):
        store = SessionStore()
        session = store.create(
            steps=["TURN_LEFT"], ttl_s=60.0, correlation_id="c1",
            transaction_type="sale", transaction_ref="req:bal",
        )
        fetched = store.get(session.session_id)
        assert fetched is not None
        assert fetched.used is False
        # still consumable after a plain get()
        got, err = store.consume(session.session_id)
        assert err is None
