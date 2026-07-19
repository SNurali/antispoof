"""Tests for app/liveness_session.py — challenge generation + session store."""
import random
import secrets
import time

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.liveness_session import SessionStore, generate_challenge_spec, generate_step_windows


class TestGenerateChallengeSpecDefaultRng:
    """Фаза 0 (CHALLENGE_ENTROPY_SPRINT_v1.md §3): продовый дефолтный ГСЧ
    должен быть secrets.SystemRandom(), не модуль `random`."""

    def test_default_rng_is_secrets_system_random(self):
        """Calling without an explicit rng must use secrets.SystemRandom
        under the hood — verified by rebinding the `secrets` NAME inside
        app.liveness_session's own module namespace (not the real global
        secrets module) to a spy, so nothing outside this test is affected."""
        import inspect
        import types

        import app.liveness_session as ls

        sig = inspect.signature(generate_challenge_spec)
        assert sig.parameters["rng"].default is None

        calls = []

        class _SpySystemRandom(secrets.SystemRandom):
            def __init__(self):
                calls.append(1)
                super().__init__()

        original = ls.secrets
        ls.secrets = types.SimpleNamespace(SystemRandom=_SpySystemRandom)
        try:
            steps = generate_challenge_spec(["TURN_LEFT", "TURN_RIGHT"], 2, 2)
        finally:
            ls.secrets = original
        assert calls == [1]
        assert set(steps) == {"TURN_LEFT", "TURN_RIGHT"}


class TestGenerateChallengeSpec:
    def test_returns_subset_of_pool(self):
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        steps = generate_challenge_spec(pool, 2, 2, rng=random.Random(42))
        assert set(steps).issubset(set(pool))
        assert len(steps) == 2

    def test_step_count_capped_at_pool_size(self):
        """Пул меньше step_count_min/_max — диапазон честно клампится к
        len(pool), а не падает (CHALLENGE_ENTROPY_SPRINT_v1.md §5.1)."""
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        steps = generate_challenge_spec(pool, 3, 10, rng=random.Random(1))
        assert len(steps) == len(pool)

    def test_randomization_varies_order(self):
        """With a 2-element pool there are only 2 possible orderings — assert
        both are reachable (documents the known low-entropy limitation,
        does not assert a specific distribution)."""
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        seen = set()
        for seed in range(20):
            steps = generate_challenge_spec(pool, 2, 2, rng=random.Random(seed))
            seen.add(tuple(steps))
        assert len(seen) == 2  # both orderings observed across seeds

    def test_pool_of_two_stays_deterministic_with_min3_max4(self):
        """Прод-поведение при пуле=2 (сегодняшний LIVENESS_CHALLENGE_STEPS_POOL)
        должно остаться ТЕМ ЖЕ, что и раньше (всегда оба шага), даже когда
        LIVENESS_CHALLENGE_STEP_COUNT_MIN/_MAX=3/4 — клампинг делает диапазон
        [2,2] детерминированно, не даёт rng.sample упасть."""
        pool = ["TURN_LEFT", "TURN_RIGHT"]
        for seed in range(10):
            steps = generate_challenge_spec(pool, 3, 4, rng=random.Random(seed))
            assert len(steps) == 2
            assert set(steps) == set(pool)

    def test_k_distribution_across_pool_of_four_or_more(self):
        """С пулом >=4 диапазон [3,4] реально работает: оба значения k
        должны быть достижимы, и порядок должен варьироваться (не единичный
        ручной прогон — статистический тест на инжектированном seed-ГСЧ,
        DoD в CHALLENGE_ENTROPY_SPRINT_v1.md §11)."""
        pool = ["TURN_LEFT", "TURN_RIGHT", "NOD_UP", "NOD_DOWN", "BLINK", "SMILE"]
        seen_k = set()
        seen_orderings = set()
        for seed in range(200):
            steps = generate_challenge_spec(pool, 3, 4, rng=random.Random(seed))
            assert len(steps) in (3, 4)
            assert set(steps).issubset(set(pool))
            assert len(set(steps)) == len(steps)  # no duplicates within one spec
            seen_k.add(len(steps))
            seen_orderings.add(tuple(steps))
        assert seen_k == {3, 4}
        assert len(seen_orderings) > 50  # wide spread, not a handful of repeats

    def test_inverted_step_count_range_does_not_raise(self):
        """MEDIUM finding (MF DOOM code review, 2026-07-20): a misconfigured
        LIVENESS_CHALLENGE_STEP_COUNT_MIN > _MAX must not crash
        rng.randint(lo, hi) either — `hi = min(step_count_max, len(pool))`,
        `lo = min(step_count_min, hi)` already guarantees lo <= hi
        regardless of which of MIN/MAX is larger."""
        pool = ["TURN_LEFT", "TURN_RIGHT", "NOD_UP", "NOD_DOWN"]
        for seed in range(20):
            steps = generate_challenge_spec(pool, step_count_min=4, step_count_max=1, rng=random.Random(seed))
            assert set(steps).issubset(set(pool))
            assert len(steps) == len(set(steps))


class TestGenerateStepWindows:
    def test_windows_cover_every_step_in_order(self):
        steps = ["TURN_LEFT", "TURN_RIGHT"]
        windows = generate_step_windows(steps, 400, 1500, rng=random.Random(7))
        assert [w["step"] for w in windows] == steps

    def test_min_always_less_or_equal_max(self):
        steps = ["TURN_LEFT", "TURN_RIGHT", "BLINK", "NOD_UP"]
        for seed in range(50):
            windows = generate_step_windows(steps, 400, 1500, rng=random.Random(seed))
            for w in windows:
                assert 400 <= w["min_delay_ms"] <= w["max_delay_ms"] <= 1500

    def test_default_rng_does_not_raise(self):
        """Дефолтный secrets.SystemRandom() путь — не мокается, просто
        проверяем что вызов без rng не падает и возвращает валидные окна."""
        windows = generate_step_windows(["TURN_LEFT"], 400, 1500)
        assert len(windows) == 1
        assert windows[0]["min_delay_ms"] <= windows[0]["max_delay_ms"]

    def test_inverted_config_range_does_not_raise(self):
        """MEDIUM finding (MF DOOM code review, 2026-07-20): a misconfigured
        LIVENESS_STEP_DELAY_MIN_MS > LIVENESS_STEP_DELAY_MAX_MS (env-var
        typo/swap) must NOT make rng.randint(delay_min_ms, delay_max_ms)
        raise ValueError('empty range for randrange()') — i.e. no 500 on
        /liveness/challenge — it clamps to (min, max) of the two bounds
        first, symmetric to generate_challenge_spec's own lo/hi clamp."""
        for seed in range(20):
            windows = generate_step_windows(
                ["TURN_LEFT", "TURN_RIGHT"], delay_min_ms=1500, delay_max_ms=400, rng=random.Random(seed),
            )
            for w in windows:
                assert 400 <= w["min_delay_ms"] <= w["max_delay_ms"] <= 1500


class TestStepCountAndDelayConfigValidation:
    """MEDIUM finding (MF DOOM code review, 2026-07-20): Field(ge=0) on
    LIVENESS_CHALLENGE_STEP_COUNT_MIN/_MAX and LIVENESS_STEP_DELAY_MIN_MS/
    _MAX_MS (app/config.py) — negative values fail fast at Settings()
    construction; an INVERTED (min > max) range is deliberately still
    accepted here (not a validation error) since generate_challenge_spec/
    generate_step_windows already clamp it, see their own tests above."""

    def test_negative_step_count_min_rejected(self):
        with pytest.raises(ValidationError):
            Settings(SERVICE_TOKEN="", LIVENESS_CHALLENGE_STEP_COUNT_MIN=-1)

    def test_negative_step_delay_min_ms_rejected(self):
        with pytest.raises(ValidationError):
            Settings(SERVICE_TOKEN="", LIVENESS_STEP_DELAY_MIN_MS=-1)

    def test_inverted_range_constructs_without_error(self):
        """Settings() itself does not hard-fail on MIN > MAX — that
        degrades gracefully downstream via the generator functions' own
        clamping, not via rejecting the config outright (see module
        docstring above)."""
        settings = Settings(
            SERVICE_TOKEN="",
            LIVENESS_CHALLENGE_STEP_COUNT_MIN=4,
            LIVENESS_CHALLENGE_STEP_COUNT_MAX=1,
            LIVENESS_STEP_DELAY_MIN_MS=1500,
            LIVENESS_STEP_DELAY_MAX_MS=400,
        )
        assert settings.LIVENESS_CHALLENGE_STEP_COUNT_MIN == 4
        assert settings.LIVENESS_STEP_DELAY_MIN_MS == 1500


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
