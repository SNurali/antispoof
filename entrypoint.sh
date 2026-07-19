#!/bin/sh
# Locks the container to exactly ONE uvicorn worker process, no matter what
# WEB_CONCURRENCY value ends up in the container's environment (.env, a
# `docker compose run -e WEB_CONCURRENCY=4 ...`, etc).
#
# WHY THIS EXISTS (2PAC review, 2026-07-18, see app/main.py's own
# WEB_CONCURRENCY startup guard for the full writeup): that guard can only
# see the WEB_CONCURRENCY *env var* — it cannot see a `--workers N` CLI flag,
# because uvicorn's --workers forks additional worker processes AFTER the
# guard has already run once in the parent process. A deploy that quietly
# switched to `uvicorn ... --workers 4` would run 4 independent in-memory
# SessionStores (app/liveness_session.py) with no error and no warning —
# silent SESSION_NOT_FOUND failures on whichever worker didn't mint a given
# challenge session. Hardcoding --workers 1 on the actual exec line below
# closes that gap at the process-launch level instead of relying on env var
# hygiene alone.
#
# Before ever raising this above 1 worker: SESSION_STORE_BACKEND must be
# "redis" (not "memory") in .env — app/main.py's own guard will otherwise
# refuse to start. See docker-compose.yml's redis service (antispoof-redis).
set -eu

export WEB_CONCURRENCY=1

exec python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8090 \
    --workers 1
