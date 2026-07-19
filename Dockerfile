# CPU-only prod image (egaz-02-compatible target: no GPU, Ubuntu 20.04 host,
# Docker 28.x / Compose v2.34.x). Do NOT add onnxruntime-gpu or a CUDA torch
# index here — see requirements.txt comments; this Dockerfile intentionally
# only ever installs the CPU wheels.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    WEB_CONCURRENCY=1

# libgl1/libglib2.0-0/libsm6/libxext6/libxrender-dev — opencv-python runtime
# deps under python-slim (no X server, but opencv still dlopens these).
# libgomp1 — OpenMP runtime some CPU BLAS/onnxruntime builds dlopen at
# import time; installed defensively even though today's default
# (LIVENESS_ENDPOINTS_ENABLED=false) never imports onnxruntime at startup.
# curl — required INSIDE the container for docker-compose.yml's healthcheck
# (`curl -f http://localhost:8090/health`); python-slim ships without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only wheels only. The --index-url flag is mandatory: without it pip
# resolves a multi-GB CUDA build of torch that this CPU-only target
# (Xeon E5-4620 v2, no GPU) never uses. requirements.txt's own `onnxruntime`
# (not `onnxruntime-gpu`) already resolves to a CPU wheel from PyPI.
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/
COPY src/ src/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root. Port 8090 is unprivileged (>1024) so no CAP_NET_BIND_SERVICE
# needed. chown after COPY so the app user can read app/models/src (pip
# packages under /usr/local stay root-owned but world-readable, which is
# the default and is fine — the app process never writes there).
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin antispoof \
    && chown -R antispoof:antispoof /app
USER antispoof

EXPOSE 8090

# Redundant with docker-compose.yml's own healthcheck block (kept here too
# so `docker run` without compose still reports health via `docker ps`).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://127.0.0.1:8090/health || exit 1

# entrypoint.sh hardcodes `--workers 1` on the actual uvicorn exec — this is
# what closes the gap app/main.py's own WEB_CONCURRENCY guard cannot see on
# its own (a --workers N CLI flag forks workers AFTER that guard already ran
# once in the parent process; see the guard's module-level comment in
# app/main.py). Do not replace this with a bare uvicorn CMD without reading
# entrypoint.sh's comments first.
ENTRYPOINT ["/entrypoint.sh"]
