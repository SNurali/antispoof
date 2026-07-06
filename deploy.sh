#!/bin/bash
# deploy.sh — idempotent CPU deploy for the E-GAZ anti-spoofing (liveness) microservice.
#
# Usage:
#   ./deploy.sh                    # setup venv+deps+cert, run in foreground (HTTPS)
#   ./deploy.sh --nohup            # setup, then run detached with nohup (survives shell exit)
#   ./deploy.sh --install-service  # setup, then install+start a systemd --user service
#   MODE=http ./deploy.sh          # plain HTTP instead of HTTPS (no camera dashboard needed)
#
# Env overrides (all optional, sane CPU-prod defaults):
#   PORT=8090 HOST=127.0.0.1 DEVICE=cpu MODE=https LIVENESS_THRESHOLD=0.5 MAX_BATCH=16
#   SERVICE_TOKEN=<secret> RATE_LIMIT_BURST=20 RATE_LIMIT_SUSTAINED=5.0
#
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# ---------- config (env overridable) ----------
# Loopback-only by default (0.0.0.0 bypasses ufw/firewall rules). Override HOST to a
# private VPC IP only if Laravel calls this service from a different host on a closed network.
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8090}"
export DEVICE="${DEVICE:-cpu}"                 # prod is CPU-only
export LIVENESS_THRESHOLD="${LIVENESS_THRESHOLD:-0.5}"
export MAX_BATCH="${MAX_BATCH:-16}"
export SERVICE_TOKEN="${SERVICE_TOKEN:-}"        # shared-secret with Laravel; empty = dev mode (no auth)
export RATE_LIMIT_BURST="${RATE_LIMIT_BURST:-20}"
export RATE_LIMIT_SUSTAINED="${RATE_LIMIT_SUSTAINED:-5.0}"
MODE="${MODE:-https}"                          # https (dashboard/camera) | http

VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { echo "[deploy] $*"; }

# ---------- 1. venv ----------
if [ ! -d "$VENV_DIR" ]; then
    log "Creating venv at ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    log "venv already exists, reusing ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------- 2. deps (CPU-only torch, no CUDA bloat) ----------
log "Upgrading pip"
pip install --quiet --upgrade pip

log "Installing CPU-only torch/torchvision (index: download.pytorch.org/whl/cpu)"
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu

log "Installing requirements.txt"
pip install --quiet -r requirements.txt

# ---------- 3. self-signed TLS cert (idempotent) ----------
mkdir -p certs
if [ "$MODE" = "https" ] && [ ! -f certs/cert.pem ]; then
    log "Generating self-signed TLS certificate in certs/"
    openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
    chmod 600 certs/key.pem
else
    log "TLS cert present or HTTP mode selected, skipping cert generation"
fi

# ---------- uvicorn command builder ----------
build_uvicorn_args() {
    local args=(-m uvicorn app.main:app --host "$HOST" --port "$PORT")
    if [ "$MODE" = "https" ]; then
        args+=(--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem)
    fi
    echo "${args[@]}"
}

# ---------- health-check helper ----------
health_check() {
    local scheme="http"
    [ "$MODE" = "https" ] && scheme="https"
    log "Waiting for service on ${scheme}://${HOST}:${PORT}/health ..."
    local tries=30
    while [ $tries -gt 0 ]; do
        if curl --silent --fail --max-time 2 -k "${scheme}://127.0.0.1:${PORT}/health" >/tmp/antispoof_health.json 2>/dev/null; then
            log "Health check OK:"
            cat /tmp/antispoof_health.json
            echo
            return 0
        fi
        sleep 1
        tries=$((tries - 1))
    done
    log "Health check FAILED after 30s — check logs."
    return 1
}

# ---------- 4. run mode ----------
MODE_FLAG="${1:-}"

if [ "$MODE_FLAG" = "--install-service" ]; then
    : "${SERVICE_TOKEN:?SERVICE_TOKEN is required for a persistent/production deploy (systemd service). Set SERVICE_TOKEN=<secret>, e.g. \$(openssl rand -hex 32).}"

    log "Installing systemd --user service (antispoof.service)"
    SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
    mkdir -p "$SYSTEMD_USER_DIR"

    UVICORN_ARGS=(-m uvicorn app.main:app --host "$HOST" --port "$PORT")
    if [ "$MODE" = "https" ]; then
        UVICORN_ARGS+=(--ssl-keyfile "${SCRIPT_DIR}/certs/key.pem" --ssl-certfile "${SCRIPT_DIR}/certs/cert.pem")
    fi

    cat > "${SYSTEMD_USER_DIR}/antispoof.service" <<EOF
[Unit]
Description=E-GAZ Anti-Spoofing (Liveness) Microservice
After=network.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
Environment=DEVICE=${DEVICE}
Environment=HOST=${HOST}
Environment=PORT=${PORT}
Environment=LIVENESS_THRESHOLD=${LIVENESS_THRESHOLD}
Environment=MAX_BATCH=${MAX_BATCH}
Environment=SERVICE_TOKEN=${SERVICE_TOKEN}
Environment=RATE_LIMIT_BURST=${RATE_LIMIT_BURST}
Environment=RATE_LIMIT_SUSTAINED=${RATE_LIMIT_SUSTAINED}
ExecStart=${VENV_DIR}/bin/python ${UVICORN_ARGS[@]}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload
    systemctl --user enable --now antispoof.service
    log "Service installed. Enabling linger recommended so it survives logout:"
    log "  sudo loginctl enable-linger \$(whoami)"
    log "Status: systemctl --user status antispoof.service"
    log "Logs:   journalctl --user -u antispoof.service -f"

    sleep 2
    health_check
    exit $?
fi

if [ "$MODE_FLAG" = "--nohup" ]; then
    log "Starting detached (nohup) on ${MODE}://${HOST}:${PORT}"
    # shellcheck disable=SC2046
    nohup "${VENV_DIR}/bin/python" $(build_uvicorn_args) \
        > antispoof.out.log 2>&1 &
    echo $! > antispoof.pid
    log "PID $(cat antispoof.pid), logs: antispoof.out.log"
    health_check
    exit $?
fi

# default: foreground
log "Starting in foreground on ${MODE}://${HOST}:${PORT} (Ctrl+C to stop)"
log "Tip: use './deploy.sh --nohup' or './deploy.sh --install-service' for persistent deploy"
# shellcheck disable=SC2046
exec "${VENV_DIR}/bin/python" $(build_uvicorn_args)
