#!/bin/bash
# Anti-Spoofing Liveness Service — control script
# Usage: spoof {start|stop|status|restart}
#
# Install: ln -sf $(pwd)/ctl.sh /usr/local/bin/spoof

set -e

# Resolve real script location (follows symlinks)
SCRIPT_PATH="$(readlink -f "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_PATH")"
PIDFILE="$PROJECT_DIR/.pid"
LOGFILE="$PROJECT_DIR/server.log"

cd "$PROJECT_DIR"

_spoof_start() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "SPOOF already running (PID $(cat $PIDFILE))"
        return 1
    fi

    # Clean stale PID file
    rm -f "$PIDFILE" 2>/dev/null

    # Kill zombie on port if any
    fuser -k 8090/tcp &>/dev/null || true
    sleep 1

    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
    fi

    nohup uvicorn app.main:app --host 0.0.0.0 --port 8090 >> "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 3

    if kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        echo "SPOOF STARTED — PID $(cat $PIDFILE) — http://0.0.0.0:8090/spoof-server"
    else
        echo "SPOOF FAILED TO START — check $LOGFILE"
        tail -5 "$LOGFILE"
        rm -f "$PIDFILE"
        return 1
    fi
}

_spoof_stop() {
    local stopped=0

    # Try PID file first
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        local pid=$(cat "$PIDFILE")
        kill "$pid"
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        echo "SPOOF STOPPED — PID $pid"
        stopped=1
    fi

    # Fallback: kill anything on port 8090
    if fuser 8090/tcp &>/dev/null; then
        fuser -k 8090/tcp &>/dev/null || true
        echo "SPOOF STOPPED — killed process on port 8090"
        stopped=1
    fi

    rm -f "$PIDFILE"

    if [ "$stopped" -eq 0 ]; then
        echo "SPOOF not running"
    fi
}

_spoof_status() {
    if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
        local pid=$(cat "$PIDFILE")
        echo "SPOOF RUNNING — PID $pid — http://127.0.0.1:8090"
        curl -s http://127.0.0.1:8090/health 2>/dev/null || echo "  (health check failed)"
    else
        echo "SPOOF STOPPED"
    fi
}

case "${1:-}" in
    start)   _spoof_start ;;
    stop)    _spoof_stop ;;
    status)  _spoof_status ;;
    restart)
        set +e
        _spoof_stop
        sleep 2
        # Ensure port is free
        fuser -k 8090/tcp &>/dev/null
        sleep 1
        _spoof_start
        ;;
    *)
        echo "Usage: spoof {start|stop|status|restart}"
        exit 1
        ;;
esac
