#!/bin/bash
# Run the anti-spoofing service (HTTP)

set -e
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "Starting server on http://127.0.0.1:8090"
echo "Health: http://127.0.0.1:8090/health"
echo ""

exec uvicorn app.main:app --host 127.0.0.1 --port 8090
