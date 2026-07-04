#!/bin/bash
# Run the anti-spoofing service (HTTP)
# For webcam access from other machines, use run-https.sh instead
cd "$(dirname "$0")"

echo "Starting server on http://0.0.0.0:8090"
echo "Web UI: http://localhost:8090"
echo "Health: http://localhost:8090/health"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port 8090
