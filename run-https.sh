#!/bin/bash
# Run with HTTPS (required for webcam access from other machines)
cd "$(dirname "$0")"

if [ ! -f certs/cert.pem ]; then
    echo "Generating self-signed SSL certificate..."
    mkdir -p certs
    openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
fi

echo "Starting server on https://0.0.0.0:8090"
echo "From this machine: https://localhost:8090"
echo "From other machines: https://<YOUR_IP>:8090 (accept certificate warning)"
echo ""

exec uvicorn app.main:app \
    --host 0.0.0.0 --port 8090 \
    --ssl-keyfile certs/key.pem \
    --ssl-certfile certs/cert.pem
