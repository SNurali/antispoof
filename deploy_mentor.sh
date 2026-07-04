#!/bin/bash
# Деплой antispoof на сервере наставника
# Директория: /var/lib/mysql/logs/yoyo/antispoof

set -e

cd /var/lib/mysql/logs/yoyo/antispoof

echo "=== Создаю venv ==="
python3.11 -m venv .venv
source .venv/bin/activate

echo "=== Ставлю PyTorch ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Ставлю зависимости ==="
pip install -r requirements.txt

echo "=== Запускаю сервер ==="
./run.sh
