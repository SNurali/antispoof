#!/bin/bash
# Установка antispoof как systemd-демон
# Запустить от root: sudo bash install-daemon.sh

set -e

SERVICE_NAME="antispoof"
INSTALL_DIR="/var/lib/mysql/logs/yoyo/antispoof"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== Установка Antispoof как демон ==="
echo ""

# Проверка что мы в правильной папке
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
    echo "⚠ Скрипт запущен из $SCRIPT_DIR"
    echo "  Ожидается: $INSTALL_DIR"
    echo "  Копирую файлы..."
    mkdir -p "$INSTALL_DIR"
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/" 2>/dev/null || true
fi

cd "$INSTALL_DIR"

# Проверка venv
if [ ! -f ".venv/bin/python" ]; then
    echo "✗ Не найден .venv. Создаю..."
    python3.11 -m venv .venv
    source .venv/bin/activate
    echo "  Устанавливаю torch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    echo "  Устанавливаю зависимости..."
    pip install -r requirements.txt -q
    echo "✓ Зависимости установлены"
fi

# Останавливаем если уже запущен
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo "  Останавливаю текущий сервис..."
    systemctl stop "$SERVICE_NAME"
fi

# Создаём сервис
echo "  Создаю systemd-сервис..."
cat > "$SERVICE_FILE" << 'SERVICEEOF'
[Unit]
Description=Antispoof Face Liveness Detection Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/lib/mysql/logs/yoyo/antispoof
ExecStart=/var/lib/mysql/logs/yoyo/antispoof/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8090
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Включаем и запускаем
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

echo ""
echo "✓ Сервис установлен и запущен!"
echo ""

# Ждём пока поднимется
sleep 3

# Проверяем
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "=== Статус ==="
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    echo "=== Проверка health ==="
    curl -s http://localhost:8090/health 2>/dev/null && echo "" || echo "⚠ Сервис ещё запускается, подождите пару секунд"
else
    echo "✗ Сервис не запустился. Логи:"
    journalctl -u "$SERVICE_NAME" --no-pager -n 20
    exit 1
fi

echo ""
echo "=== Готово! ==="
echo "Команды:"
echo "  sudo systemctl status antispoof     # статус"
echo "  sudo systemctl restart antispoof    # перезапуск"
echo "  sudo systemctl stop antispoof       # остановить"
echo "  sudo journalctl -u antispoof -f     # логи"
