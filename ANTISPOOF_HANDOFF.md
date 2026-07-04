# ANTISPOOF — Handoff

**Что это:** сервис проверки лица на спуфинг (живой человек или фото/экран). Работает локально, без интернета, без авторизации.

---

## Деплой (на сервере наставника)

```bash
cd /var/lib/mysql/logs/yoyo
git clone git@github.com:SNurali/antispoof.git
cd antispoof
```

Установка и запуск:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
./run.sh
```

Или одной командой:

```bash
bash deploy_mentor.sh
```

После установки — симлинк для управления:

```bash
sudo ln -sf /var/lib/mysql/logs/yoyo/antispoof/ctl.sh /usr/local/bin/spoof
```

---

## Управление сервисом

| Команда | Действие |
|---|---|
| `spoof start` | Запустить |
| `spoof stop` | Остановить |
| `spoof status` | Статус + health check |
| `spoof restart` | Перезапустить |

Автостарт при загрузке сервера:

```bash
crontab -e
# добавить строку:
@reboot sleep 5 && spoof start
```

---

## API

**URL:** `http://127.0.0.1:8090/spoof-server`  
**Метод:** POST  
**Content-Type:** application/json

### Запрос

```json
{
  "photo": "base64_encoded_image_string"
}
```

### Ответ

```json
{
  "elapsed_time": 0.023,
  "is_spoof": 1
}
```

| Поле | Тип | Описание |
|---|---|---|
| `elapsed_time` | float | Время обработки в секундах |
| `is_spoof` | int | `1` — спуф (фото/экран/нет лица), `0` — живое лицо |

### Пример вызова

```bash
PHOTO_B64=$(base64 -w0 face.jpg)
curl -s -X POST http://127.0.0.1:8090/spoof-server \
  -H "Content-Type: application/json" \
  -d "{\"photo\": \"$PHOTO_B64\"}"
```

---

## Переменные окружения (опционально)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LIVENESS_THRESHOLD` | `0.5` | Порог уверенности (0–1) |
| `PORT` | `8090` | Порт |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

---

## Что внутри

- **RetinaFace** — детекция лица
- **MiniFASNetV1 + MiniFASNetV2** — двойная нейросеть anti-spoof
- **7 сигналов** — FFT, текстура (LBP), цвет (YCbCr), муар, резкость, JPEG, recapture

---

## Порт и безопасность

- Слушает **только** `127.0.0.1:8090` — недоступен из сети
- **Без авторизации** — чистый фильтр
- **Нет исходящих соединений** — полностью локальный

---

## Контакты

- Разработчик: Нурали
- Репозиторий: https://github.com/SNurali/antispoof
