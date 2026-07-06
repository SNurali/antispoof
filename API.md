# Antispoof API — Документация

Face liveness detection (anti-spoofing) сервис. Определяет живое ли лицо на фото или это спуф (фото/экран/принт).

**Стек:** FastAPI + PyTorch + RetinaFace + MiniFASNet  
**Порт:** `8090`  
**Режим:** Локальный сервер (не публикуется в интернет)

---

## Быстрый старт

```bash
cd /home/mrnurali/E-GAZ/antispoof

# CPU
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090

# GPU (RTX 3080)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090

# Docker
docker build -t antispoof .
docker run -p 8090:8090 antispoof
```

---

## Эндпоинты

### GET /health

Проверка состояния сервиса.

```bash
curl -s http://localhost:8090/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3080",
  "models_loaded": true,
  "model_version": "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1"
}
```

**HTTP код:** 200 если модели загружены, 503 если модели не готовы.

---

## Phase 1 Production Contract

### POST /pad/check (Phase 1 PAD-gate)

**Основной эндпоинт** для интеграции с Laravel (see FACEID_PHASE1_PAD_GATE.md).  
Проверяет живое ли лицо и возвращает вердикт для компенсирующего контроля на продажах.

**Аутентификация:**
- Требуется заголовок `X-Service-Token` с shared-secret (если `SERVICE_TOKEN` установлена в env)
- IP-allowlist: 127.0.0.0/8, 192.168.0.0/24, 10.0.0.0/8, 172.16.0.0/12
- Rate limit: 20 req/1s burst, 5 req/s sustained (окно 60s) per IP

**Запрос:**
```bash
curl -X POST http://localhost:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: your-secret-token" \
  -d '{
    "correlation_id": "abc123-uuid",
    "transaction_type": "sale",
    "transaction_ref": "id_request:id_ballon",
    "face_photo": "base64-encoded-jpeg-or-png"
  }'
```

**Параметры запроса (JSON):**
| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `correlation_id` | string (UUID) | да | Для трассировки логов (генерирует Laravel) |
| `transaction_type` | string | да | Только `"sale"` в v1 (может расширяться) |
| `transaction_ref` | string | да | `id_request:id_ballon` — натуральный ключ транзакции |
| `face_photo` | string (base64) | да | Base64 JPEG/PNG, тот же кадр, что Adliya |

**Ответ (200 OK):**
```json
{
  "verdict": "live",
  "reason": null,
  "score": 0.87,
  "threshold": 0.5,
  "face_detected": true,
  "signals": {
    "nn_label": "real",
    "nn_score": 0.98,
    "signal_scores": {
      "recapture": 0.02,
      "fft": 0.1,
      "lbp": 0.05,
      "color": 0.03,
      "moire": 0.01,
      "sharpness": 0.4,
      "jpeg": 0.1
    },
    "spoof_probability": 0.015
  },
  "save_frame": false,
  "model_version": "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1",
  "processing_ms": 18.4
}
```

**Поля ответа:**
| Поле | Тип | Описание |
|---|---|---|
| `verdict` | string | `"live"`, `"spoof"` или `"low_quality"` |
| `reason` | string\|null | `null`, `"PASSIVE_PAD_SPOOF"`, `"NO_FACE"`, `"LOW_QUALITY"`, `"TIMEOUT"`, `"INTERNAL_ERROR"` |
| `score` | float | Уверенность 0.0–1.0 |
| `threshold` | float | Порог (по умолчанию 0.5) |
| `face_detected` | bool | Найдено ли лицо на изображении |
| `signals` | object | Детальная разбивка по нейросети и 7 сигналам |
| `save_frame` | bool | Сервис уже решил сохранить кадр (для следствия) |
| `model_version` | string | Версия модели/ансамбля |
| `processing_ms` | float | Время обработки в миллисекундах |

**Логика вердиктов:**
- `verdict="live"` → `reason=null` → Laravel продолжает `/sales` как обычно
- `verdict="spoof"` → `reason="PASSIVE_PAD_SPOOF"` → Laravel отклоняет продажу (атака обнаружена)
- `verdict="low_quality"` → `reason="NO_FACE"|"LOW_QUALITY"` → UX-реджект "переснимите" (не безопасность, переснять)
- `reason="TIMEOUT"|"INTERNAL_ERROR"` → Сервис недоступен; политика Phase 1 — **fail-open** (решение владельца 2026-07-05): продажа продолжается, Laravel пишет `security_events` + метрику недоступности. Пересмотр на Phase 2.

**Ошибки:**
| HTTP код | Описание |
|----------|----------|
| 200 | OK (всегда возвращает вердикт, даже при `low_quality`) |
| 400 | Invalid base64 / Image size limit / Image dimensions |
| 401 | Missing or invalid X-Service-Token |
| 403 | IP not allowed |
| 422 | `transaction_type` не "sale" |
| 429 | Rate limit exceeded |
| 500 | Internal server error (возвращает контракт с `verdict="low_quality", reason="INTERNAL_ERROR"`) |

---

## Legacy Endpoints (Backward Compatible)

Старые эндпоинты остаются для совместимости с существующим кодом, но **НЕ рекомендуются для новых интеграций**.

### POST /verify

Проверка одного изображения на лайвнесс.

```bash
curl -s -F "image=@face.jpg" http://localhost:8090/verify
```

**Параметры:**
| Поле | Тип | Описание |
|---|---|---|
| `image` | multipart file | Изображение лица (JPEG/PNG) |

**Ответ:**
```json
{
  "is_real": true,
  "label": "live",
  "score": 0.87,
  "threshold": 0.5,
  "face_detected": true,
  "processing_ms": 42.3,
  "signals": {
    "signal_scores": {
      "recapture": 0.12,
      "fft": 0.05,
      "lbp": 0.15,
      "color": 0.10,
      "moire": 0.0,
      "sharpness": 0.05,
      "jpeg": 0.0
    },
    "spoof_probability": 0.08,
    "signals_triggered": []
  }
}
```

**Поля ответа:**
| Поле | Тип | Описание |
|---|---|---|
| `is_real` | bool | `true` если лицо живое и score >= threshold |
| `label` | string | `"live"`, `"spoof"` или `"no_face"` |
| `score` | float | Уверенность 0.0–1.0 |
| `threshold` | float | Порог (по умолчанию 0.5) |
| `face_detected` | bool | Найдено ли лицо на изображении |
| `processing_ms` | float | Время обработки в миллисекундах |
| `signals` | object | Детальная разбивка по сигналам |

---

### POST /verify_batch

Пакетная проверка (до 16 изображений).

```bash
curl -s -F "images=@img1.jpg" -F "images=@img2.jpg" http://localhost:8090/verify_batch
```

**Ответ:**
```json
{
  "results": [
    {"is_real": true, "label": "live", "score": 0.87, "face_detected": true},
    {"is_real": false, "label": "spoof", "score": 0.92, "face_detected": true}
  ],
  "total_ms": 78.5,
  "count": 2
}
```

---

### POST /spoof-server

Проверка liveness через base64 JSON. Формат совместим с внешним сервером.

```bash
# Подготовить base64 и отправить
PHOTO_B64=$(base64 -w0 face.jpg)
curl -s -X POST http://localhost:8090/spoof-server \
  -H "Content-Type: application/json" \
  -d "{\"photo\": \"$PHOTO_B64\"}"
```

**Вход (JSON):**
```json
{
  "photo": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAg..."  
}
```

| Поле | Тип | Описание |
|---|---|---|
| `photo` | string | Изображение в формате base64 |

**Выход (JSON):**
```json
{
  "elapsed_time": 0.023,
  "is_spoof": 0
}
```

| Поле | Тип | Описание |
|---|---|---|
| `elapsed_time` | float | Время обработки в секундах |
| `is_spoof` | int | `0` — живое лицо, `1` — спуф или нет лица |

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LIVENESS_THRESHOLD` | `0.5` | Порог для вердикта LIVE |
| `HOST` | `127.0.0.1` | Хост сервера (по умолчанию только локальный) |
| `PORT` | `8090` | Порт сервера |
| `MODEL_DIR` | `./models` | Путь к весам моделей |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `MAX_BATCH` | `16` | Максимум изображений в batch |
| `SERVICE_TOKEN` | `` (empty) | Shared-secret для X-Service-Token (обязателен для systemd/Docker) |
| `RATE_LIMIT_BURST` | `20` | Макс. запросов в 1 секунду (burst) |
| `RATE_LIMIT_SUSTAINED` | `5` | Макс. запросов в секунду (среднее за 60s) |
| `SAVE_FRAME_VERDICTS` | `spoof,low_quality` | Вердикты, при которых сохранять кадр |
| `AUDIT_LOG_STDOUT` | `` | Если `1`/`true` — писать audit логи в stdout (вместо файла) |

---

## Интеграция с другими сервисами

Сервис работает по HTTP. Другие сервисы подключаются через `http://<IP>:8090`.

### Python (httpx) — /pad/check (рекомендуется)

```python
import httpx
import json

ANTISPOOF_URL = "http://localhost:8090"
SERVICE_TOKEN = "your-secret-token"

async def check_pad_gate(correlation_id: str, transaction_ref: str, face_photo_base64: str) -> dict:
    """Вызов /pad/check для PAD-gate контроля."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ANTISPOOF_URL}/pad/check",
            json={
                "correlation_id": correlation_id,
                "transaction_type": "sale",
                "transaction_ref": transaction_ref,
                "face_photo": face_photo_base64,
            },
            headers={"X-Service-Token": SERVICE_TOKEN},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()

# Использование:
import base64
with open("face.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    
result = await check_pad_gate("uuid-123", "req:bal", b64)
if result["verdict"] == "live":
    print("Лицо живое, продолжить продажу")
elif result["verdict"] == "spoof":
    print(f"Спуф! Атака обнаружена: {result['reason']}")
else:  # low_quality
    print(f"Переснимите: {result['reason']}")
```

### Python (requests) — /verify (legacy)

```python
import requests

def check_liveness(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://localhost:8090/verify",
            files={"image": f},
            timeout=5,
        )
    return resp.json()
```

### cURL

```bash
# PAD-gate (рекомендуется)
PHOTO_B64=$(base64 -w0 face.jpg)
curl -s -X POST http://localhost:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: your-secret" \
  -d "{
    \"correlation_id\": \"uuid-123\",
    \"transaction_type\": \"sale\",
    \"transaction_ref\": \"req:bal\",
    \"face_photo\": \"$PHOTO_B64\"
  }" | python3 -m json.tool

# Verify (legacy)
curl -s -F "image=@photo.jpg" http://localhost:8090/verify

# Health check
curl -s http://localhost:8090/health
```

---

## Docker Compose — объединение сервисов

```yaml
version: "3.8"

services:
  antispoof:
    build: /home/mrnurali/E-GAZ/antispoof
    ports:
      - "127.0.0.1:8090:8090"
    environment:
      - DEVICE=auto
      - LIVENESS_THRESHOLD=0.5
      - SERVICE_TOKEN=your-secret-token-here
      - HOST=127.0.0.1
    restart: unless-stopped

  tracker:
    build: /home/mrnurali/E-GAZ/tracker
    environment:
      ANTISPOOF_URL: http://antispoof:8090
    ports:
      - "8000:8000"
    depends_on:
      - antispoof
    restart: unless-stopped
```

Внутри Docker-сети сервисы видят друг друга по имени контейнера:
```
Laravel  →  http://antispoof:8090/pad/check
```

---

## Архитектура обработки

```
Камера / Изображение
        │
        ▼
┌───────────────────┐
│  RetinaFace        │  ← Детекция лица (bbox)
│  face_detect.py    │
└────────┬──────────┘
         │ bbox [x, y, w, h]
         ▼
┌───────────────────────────────────┐
│  LivenessEngine                    │
│  liveness.py                       │
│                                    │
│  ┌─────────────┐ ┌──────────────┐ │
│  │MiniFASNetV2 │ │MiniFASNetV1SE│ │  ← Двойная нейросеть
│  └──────┬──────┘ └──────┬───────┘ │
│         │               │         │
│         └───────┬───────┘         │
│                 │ softmax fusion   │
│                 ▼                  │
│  ┌─────────────────────────────┐  │
│  │  multisignal.py              │  │  ← 7 эвристик:
│  │  FFT + LBP + Color + Moire  │  │    частоты, текстура,
│  │  + Sharpness + JPEG          │  │    цвет, муар, резкость,
│  │  + Recapture (доминант)      │  │    JPEG, recapture
│  └──────────────┬──────────────┘  │
│                 │                  │
│                 ▼                  │
│         _fuse() → финальный verdict│
└────────────────┬──────────────────┘
                 │
                 ▼
    {verdict, reason, score, ...}
```

---

## Сигналы (multisignal.py)

| Сигнал | Вес | Что ловит |
|---|---|---|
| `recapture` | 45% | Низкодетальные пересъёмы (доминирующий сигнал) |
| `lbp` | 15% | Плоская текстура принтов/экранов |
| `color` | 10% | Узкий цветовой гамут экранов в YCbCr |
| `moire` | 10% | Муар от пиксельной сетки экрана |
| `jpeg` | 10% | Двойные JPEG-артефакты |
| `fft` | 5% | Периодические паттерны в частотной области |
| `sharpness` | 5% | Однородная резкость (нет глубины резкости) |

---

## Безопасность

- По умолчанию сервис слушает на `127.0.0.1:8090` (только локальный хост)
- Для сетевого доступа из Docker/другой машины в той же LAN: `HOST=192.168.0.1` + firewall
- Трафик **без TLS** (HTTP) — допустимо для локальной сети
- Для HTTPS: используйте nginx reverse proxy или самоподписанный сертификат
- **Аутентификация:** обязательный `X-Service-Token` (shared-secret из env)
- **IP-allowlist:** только разрешённые подсети могут вызвать сервис (иначе 403)
- **Rate limit:** 20 burst / 5 sustained per IP (защита от DoS/retry-шторма)
- **2-сек timeout:** на инференс (защита от зависания процесса)
- **Пассивная PAD:** one-frame проверка НЕ защищает от:
  - Прямой загрузки файла (минуя камеру) — закрывается на уровне мобильного клиента (camera-only capture)
  - Видео-реплей и дипфейков (требуют Phase 2 с challenge-response и temporal анализом)

---

## Скорость

| Устройство | Одиночный запрос | Batch (16 фото) |
|---|---|---|
| RTX 3080 | ~18–42 ms | ~200 ms |
| CPU | ~300–500 ms | ~2–4 сек |

---

## Файловая структура

```
antispoof/
├── app/
│   ├── main.py            # FastAPI роуты (health, verify, verify_batch, pad/check)
│   ├── liveness.py         # Загрузка моделей + инференс + fusion
│   ├── face_detect.py      # RetinaFace детекция лиц
│   ├── multisignal.py      # 7 эвристических сигналов anti-spoof
│   ├── config.py           # Pydantic Settings (env vars)
│   └── static/             # Тестовый веб-UI
├── src/
│   └── model_lib/          # Архитектуры MiniFASNet (из Silent-Face repo)
├── models/                 # Веса моделей (.pth + detection_model/)
├── scripts/
│   ├── test_local.py       # Тестирование папки изображений
│   └── bench.py            # Нагрузочное тестирование
├── docker-compose.yml
├── Dockerfile / Dockerfile.gpu
├── requirements.txt
└── API.md                  # ← Этот файл
```
