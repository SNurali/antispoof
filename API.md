# Antispoof API — Документация

Face liveness detection (anti-spoofing) сервис. Определяет живое ли лицо на фото или это спуф (фото/экран/принт).

**Стек:** FastAPI + PyTorch + RetinaFace + MiniFASNet  
**Порт:** `8090`  
**Режим:** Локальный сервер (не暴露ывается в интернет)

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
  "models_loaded": true
}
```

---

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
  "label": "real",
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
| `label` | string | `"real"`, `"spoof"` или `"no_face"` |
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
    {"is_real": true, "label": "real", "score": 0.87, "face_detected": true},
    {"is_real": false, "label": "spoof", "score": 0.92, "face_detected": true}
  ],
  "total_ms": 78.5,
  "count": 2
}
```

---

### POST /spoof-server

Проверка liveness через base64 JSON. Формат совместим с внешним сервером наставника.

```bash
# Подготовить base64 и отправить
PHOTO_B64=$(base64 -w0 face.jpg)
curl -s -X POST http://localhost:8090/spoop-server \
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
  "is_spoof": 1
}
```

| Поле | Тип | Описание |
|---|---|---|
| `elapsed_time` | float | Время обработки в секундах |
| `is_spoof` | int | `1` — спуф (фото/экран), `0` — живое лицо |

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LIVENESS_THRESHOLD` | `0.5` | Порог для вердикта REAL |
| `HOST` | `0.0.0.0` | Хост сервера |
| `PORT` | `8090` | Порт сервера |
| `MODEL_DIR` | `./models` | Путь к весам моделей |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `MAX_BATCH` | `16` | Максимум изображений в batch |

---

## Интеграция с другими сервисами

Сервис работает по HTTP. Другие сервисы подключаются через `http://<IP>:8090`.

### Python (httpx)

```python
import httpx

ANTISPOOF_URL = "http://localhost:8090"

async def check_liveness(image_bytes: bytes) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ANTISPOOF_URL}/verify",
            files={"image": ("face.jpg", image_bytes, "image/jpeg")},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()

# Использование:
result = await check_liveness(jpeg_bytes)
if result["is_real"]:
    print(f"Живое лицо, score={result['score']}")
else:
    print(f"Спуф! label={result['label']}, score={result['score']}")
```

### Python (requests — синхронный)

```python
import requests

def check_liveness_sync(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            "http://localhost:8090/verify",
            files={"image": f},
            timeout=5,
        )
    return resp.json()
```

### Kotlin / Android (Retrofit + OkHttp)

```kotlin
// --- API interface ---
interface AntispoofApi {
    @Multipart
    @POST("/verify")
    suspend fun verify(@Part image: MultipartBody.Part): LivenessResult

    @Multipart
    @POST("/verify_batch")
    suspend fun verifyBatch(@Part images: List<MultipartBody.Part>): BatchResult
}

// --- Models ---
data class LivenessResult(
    val is_real: Boolean,
    val label: String,
    val score: Double,
    val threshold: Double,
    val face_detected: Boolean,
    val processing_ms: Double
)

data class BatchResult(
    val results: List<LivenessResult>,
    val total_ms: Double,
    val count: Int
)

// --- Создание клиента ---
val antispoof = Retrofit.Builder()
    .baseUrl("http://192.168.0.100:8090")  // IP локального сервера
    .addConverterFactory(GsonConverterFactory.create())
    .build()
    .create(AntispoofApi::class.java)

// --- Вызов ---
suspend fun verifyFace(jpegBytes: ByteArray): Boolean {
    val body = jpegBytes.toRequestBody("image/jpeg".toMediaType())
    val part = MultipartBody.Part.createFormData("image", "face.jpg", body)
    val result = antispoof.verify(part)
    return result.is_real
}
```

### JavaScript / Node.js

```js
async function checkLiveness(imageBuffer) {
  const form = new FormData();
  form.append('image', new Blob([imageBuffer], { type: 'image/jpeg' }), 'face.jpg');

  const res = await fetch('http://localhost:8090/verify', {
    method: 'POST',
    body: form,
  });
  return res.json();
}
```

### cURL

```bash
# Одиночная проверка
curl -s -F "image=@photo.jpg" http://localhost:8090/verify | python3 -m json.tool

# Batch проверка
curl -s -F "images=@1.jpg" -F "images=@2.jpg" http://localhost:8090/verify_batch

# Проверка здоровья
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
      - "8090:8090"
    environment:
      - DEVICE=auto
      - LIVENESS_THRESHOLD=0.5
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
tracker  →  http://antispoof:8090/verify
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
         {is_real, label, score}
```

---

## Сигналы (multisignal.py)

| Сignal | Вес | Что ловит |
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

- Сервис **не шифрует** трафик (HTTP) — допустимо для локальной сети
- Для HTTPS: используйте `run-https.sh` (самоподписанный сертификат) или nginx reverse proxy
- **Не暴露ывать** в интернет — слушает на `0.0.0.0:8090` без аутентификации
- Пассивная лайвнесс (один кадр) НЕ защищает от:
  - Прямой загрузки файла (минуя камеру)
  - Видео-реplays и дипфейков
  - Для продакшена: nonce + серия кадров + challenge-response (моргнуть/повернуть голову)

---

## Скорость

| Устройство | Одиночный запрос | Batch (16 фото) |
|---|---|---|
| RTX 3080 | ~42 ms | ~200 ms |
| CPU | ~300–500 ms | ~2–4 сек |

---

## Файловая структура

```
antispoof/
├── app/
│   ├── main.py            # FastAPI роуты (health, verify, verify_batch)
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
