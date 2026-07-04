# Antispoof Service — Интеграция

## О сервисе

**Face Liveness Detection** — сервис определяет живое ли лицо на фото или это спуф (фото с экрана, принт, подделка).

**Адрес:** `http://<IP>:8090`  
**Протокол:** HTTP (JSON / multipart)  
**Токен/авторизация:** нет (локальная сеть)

---

## Доступ

Сервис принимает запросы только из:
- `127.0.0.0/8` (localhost)
- `192.168.0.0/24` (локальная сеть)

Остальные IP получат `403 Access denied`.

---

## Эндпоинты

### `GET /health`

Проверка состояния сервиса.

**Запрос:**
```bash
curl http://192.168.0.6:8090/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "gpu": "N/A",
  "models_loaded": true
}
```

**Использование:** проверять перед отправкой фото. Если `models_loaded: false` — сервис не готов.

---

### `POST /verify`

Проверка одного изображения (multipart файл).

**Запрос:**
```bash
curl -F "image=@photo.jpg" http://192.168.0.6:8090/verify
```

**Параметры:**
| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `image` | file | да | JPEG/PNG изображение лица |

**Ответ:**
```json
{
  "is_real": true,
  "label": "real",
  "score": 0.9987,
  "threshold": 0.5,
  "face_detected": true,
  "processing_ms": 508.0,
  "signals": {
    "signal_scores": {
      "recapture": 0.84,
      "fft": 0.8,
      "lbp": 0.15,
      "color": 0.0,
      "moire": 0.0,
      "sharpness": 0.0,
      "jpeg": 0.0
    },
    "spoof_probability": 0.44,
    "signals_triggered": ["fft", "lbp", "recapture"],
    "nn_label": "real",
    "nn_score": 0.9987,
    "combined_label": "real",
    "combined_score": 0.9987
  }
}
```

**Поля ответа:**
| Поле | Тип | Описание |
|------|-----|----------|
| `is_real` | bool | `true` если лицо живое и score >= threshold |
| `label` | string | `"real"`, `"spoof"` или `"no_face"` |
| `score` | float | Уверенность 0.0–1.0 |
| `threshold` | float | Порог (по умолчанию 0.5) |
| `face_detected` | bool | Найдено ли лицо на изображении |
| `processing_ms` | float | Время обработки в мс |
| `signals` | object | Детальная разбивка по 7 сигналам |

---

### `POST /spoof-server`

Проверка liveness через base64 JSON. **Рекомендуемый формат для интеграции.**

**Запрос:**
```bash
PHOTO=$(base64 -w0 photo.jpg)
curl -X POST http://192.168.0.6:8090/spoof-server \
  -H "Content-Type: application/json" \
  -d "{\"photo\": \"$PHOTO\"}"
```

**Тело запроса (JSON):**
```json
{
  "photo": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `photo` | string | да | Изображение в формате base64 |

**Ответ:**
```json
{
  "elapsed_time": 0.023,
  "is_spoof": 0
}
```

| Поле | Тип | Описание |
|------|-----|----------|
| `elapsed_time` | float | Время обработки в секундах |
| `is_spoof` | int | `0` = живое лицо, `1` = спуф или лицо не найдено |

---

### `POST /verify_batch`

Пакетная проверка (до 16 изображений).

**Запрос:**
```bash
curl -F "images=@1.jpg" -F "images=@2.jpg" http://192.168.0.6:8090/verify_batch
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

## Примеры интеграции

### Python (requests)

```python
import requests
import base64

ANTISPOOF_URL = "http://192.168.0.6:8090"

def check_liveness(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{ANTISPOOF_URL}/verify",
            files={"image": f},
            timeout=10,
        )
    resp.raise_for_status()
    return resp.json()

def check_liveness_base64(image_path: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        f"{ANTISPOOF_URL}/spoof-server",
        json={"photo": b64},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()

result = check_liveness("/path/to/face.jpg")
print(f"Real: {result['is_real']}, Score: {result['score']}")
```

### Kotlin / Android (Retrofit)

```kotlin
interface AntispoofApi {
    @Multipart
    @POST("/verify")
    suspend fun verify(@Part image: MultipartBody.Part): LivenessResult

    @POST("/spoof-server")
    suspend fun verifyBase64(@Body request: SpoofRequest): SpoofResponse
}

data class LivenessResult(
    val is_real: Boolean,
    val label: String,
    val score: Double,
    val threshold: Double,
    val face_detected: Boolean,
    val processing_ms: Double
)

data class SpoofRequest(val photo: String)
data class SpoofResponse(
    val elapsed_time: Double,
    val is_spoof: Int
)

// Создание клиента
val api = Retrofit.Builder()
    .baseUrl("http://192.168.0.6:8090")
    .addConverterFactory(GsonConverterFactory.create())
    .build()
    .create(AntispoofApi::class.java)

// Использование (base64)
suspend fun isLiveFace(jpegBytes: ByteArray): Boolean {
    val b64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    val result = api.verifyBase64(SpoofRequest(b64))
    return result.is_spoof == 0
}

// Использование (multipart)
suspend fun isLiveFaceMultipart(jpegBytes: ByteArray): Boolean {
    val body = jpegBytes.toRequestBody("image/jpeg".toMediaType())
    val part = MultipartBody.Part.createFormData("image", "face.jpg", body)
    val result = api.verify(part)
    return result.is_real
}
```

### JavaScript / Node.js

```javascript
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

const ANTISPOOF_URL = 'http://192.168.0.6:8090';

// multipart
async function checkLiveness(imagePath) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imagePath));
  
  const res = await axios.post(`${ANTISPOOF_URL}/verify`, form, {
    headers: form.getHeaders(),
    timeout: 10000,
  });
  return res.data;
}

// base64
async function checkLivenessBase64(imagePath) {
  const b64 = fs.readFileSync(imagePath).toString('base64');
  const res = await axios.post(`${ANTISPOOF_URL}/spoof-server`, {
    photo: b64
  }, { timeout: 10000 });
  return res.data;
}
```

### cURL

```bash
# Multipart
curl -s -F "image=@photo.jpg" http://192.168.0.6:8090/verify

# Base64
PHOTO=$(base64 -w0 photo.jpg)
curl -s -X POST http://192.168.0.6:8090/spoof-server \
  -H "Content-Type: application/json" \
  -d "{\"photo\": \"$PHOTO\"}"

# Health check
curl -s http://192.168.0.6:8090/health
```

---

## Логика интеграции

### Рекомендуемый флоу

```
Камера захватила кадр
        │
        ▼
Отправить на /spoof-server (base64)
        │
        ▼
Получить ответ {is_spoof}
        │
        ├── is_spoof == 0 → Живое лицо → продолжить распознавание
        │
        └── is_spoof == 1 → Спуф → отклонить, попросить повторить
```

### Псевдокод

```python
def verify_and_recognize(camera_frame):
    # 1. Проверка liveness
    liveness = antispoof.verify(camera_frame)
    
    if liveness.is_spoof == 1:
        return "Отклонено: фото/экран вместо лица"
    
    # 2. Распознавание лица (AdaFace и т.д.)
    person = face_recognizer.identify(camera_frame)
    
    if person is None:
        return "Лицо не распознано"
    
    return f"Добро пожаловать, {person.name}"
```

---

## Ошибки

| HTTP код | Описание | Причина |
|----------|----------|---------|
| 200 | OK | Успешная проверка |
| 400 | Could not decode image | Файл не является изображением или повреждён |
| 400 | Empty file | Отправлен пустой файл |
| 400 | Invalid base64 | Некорректный base64 |
| 400 | File must be an image | Content-Type не image/* |
| 400 | Max batch size is 16 | Больше 16 изображений в batch |
| 403 | Access denied | IP не в списке разрешённых |
| 500 | Internal server error | Ошибка сервера (логи: `journalctl -u antispoof`) |

---

## Скорость

| Устройство | Одиночный запрос | Batch (16 фото) |
|------------|------------------|-----------------|
| RTX 3080 | ~42 ms | ~200 ms |
| CPU | ~300–500 ms | ~2–4 сек |

---

## Сигналы (для отладки)

| Сигнал | Вес | Что ловит |
|--------|-----|-----------|
| `recapture` | 45% | Низкодетальные пересъёмы (доминирующий) |
| `lbp` | 15% | Плоская текстура принтов/экранов |
| `color` | 10% | Узкий цветовой гамут экранов |
| `moire` | 10% | Муар от пиксельной сетки экрана |
| `jpeg` | 10% | Двойные JPEG-артефакты |
| `fft` | 5% | Периодические паттерны |
| `sharpness` | 5% | Однородная резкость (нет глубины) |

---

## Управление сервисом

```bash
sudo systemctl status antispoof     # статус
sudo systemctl restart antispoof    # перезапуск
sudo systemctl stop antispoof       # остановить
sudo journalctl -u antispoof -f     # логи в реальном времени
```

---

## Docker Compose

```yaml
version: "3.8"

services:
  antispoof:
    image: antispoof:latest
    ports:
      - "8090:8090"
    environment:
      - DEVICE=auto
      - LIVENESS_THRESHOLD=0.5
    restart: unless-stopped

  your-service:
    image: your-service:latest
    environment:
      ANTISPOOF_URL: http://antispoof:8090
    depends_on:
      - antispoof
```
