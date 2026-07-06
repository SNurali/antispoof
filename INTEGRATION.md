# Antispoof Service — Интеграция

## О сервисе

**Face Liveness Detection** — сервис определяет живое ли лицо на фото или это спуф (фото с экрана, принт, подделка).

**Адрес:** `http://<IP>:8090`  
**Протокол:** HTTP (JSON / multipart)  
**Аутентификация:** `X-Service-Token` (shared-secret)  
**Rate limit:** 20 req/1s burst, 5 req/s sustained per IP

---

## Доступ

Сервис принимает запросы только из:
- `127.0.0.0/8` (localhost)
- `192.168.0.0/24` (локальная LAN)
- `10.0.0.0/8` (Docker VPC)
- `172.16.0.0/12` (Docker VPC)

Остальные IP получат `403 Access denied`.

---

## Phase 1 Production: POST /pad/check

**Рекомендуемый эндпоинт для Laravel-интеграции (Phase 1 PAD-gate).**

Проверяет лицо после успешной Adliya-идентичности (порог 70%) и возвращает компенсирующий контроль: живое ли это лицо или подделка.

### Запрос

```bash
curl -X POST http://192.168.0.6:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: your-secret-token" \
  -d '{
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "transaction_type": "sale",
    "transaction_ref": "12345:67890",
    "face_photo": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  }'
```

**Параметры (JSON):**
| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|----------|
| `correlation_id` | string (UUID) | да | Для трассировки логов (генерирует Laravel) |
| `transaction_type` | string | да | Строго `"sale"` (v1) — расширяется в будущем |
| `transaction_ref` | string | да | `id_request:id_ballon` (натуральный ключ транзакции, NOT pinfl) |
| `face_photo` | string (base64) | да | Base64 JPEG/PNG, тот же кадр, что в Adliya |

### Ответ

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
|------|-----|----------|
| `verdict` | string | `"live"` ‖ `"spoof"` ‖ `"low_quality"` |
| `reason` | string\|null | Причина вердикта (см. ниже) |
| `score` | float | Уверенность 0.0–1.0 |
| `threshold` | float | Порог (по умолчанию 0.5) |
| `face_detected` | bool | Найдено ли лицо на изображении |
| `signals` | object | Детальная разбивка по 7 сигналам |
| `save_frame` | bool | Сервис сохранил кадр для расследования |
| `model_version` | string | Версия модели (для воспроизводимости) |
| `processing_ms` | float | Время обработки в мс |

**Логика вердиктов (Laravel должна отличать эти три случая):**

1. **`verdict="live"` → `reason=null`**
   - Лицо живое, score >= threshold
   - Laravel: **продолжить `/sales` как обычно**

2. **`verdict="spoof"` → `reason="PASSIVE_PAD_SPOOF"`**
   - Обнаружена атака (фото/экран/пересъёмка)
   - Laravel: **ОТКЛОНИТЬ продажу** с кодом ошибки (не `FACE_ID_MISMATCH`!)
   - UX: "Фото не прошла проверку безопасности. Повторите попытку"
   - Это ИНЦИДЕНТ безопасности (лог, мониторинг)

3. **`verdict="low_quality"` → `reason="NO_FACE"|"LOW_QUALITY"|...`**
   - Нет лица на кадре ИЛИ качество недостаточно (переснять)
   - Laravel: **просит пользователя переснять** (UX-реджект, не блокировка)
   - Это НЕ инцидент безопасности — просто технический отказ
   - Подлежит повтору без штрафов

**Коды reason:**
| reason | verdict | Что делать |
|--------|---------|-----------|
| `null` | `live` | Продолжить транзакцию |
| `"PASSIVE_PAD_SPOOF"` | `spoof` | Отклонить, инцидент |
| `"NO_FACE"` | `low_quality` | Переснять лицо |
| `"LOW_QUALITY"` | `low_quality` | Переснять (слабое качество) |
| `"TIMEOUT"` | `low_quality` | Сервис не ответил вовремя (fail-closed/fail-open по политике) |
| `"INTERNAL_ERROR"` | `low_quality` | Сервис упал (fail-closed/fail-open) |

**Ошибки HTTP:**
| Код | Описание |
|-----|----------|
| 200 | OK — вердикт вернулся (даже при `low_quality`) |
| 400 | Invalid base64 / Image size > 8MB / Image dimensions > 4000×4000 |
| 401 | Missing/invalid `X-Service-Token` |
| 403 | IP not in allowlist |
| 422 | `transaction_type` не `"sale"` |
| 429 | Rate limit (20/1s or 5/s avg over 60s per IP) |
| 500 | Internal error (контракт: `{"verdict":"low_quality", "reason":"INTERNAL_ERROR", ...}`) |

---

## Legacy Endpoints (Backward Compatible)

### GET /health

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
  "models_loaded": true,
  "model_version": "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1"
}
```

**Использование:** проверять перед отправкой фото. Если `models_loaded: false` — сервис не готов (503).

---

### POST /verify

Проверка одного изображения (multipart файл). **Legacy — используйте `/pad/check` для нового кода.**

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
  "label": "live",
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
    "nn_score": 0.9987
  }
}
```

**Поля ответа:**
| Поле | Тип | Описание |
|------|-----|----------|
| `is_real` | bool | `true` если лицо живое и score >= threshold |
| `label` | string | `"live"`, `"spoof"` или `"no_face"` |
| `score` | float | Уверенность 0.0–1.0 |
| `threshold` | float | Порог (по умолчанию 0.5) |
| `face_detected` | bool | Найдено ли лицо на изображении |
| `processing_ms` | float | Время обработки в мс |
| `signals` | object | Детальная разбивка по 7 сигналам |

---

### POST /spoof-server

Проверка liveness через base64 JSON. **Legacy — используйте `/pad/check` для нового кода.**

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

### POST /verify_batch

Пакетная проверка (до 16 изображений). **Legacy.**

**Запрос:**
```bash
curl -F "images=@1.jpg" -F "images=@2.jpg" http://192.168.0.6:8090/verify_batch
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

## Примеры интеграции

### PHP / Laravel (рекомендуется)

```php
<?php
namespace App\Services;

use Illuminate\Http\Client\Response;
use Illuminate\Support\Facades\Http;
use Ramsey\Uuid\Uuid;

class PadGateService {
    private string $antispoof_url;
    private string $service_token;

    public function __construct() {
        $this->antispoof_url = env('ANTISPOOF_URL', 'http://192.168.0.6:8090');
        $this->service_token = env('ANTISPOOF_SERVICE_TOKEN');
    }

    /**
     * Проверка лайвнесс после Adliya-идентичности
     */
    public function checkPadGate(
        string $transaction_ref,
        string $face_photo_base64
    ): array {
        $correlation_id = Uuid::uuid4()->toString();

        $response = Http::withHeaders([
            'X-Service-Token' => $this->service_token,
            'Content-Type' => 'application/json',
        ])->post("{$this->antispoof_url}/pad/check", [
            'correlation_id' => $correlation_id,
            'transaction_type' => 'sale',
            'transaction_ref' => $transaction_ref,
            'face_photo' => $face_photo_base64,
        ]);

        if (!$response->successful()) {
            // Ошибка сети/сервиса — залогировать
            \Log::error('PAD-gate error', [
                'correlation_id' => $correlation_id,
                'status' => $response->status(),
                'body' => $response->body(),
            ]);
            return [
                'verdict' => 'error',
                'reason' => 'SERVICE_UNAVAILABLE',
            ];
        }

        return $response->json();
    }

    /**
     * Обработка вердикта в контроллере продажи
     */
    public function handleSaleWithPadCheck(
        int $id_request,
        int $id_ballon,
        string $face_photo_b64
    ): \Illuminate\Http\Response {
        $transaction_ref = "{$id_request}:{$id_ballon}";

        // 1. Проверка PAD-gate
        $pad_result = $this->checkPadGate($transaction_ref, $face_photo_b64);

        // 2. Отличаем инциденты от технических отказов
        match ($pad_result['verdict'] ?? null) {
            'live' => $this->continueWithSale($id_request, $id_ballon),
            'spoof' => $this->rejectWithSecurityIncident($id_request, $pad_result['reason']),
            'low_quality' => $this->askToRetake('переснимите'),
            default => $this->failClosed('PAD check unavailable'),
        };
    }
}
?>
```

### Python (httpx) — /pad/check

```python
import httpx
import base64
from uuid import uuid4

ANTISPOOF_URL = "http://192.168.0.6:8090"
SERVICE_TOKEN = "your-secret-token"

async def check_liveness_pad(
    transaction_ref: str,
    face_photo_bytes: bytes
) -> dict:
    """Проверка лайвнесс через PAD-gate (Phase 1)."""
    correlation_id = str(uuid4())
    photo_b64 = base64.b64encode(face_photo_bytes).decode()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ANTISPOOF_URL}/pad/check",
            json={
                "correlation_id": correlation_id,
                "transaction_type": "sale",
                "transaction_ref": transaction_ref,
                "face_photo": photo_b64,
            },
            headers={"X-Service-Token": SERVICE_TOKEN},
            timeout=10.0,
        )
    
    if resp.status_code == 200:
        return resp.json()
    else:
        return {
            "verdict": "error",
            "reason": "SERVICE_ERROR",
            "status": resp.status_code,
        }

def handle_pad_verdict(pad_result: dict) -> None:
    """Обработка вердикта PAD-gate."""
    match pad_result["verdict"]:
        case "live":
            print("✓ Живое лицо, продолжить продажу")
        case "spoof":
            print(f"✗ ИНЦИДЕНТ: {pad_result['reason']} — отклонить")
        case "low_quality":
            print(f"⚠ Техническая ошибка: {pad_result['reason']} — переснять")
        case _:
            print("✗ PAD check недоступен")
```

### Kotlin / Android (Retrofit) — /pad/check

```kotlin
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

interface PadGateApi {
    @POST("/pad/check")
    suspend fun checkPadGate(
        @Body request: PadCheckRequest,
        @Header("X-Service-Token") serviceToken: String,
    ): PadCheckResponse
}

data class PadCheckRequest(
    val correlation_id: String,
    val transaction_type: String = "sale",
    val transaction_ref: String,
    val face_photo: String, // base64
)

data class PadCheckResponse(
    val verdict: String,              // "live" | "spoof" | "low_quality"
    val reason: String?,              // null | "PASSIVE_PAD_SPOOF" | "NO_FACE" | ...
    val score: Double,
    val threshold: Double,
    val face_detected: Boolean,
    val signals: Map<String, Any>,
    val save_frame: Boolean,
    val model_version: String,
    val processing_ms: Float,
)

// Использование
suspend fun verifyFaceForSale(
    requestId: Int,
    ballonId: Int,
    jpegBytes: ByteArray
): Boolean {
    val b64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    val txn_ref = "$requestId:$ballonId"

    return try {
        val result = padGateApi.checkPadGate(
            PadCheckRequest(
                correlation_id = UUID.randomUUID().toString(),
                transaction_ref = txn_ref,
                face_photo = b64,
            ),
            serviceToken = BuildConfig.ANTISPOOF_TOKEN,
        )

        when (result.verdict) {
            "live" -> {
                Log.d("PadGate", "✓ Живое лицо")
                true
            }
            "spoof" -> {
                Log.e("PadGate", "ИНЦИДЕНТ: ${result.reason}")
                false
            }
            else -> {  // low_quality
                Log.w("PadGate", "Переснять: ${result.reason}")
                false
            }
        }
    } catch (e: Exception) {
        Log.e("PadGate", "Service error", e)
        false  // fail-closed
    }
}
```

### JavaScript / Node.js

```javascript
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');

const ANTISPOOF_URL = 'http://192.168.0.6:8090';
const SERVICE_TOKEN = process.env.ANTISPOOF_SERVICE_TOKEN;

async function checkLivenessPad(transactionRef, facePhotoBase64) {
  const correlationId = uuidv4();

  try {
    const response = await axios.post(
      `${ANTISPOOF_URL}/pad/check`,
      {
        correlation_id: correlationId,
        transaction_type: 'sale',
        transaction_ref: transactionRef,
        face_photo: facePhotoBase64,
      },
      {
        headers: {
          'X-Service-Token': SERVICE_TOKEN,
          'Content-Type': 'application/json',
        },
        timeout: 10000,
      }
    );

    return response.data;
  } catch (error) {
    console.error('PAD-gate error:', error.response?.status, error.message);
    return {
      verdict: 'error',
      reason: 'SERVICE_ERROR',
    };
  }
}

// Использование
async function handleSaleWithLiveness(requestId, ballonId, jpegBuffer) {
  const txnRef = `${requestId}:${ballonId}`;
  const photoB64 = jpegBuffer.toString('base64');

  const result = await checkLivenessPad(txnRef, photoB64);

  if (result.verdict === 'live') {
    console.log('✓ Live face detected, continue sale');
    return true;
  } else if (result.verdict === 'spoof') {
    console.error(`✗ Spoofing attempt detected: ${result.reason}`);
    return false;
  } else {
    console.warn(`⚠ Low quality: ${result.reason} — ask to retake`);
    return false;
  }
}
```

### cURL (быстрое тестирование)

```bash
# Кодировать фото в base64
PHOTO=$(base64 -w0 photo.jpg)

# Отправить на /pad/check
curl -s -X POST http://192.168.0.6:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: your-secret" \
  -d "{
    \"correlation_id\": \"test-uuid-123\",
    \"transaction_type\": \"sale\",
    \"transaction_ref\": \"req123:bal456\",
    \"face_photo\": \"$PHOTO\"
  }" | python3 -m json.tool

# Health check
curl -s http://192.168.0.6:8090/health | python3 -m json.tool
```

---

## Рекомендуемый флоу (Laravel)

```
Камера захватила кадр
        │
        ▼
Отправить на Adliya (идентичность)
        │
        ├─ Adliya score < 70 → отклонить "лицо не распознано"
        │
        └─ Adliya score >= 70 → продолжить
              │
              ▼
          Отправить ТЕ ЖЕ КАДР на /pad/check (лайвнесс)
              │
              ├─ verdict="live" → ПРОДОЛЖИТЬ ПРОДАЖУ
              │
              ├─ verdict="spoof" → ОТКЛОНИТЬ (инцидент! логирование, мониторинг)
              │
              └─ verdict="low_quality" → ПОПРОСИТЬ ПЕРЕСНЯТЬ (не инцидент)
```

---

## Ошибки

| HTTP код | Описание | Что делать |
|----------|----------|-----------|
| 200 | OK | Использовать `verdict` из ответа |
| 400 | Invalid base64 / Size/Dimensions | Переснять, проверить кодирование |
| 401 | Missing/invalid X-Service-Token | Проверить переменные окружения |
| 403 | Access denied (IP) | Проверить IP сервиса, firewall |
| 422 | Invalid transaction_type | Использовать `"sale"` (v1) |
| 429 | Rate limit | Снизить frequency запросов (max 20/1s burst) |
| 500 | Internal error | Контракт возвращает `verdict="low_quality", reason="INTERNAL_ERROR"` |

---

## Скорость

| Устройство | Одиночный запрос | Batch (16 фото) |
|------------|------------------|-----------------|
| RTX 3080 | ~18–42 ms | ~200 ms |
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
      - "127.0.0.1:8090:8090"
    environment:
      - DEVICE=auto
      - LIVENESS_THRESHOLD=0.5
      - SERVICE_TOKEN=your-secret-token
      - HOST=127.0.0.1
    restart: unless-stopped

  your-service:
    image: your-service:latest
    environment:
      ANTISPOOF_URL: http://antispoof:8090
      ANTISPOOF_SERVICE_TOKEN: your-secret-token
    depends_on:
      - antispoof
```
