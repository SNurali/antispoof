# Antispoof Liveness — Quick Integration Skill

## Context

Сервис проверки liveness (живое лицо или спуф). Работает в локальной сети.

## Connection

```
URL: http://192.168.0.6:8090
```

## Endpoints

### Check service health
```
GET /health
→ {"status":"healthy","device":"cpu","models_loaded":true}
```

### Verify face (multipart)
```
POST /verify
Body: multipart/form-data, field "image" = JPEG/PNG file
→ {"is_real":bool, "label":"real"|"spoof"|"no_face", "score":0.0-1.0, "face_detected":bool, "processing_ms":float}
```

### Verify face (base64) — recommended
```
POST /spoof-server
Body: {"photo": "<base64_string>"}
→ {"elapsed_time":float, "is_spoof":0|1}
```

## Decision Logic

```
is_spoof == 0 → REAL face → proceed with recognition
is_spoof == 1 → SPOOF detected → reject, ask to retry
face_detected == false → no face found → ask to reposition
```

## Code Pattern (Python)

```python
import requests, base64

def is_live_face(image_path: str) -> bool:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    r = requests.post("http://192.168.0.6:8090/spoof-server", json={"photo": b64}, timeout=10)
    return r.json()["is_spoof"] == 0
```

## Code Pattern (Kotlin/Android)

```kotlin
suspend fun isLiveFace(jpegBytes: ByteArray): Boolean {
    val b64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
    val result = api.verifyBase64(SpoofRequest(b64))
    return result.is_spoof == 0
}
```

## Speed

- CPU: ~300-500ms per request
- GPU: ~42ms per request

## Errors

- 400: invalid image/base64
- 403: IP not allowed (only 192.168.0.0/24 and 127.0.0.1)
- Connection refused: service is down

## Notes

- No auth required (local network only)
- Supports JPEG, PNG
- Max batch: 16 images (POST /verify_batch)
- Liveness threshold: 0.5 (configurable via env LIVENESS_THRESHOLD)
