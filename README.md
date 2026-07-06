# Anti-Spoofing Liveness Service

Face liveness detection (presentation attack detection) using **Silent-Face-Anti-Spoofing** models.

Classifies a face image as **LIVE** (live person) or **SPOOF** (photo/screen replay) or **LOW_QUALITY** (no face / poor image).

Supports both **GPU** (CUDA) and **CPU** — auto-detects at startup.

## Quick Start

### Option 1: venv (recommended for dev)

```bash
cd /home/mrnurali/E-GAZ/antispoof

# CPU-only (for servers without GPU):
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090

# GPU (for dev machines with NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

### Option 2: Docker

```bash
# CPU (for production servers without GPU):
docker build -t antispoof .
docker run -p 8090:8090 antispoof

# GPU (requires nvidia-container-toolkit):
docker build -f Dockerfile.gpu -t antispoof-gpu .
docker run --gpus all -p 8090:8090 antispoof-gpu
```

## Verify Device

```bash
curl -s http://localhost:8090/health | python3 -m json.tool

# GPU:  "device": "cuda",  "gpu": "NVIDIA GeForce RTX 3080"
# CPU:  "device": "cpu",   "gpu": "N/A"
```

## API — Production (Phase 1)

### POST /pad/check (Phase 1 PAD-gate)

Primary endpoint for Laravel integration. Checks liveness after Adliya match.

```bash
PHOTO=$(base64 -w0 face.jpg)
curl -s -X POST http://localhost:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: your-secret" \
  -d "{
    \"correlation_id\": \"uuid\",
    \"transaction_type\": \"sale\",
    \"transaction_ref\": \"request:balloon\",
    \"face_photo\": \"$PHOTO\"
  }" | python3 -m json.tool
```

Response:
```json
{
  "verdict": "live",
  "reason": null,
  "score": 0.87,
  "threshold": 0.5,
  "face_detected": true,
  "signals": { ... },
  "model_version": "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1",
  "processing_ms": 18.4
}
```

**Verdicts:**
- `"live"` → Continue sale
- `"spoof"` → Reject (attack detected)
- `"low_quality"` → Ask user to retake photo

See **API.md** and **INTEGRATION.md** for full documentation.

---

## API — Legacy (Backward Compatible)

### GET /health

```json
{
  "status": "healthy",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3080",
  "models_loaded": true
}
```

### POST /verify

Single image liveness check.

```bash
curl -s -F "image=@live_face.jpg" http://localhost:8090/verify | python3 -m json.tool
```

Response:
```json
{
  "is_real": true,
  "label": "live",
  "score": 0.87,
  "threshold": 0.5,
  "face_detected": true,
  "processing_ms": 42.3
}
```

### POST /verify_batch

Batch liveness check (up to 16 images).

```bash
curl -s -F "images=@img1.jpg" -F "images=@img2.jpg" http://localhost:8090/verify_batch
```

---

## Scripts

### test_local.py — test a folder of images

```bash
python3 scripts/test_local.py /path/to/test_images/
```

Prints a table: file | label | score | ms for each image.

### bench.py — load test

```bash
# Single requests
python3 scripts/bench.py live_face.jpg -n 200 --concurrency 20

# With batch comparison
python3 scripts/bench.py live_face.jpg -n 200 --concurrency 20 --batch
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LIVENESS_THRESHOLD` | 0.5 | Score threshold for LIVE verdict |
| `HOST` | 127.0.0.1 | Server host (localhost by default) |
| `PORT` | 8090 | Server port |
| `MODEL_DIR` | ./models | Path to model weights |
| `DEVICE` | auto | `auto`/`cuda`/`cpu` |
| `MAX_BATCH` | 16 | Max images per batch request |
| `SERVICE_TOKEN` | (empty) | Shared-secret for X-Service-Token |
| `RATE_LIMIT_BURST` | 20 | Max requests per 1 second |
| `RATE_LIMIT_SUSTAINED` | 5 | Max requests per second (average over 60s) |

## Model Files

Required in `models/`:
- `2.7_80x80_MiniFASNetV2.pth` (~1.8 MB)
- `4_0_0_80x80_MiniFASNetV1SE.pth` (~1.8 MB)
- `detection_model/deploy.prototxt`
- `detection_model/Widerface-RetinaFace.caffemodel`

## How It Works

1. **Face Detection**: RetinaFace Caffe model detects the face bbox
2. **Cropping**: Face region extracted with scale-aware padding
3. **Dual Model Inference**: Both MiniFASNetV2 and MiniFASNetV1SE run on the cropped face
4. **Score Fusion**: Softmax outputs are summed; vote determines label; score = max_score / 2
5. **Threshold**: `label == "live"` AND `score >= threshold` → verdict: "live"
6. **Multi-signal Analysis**: 7 heuristic signals (recapture, FFT, LBP, color, moire, sharpness, JPEG)

---

## Architecture

```
Image/Camera frame
        │
        ▼
    ┌─────────────────┐
    │  RetinaFace     │  ← Face detection
    │  face_detect.py │
    └────────┬────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  LivenessEngine              │
    │  ┌──────────┐  ┌──────────┐  │
    │  │MiniFASV2 │  │MiniFASV1 │  │  ← Dual ensemble
    │  └────┬─────┘  └────┬─────┘  │
    │       └──────┬──────┘        │
    │              │               │
    │      ┌───────────────────┐   │
    │      │ multisignal.py    │   │  ← 7 heuristics:
    │      │ FFT, LBP, Color   │   │    recapture (45%),
    │      │ Moire, Sharpness  │   │    texture, color,
    │      │ JPEG, Recapture   │   │    periodic patterns
    │      └────────┬──────────┘   │
    │              │               │
    │              ▼               │
    │         _fuse() verdict      │
    └──────────────┬───────────────┘
                   │
                   ▼
      {verdict, reason, score, ...}
```

---

## Security Notes

### What Phase 1 Covers

Passive liveness from a single frame detects:
- **Printed photo** — recapture signal dominates (45% weight)
- **Screen replay** (tablet/phone) — color gamut + periodic patterns
- **Basic artifacts** — low detail, moire, JPEG compression

Real-world test (§6.1 FACEID_PHASE1_PAD_GATE.md):
- 21 attack samples, APCER=0%, BPCER=0%
- Confidence margins: 0.05–0.5 above threshold

### What Phase 1 Does NOT Cover

Single-frame passive PAD cannot detect:
- **Quality video replay** (temporal patterns removed in single frame)
- **Deepfake / faceswap** (requires temporal analysis or specialized detector)
- **Direct file upload** (closed at client side: camera-only capture, nonce, signature)

### Recommendation for Production

Phase 1 is compensating control, not primary auth:
1. **Fail-open** if service is unavailable (`reason="TIMEOUT"|"INTERNAL_ERROR"` → proceed + log to `security_events`; owner decision 2026-07-05, Phase 1 is a compensating control — revisit at Phase 2)
2. **Device attestation** on client (Play Integrity, App Attest) to prevent file injection
3. **Phase 2** with challenge-response (blink/turn head) for high-risk scenarios

---

## Performance

| Device | Single Request | Batch (16 images) |
|--------|---|---|
| RTX 3080 | ~18–42 ms | ~200 ms |
| CPU | ~300–500 ms | ~2–4 sec |

---

## Project Structure

```
antispoof/
  app/
    main.py            # FastAPI routes (health, verify, verify_batch, pad/check)
    liveness.py        # Model loading + inference + ensemble
    face_detect.py     # RetinaFace detection
    multisignal.py     # 7 heuristic anti-spoof signals
    config.py          # Pydantic settings
    static/            # Test web UI
  src/
    model_lib/         # MiniFASNet architectures
  models/              # Model weights (.pth + detection_model/)
  scripts/
    test_local.py      # Batch image testing
    bench.py           # Load testing
  requirements.txt
  Dockerfile
  README.md            # ← This file
  API.md               # Full API documentation
  INTEGRATION.md       # Integration examples
```

---

## References

- **Silent-Face repo:** https://github.com/vlarsson/Silent-Face-Anti-Spoofing
- **Phase 1 Spec:** `FACEID_PHASE1_PAD_GATE.md`
- **API Docs:** `API.md`
- **Integration Examples:** `INTEGRATION.md`
