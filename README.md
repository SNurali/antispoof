# Anti-Spoofing Liveness Service

Face liveness detection (presentation attack detection) using **Silent-Face-Anti-Spoofing** models.

Classifies a face image as **REAL** (live person) or **SPOOF** (photo/screen replay).

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

## API

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
  "label": "real",
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
| `LIVENESS_THRESHOLD` | 0.5 | Score threshold for REAL verdict |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8090 | Server port |
| `MODEL_DIR` | ./models | Path to model weights |
| `DEVICE` | auto | `auto`/`cuda`/`cpu` |
| `MAX_BATCH` | 16 | Max images per batch request |

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
4. **Score Fusion**: Softmax outputs are summed; `argmax` determines label; score = `max_score / 2`
5. **Threshold**: `label == "real"` AND `score >= threshold` → is_real: true

## Security Notes (Phase 2)

Passive liveness from a single frame covers basic photo/screen attacks but does NOT cover:
- Direct file upload bypassing camera
- Video replay and deepfake attacks

For production: server-side nonce + short-lived capture session, frame series instead of single JPEG, active challenge-response (blink/turn head), device attestation (Play Integrity).

## Project Structure

```
antispoof/
  app/
    main.py            # FastAPI routes
    liveness.py        # Model loading + inference
    face_detect.py     # RetinaFace detection
    config.py          # Pydantic settings
  src/
    model_lib/         # MiniFASNet architecture (from Silent-Face repo)
  models/              # Model weights (.pth + detection_model/)
  scripts/
    test_local.py      # Batch image testing
    bench.py           # Load testing
  requirements.txt
  Dockerfile
  README.md
```
