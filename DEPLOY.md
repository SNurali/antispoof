# Деплой anti-spoofing (liveness) микросервиса

Временный микросервис проверки живости лица (liveness / anti-spoofing) для E-GAZ.
FastAPI + uvicorn, Python 3.11. **Прод — CPU-only** (~19 мс/кадр на CPU, GPU не требуется).

## 1. Требования сервера

| Параметр | Значение |
|---|---|
| ОС | Ubuntu/Debian (или любой Linux с python3.11) |
| Python | 3.11 (venv) |
| Диск | ~6 ГБ свободно (venv + torch CPU-сборка + opencv) |
| Порт | 8090 (наружу или за reverse-proxy) |
| CPU/GPU | Только CPU, GPU не нужен и не используется в проде |
| Веса моделей | уже в репо (`models/`, ~5.4 МБ), ничего докачивать не надо |

## 2. Вариант A — нативный деплой (venv + systemd)

### 2.1. Клонирование

```bash
git clone git@github.com:SNurali/antispoof.git
cd antispoof
```

### 2.2. Разовый запуск (проверка)

```bash
./deploy.sh
```

Скрипт идемпотентен:
- создаёт `.venv/` если его нет;
- ставит **CPU-сборку torch** через `--index-url https://download.pytorch.org/whl/cpu`
  (важно — без этого флага pip тянет CUDA-сборку на несколько ГБ, которая на проде не нужна);
- ставит остальные зависимости из `requirements.txt`;
- генерит self-signed TLS-сертификат в `certs/` (если отсутствует);
- запускает uvicorn и ждёт `/health`.

По умолчанию — **HTTPS** (нужен для веб-дашборда с камерой, браузер даёт `getUserMedia`
только по TLS). Для чистого API без дашборда можно поднять по HTTP:

```bash
MODE=http ./deploy.sh --nohup
```

### 2.3. Режимы запуска

| Команда | Поведение |
|---|---|
| `./deploy.sh` | foreground, лог в терминал, `Ctrl+C` останавливает |
| `./deploy.sh --nohup` | детач через `nohup`, лог в `antispoof.out.log`, PID в `antispoof.pid` |
| `./deploy.sh --install-service` | ставит systemd **user**-сервис `antispoof.service`, автозапуск + рестарт при падении |

### 2.4. Постоянный деплой (рекомендуется) — systemd

```bash
./deploy.sh --install-service
```

Дальше:

```bash
systemctl --user status antispoof.service      # статус
systemctl --user restart antispoof.service     # рестарт
journalctl --user -u antispoof.service -f      # логи в реальном времени
```

Чтобы сервис пережил разлогин/ребут сервера — включить linger один раз:

```bash
sudo loginctl enable-linger $(whoami)
```

### 2.5. Где смотреть дашборд

- HTTPS (по умолчанию): `https://<IP_СЕРВЕРА>:8090/` — браузер покажет предупреждение
  про self-signed сертификат, нужно подтвердить "продолжить" один раз.
- HTTP (`MODE=http`): `http://<IP_СЕРВЕРА>:8090/` — работает, но веб-камера в браузере
  доступна только с `localhost`, с другого хоста getUserMedia будет заблокирован без TLS.

## 3. Вариант B — Docker

### 3.1. CPU (прод)

```bash
docker compose up -d
```

`docker-compose.yml` уже настроен на CPU (`DEVICE=auto` определит `cpu`, т.к. в контейнере
нет GPU-доступа). Порт пробрасывается `8090:8090`.

Логи: `docker compose logs -f`
Остановка: `docker compose down`

### 3.2. GPU (опционально, не прод)

Если когда-нибудь понадобится GPU-инференс (например, локальная разработка на RTX 3080):

```bash
docker build -f Dockerfile.gpu -t antispoof:gpu .
docker run --gpus all -p 8090:8090 -e DEVICE=cuda antispoof:gpu
```

Требует установленный `nvidia-container-toolkit` на хосте.

**Два независимых слоя честно проверяют DEVICE (2026-07-17):** MiniFASNet passive-PAD
(`app/liveness.py`, torch) — уже умел `DEVICE=cuda/cpu/auto` — и identity-слой
Layer 0/2/3 активного liveness (`app/adaface.py` AdaFace IR-101 + `app/face_landmarks.py`
SCRFD/landmark_3d_68, оба onnxruntime/insightface) — теперь тоже читает тот же DEVICE
(`app/config.py::onnx_providers`). ВАЖНО: `Dockerfile.gpu` выше ставит только
`torch ... --index-url .../cu121` + обычный `requirements.txt` (CPU-wheel
`onnxruntime`) — значит identity-слой в этом образе останется на CPU, даже с
`DEVICE=cuda` (мягкий fallback, не падает, но и GPU не использует). Чтобы ускорить
и его — см. `requirements.txt`, секцию "OPTIONAL: GPU mode for the Active-liveness
identity layer": нужен `onnxruntime-gpu==1.20.1` вместо `onnxruntime`, cuDNN 9 в
образе (например базовый образ `nvidia/cuda:12.1.1-cudnn9-runtime-ubuntu22.04` вместо
текущего `...-runtime-...` без cuDNN) и `LD_LIBRARY_PATH` до нужных `.so`. На прод
(egaz-02.uz, CPU-only) это не влияет — там `DEVICE=cpu` жёстко в `deploy.sh`, ONNX
identity-слой и так CPU-only и остаётся CPU-only.

## 4. Сертификаты — почему self-signed и как заменить

`certs/` **не хранится в git** (см. `.gitignore`) — приватный ключ `key.pem` генерируется
заново на каждом сервере при первом запуске `deploy.sh` или `run-https.sh`. Это нормально
для временного/внутреннего сервиса, но:

- self-signed сертификат браузер помечает как небезопасный (ожидаемо, подтверждаете вручную);
- для настоящего прод-домена лучше поставить **nginx** перед сервисом и получить
  сертификат через **Let's Encrypt / certbot**, а uvicorn слушать по HTTP на `127.0.0.1:8090`
  (проксировать TLS-терминацию на nginx). Спросить BUSTA RHYMES/GHOSTFACE, если нужен
  такой проброс с реальным доменом.

## 5. Переменные окружения

| Переменная | Дефолт | Описание |
|---|---|---|
| `DEVICE` | `cpu` (в deploy.sh) / `auto` (в Docker) | `cpu`/`cuda`/`auto`. **На проде всегда `cpu`** |
| `HOST` | `0.0.0.0` | адрес прослушивания |
| `PORT` | `8090` | порт |
| `LIVENESS_THRESHOLD` | `0.5` | порог "живое лицо" (score >= threshold → real) |
| `MODEL_DIR` | `<repo>/models` | папка с весами (обычно не менять) |
| `MAX_BATCH` | `16` | максимальный размер батча в `/verify_batch` |

Пример переопределения:

```bash
PORT=9090 LIVENESS_THRESHOLD=0.6 ./deploy.sh --install-service
```

## 6. Обновление сервиса

```bash
cd antispoof
git pull
./deploy.sh --install-service   # переустановит зависимости при изменении requirements.txt
systemctl --user restart antispoof.service
```

(Если сервис уже установлен, `--install-service` безопасно перезапишет unit-файл и
перезапустит сервис — идемпотентно.)

## 7. Health-check / smoke-тест

```bash
# Health
curl -sk https://localhost:8090/health
# {"status":"ok","device":"cpu",...}

# Verify (multipart, одно фото)
curl -sk -X POST https://localhost:8090/verify \
  -F "file=@/path/to/face.jpg"
```

Python-пример:

```python
import requests

resp = requests.post(
    "https://localhost:8090/verify",
    files={"file": open("face.jpg", "rb")},
    verify=False,  # self-signed cert
)
print(resp.json())
```

## 8. Безопасность

- `certs/key.pem` (приватный ключ) **не коммитится**, генерируется на месте.
- `.env`, `*.log`, `.venv/` также исключены из git (`.gitignore`).
- Сервис временный/внутренний — не выставлять напрямую в открытый интернет без
  дополнительного слоя аутентификации/firewall (см. BUSTA RHYMES для периметра).
