# Миграция antispoof: systemd → Docker (v1)

Пошаговая инструкция для перевода сервиса `antispoof` с нативного systemd-деплоя
(`antispoof.service`, venv, `uvicorn` напрямую на `0.0.0.0:8090`) на Docker Compose
(`antispoof-app` + `antispoof-redis`, `127.0.0.1:8090` строго, non-root, healthcheck,
1 воркер жёстко). Выполняется на целевом хосте (Ubuntu 20.04.6 LTS, Docker 28.0.2,
Compose v2.34.0, CPU-only, тот же хост, что и Laravel).

**Канал auth:** X-Service-Token (без mTLS) — сервис слушает только `127.0.0.1:8090`,
Laravel стучится локально на этот же хост.

---

## 0. ⚠️ Известный инцидент — обязателен ДО любого другого шага

На проде на 8090 сейчас **два** процесса `uvicorn`:
- старый (с 16.07) реально держит порт и отвечает на запросы;
- свежий крутится вхолостую на ~110% CPU, порт не держит (осиротевший процесс,
  не был корректно остановлен при последнем рестарте/деплое).

**Если это не зачистить до `docker compose up`, новый Docker-контейнер не сможет
забиндиться на `127.0.0.1:8090` — порт будет занят.** Начинаем строго отсюда.

```bash
# 0.1 — что вообще слушает 8090 и кто держит CPU
ss -ltnp | grep ':8090'
ps aux | grep -i '[u]vicorn'

# Ожидаемо увидим 2+ процесса uvicorn — например:
#   root  12345  ...  uvicorn app.main:app --host 0.0.0.0 --port 8090   <- старый, держит порт
#   root  23456  ... (110% CPU) uvicorn app.main:app ...                <- осиротевший, вхолостую
```

Зафиксировать оба PID перед остановкой (пригодится, если что-то пойдёт не так).

---

## 1. Полная остановка systemd-сервиса

```bash
sudo systemctl stop antispoof.service
sudo systemctl disable antispoof.service
sudo systemctl status antispoof.service   # ожидаем: inactive (dead)
```

Если сервис управлялся через `ctl.sh`/`spoof` (symlink `/usr/local/bin/spoof`),
дополнительно:

```bash
spoof stop || true
```

---

## 2. Зачистка осиротевших uvicorn-процессов

`systemctl stop` иногда не убивает форк-детей/зомби-процессы (это и есть причина
инцидента из шага 0). Проверяем и добиваем вручную:

```bash
# 2.1 — что осталось живо после systemctl stop
ps aux | grep -i '[u]vicorn'

# 2.2 — если что-то осталось, убить по паттерну (сначала мягко, потом жёстко)
pkill -f 'uvicorn app.main:app' || true
sleep 2
pkill -9 -f 'uvicorn app.main:app' || true

# 2.3 — убедиться, что порт 8090 СВОБОДЕН (это условие, без которого дальше не идти)
ss -ltnp | grep ':8090' && echo "❌ ПОРТ ВСЁ ЕЩЁ ЗАНЯТ — искать держателя вручную" || echo "✅ порт 8090 свободен"

# 2.4 — альтернативная проверка (если ss недоступен/не даёт -p без root)
sudo lsof -i :8090 || echo "✅ lsof: ничего не слушает 8090"
```

**Не переходить к шагу 3, пока `ss -ltnp | grep ':8090'` не пустой.** Если порт
всё ещё занят после `pkill -9`, найти держателя явно (`sudo fuser 8090/tcp`) и
разобраться, что это — не глушить вслепую что-то постороннее на хосте.

---

## 3. Проверка Docker/Compose на хосте

```bash
docker --version          # ожидаем 28.0.2 (или совместимая 28.x)
docker compose version    # ожидаем v2.34.0 (или совместимая v2.3x)
docker info | grep -i 'server version\|cgroup'   # sanity-check демона
```

---

## 4. Получение образа — способ передачи (без registry)

Выбран способ: **`docker save | gzip`**, передача архива файлом (не registry —
партнёр не даёт нам push-доступ, свой registry поднимать ради одного образа
избыточно). Альтернатива Б (сборка из исходников на их стороне) описана в
конце этого раздела, если способ А по какой-то причине не подходит.

### Вариант А — образ передан файлом (`antispoof-1.0.0-cpu.tar.gz`)

Собран и проверен локально (50 CENT, 2026-07-19): `docker save antispoof:1.0.0-cpu
| gzip` → **~1.1 ГБ** (образ до сжатия — 2.4 ГБ; в основном torch CPU-wheels +
opencv + onnxruntime/insightface + веса моделей `models/`, 254 МБ). Передавать
контрагенту одним файлом (флешка/`scp`/файлообменник — как договоритесь).

```bash
# 4.1 — распаковать/загрузить образ в локальный Docker
gunzip -c antispoof-1.0.0-cpu.tar.gz | docker load
# ожидаем в выводе: Loaded image: antispoof:1.0.0-cpu

# 4.2 — проверить, что образ на месте
docker images | grep antispoof
```

Дальше репозиторий (код + `docker-compose.yml` + `.env.example` + `models/`)
должен быть распакован в рабочую директорию на хосте, например:

```bash
cd /var/lib/mysql/logs/yoyo/antispoof   # тот же путь, что и раньше — WorkingDirectory
# сюда же — docker-compose.yml, .env (см. шаг 5), entrypoint.sh не нужен отдельно
# (он уже ВНУТРИ загруженного образа, слоем)
```

Т.к. образ загружен через `docker load`, `docker-compose.yml`'s `build: .` НЕ
нужно триггерить — задать явно `image: antispoof:1.0.0-cpu` уже стоит в
`docker-compose.yml`, и `docker compose up -d` при найденном локально образе
пересобирать НЕ станет (`build:` используется только если такого image ещё нет
локально ИЛИ вызван `docker compose build`/`--build`).

### Вариант Б — сборка из исходников на их стороне (если файл не подошёл)

Нужно им передать:
- весь репозиторий (`git clone` доступа ИЛИ архив с исходниками, включая
  `models/` — 254 МБ, обязательно, веса моделей не докачиваются на рантайме);
- `Dockerfile`, `entrypoint.sh`, `docker-compose.yml`, `.env.example` (уже в
  репозитории);
- команда сборки на их стороне:
  ```bash
  cd antispoof
  docker compose build   # соберёт antispoof:1.0.0-cpu локально из Dockerfile
  ```
- **важно предупредить:** первая сборка тянет `torch`/`torchvision` (CPU-wheels)
  и `onnxruntime`/`insightface` из PyPI — нужен исходящий интернет на хосте
  сборки хотя бы один раз (кэшируется слоями дальше). Если на их хосте
  исходящего интернета нет — только вариант А (offline `docker load`).

---

## 5. Файл `.env`

```bash
cp .env.example .env
```

Обязательно заменить:

```
SERVICE_TOKEN=CHANGE_ME     # ⚠️ заменить на реальный shared-secret с Laravel
```

Значение `SERVICE_TOKEN` — то же самое, что уже настроено на стороне Laravel
(shared secret, заголовок `X-Service-Token`). Если оно менялось при миграции —
согласовать с командой Laravel/Umid ДО запуска, иначе `/pad/check` начнёт
получать 401 на всех запросах.

Остальные дефолты в `.env.example` уже рассчитаны на этот Docker-стек
(`ENVIRONMENT=prod`, `SESSION_STORE_BACKEND=redis` → указывает на встроенный
`antispoof-redis`, `WEB_CONCURRENCY=1`, `DEVICE=cpu`) — трогать не нужно, если
нет отдельной договорённости.

`chmod 600 .env` — файл содержит секрет, не должен быть group/world-readable.

---

## 6. Запуск стека

```bash
docker compose up -d
```

Ожидаемый вывод: два контейнера — `antispoof-app`, `antispoof-redis`, оба
`Started`/`Healthy` через некоторое время (healthcheck `start_period` — 30с
для antispoof, 5с для redis).

```bash
# 6.1 — статус и health обоих контейнеров
docker compose ps
# STATUS должен дойти до "Up X seconds (healthy)" для обоих

# 6.2 — если antispoof не становится healthy — смотреть логи
docker compose logs -f antispoof
```

---

## 7. Smoke-тест (с хоста, изнутри — 127.0.0.1)

```bash
# 7.1 — health, без токена
curl -s http://127.0.0.1:8090/health
# ожидаем: {"status":"healthy","device":"cpu","models_loaded":true,...}
# ⚠️ status="not_ready" / models_loaded=false + HTTP 503 = модели ещё грузятся
# или не загрузились — не считать деплой успешным, пока не healthy

# 7.2 — verify, без токена (эндпоинт /verify не требует SERVICE_TOKEN),
#       нужен любой файл-фото с лицом для реального прогона инференса
curl -s -X POST http://127.0.0.1:8090/verify \
  -F "image=@/path/to/face.jpg;type=image/jpeg"
# ожидаем JSON с полем verdict/is_spoof, HTTP 200 — не 500

# 7.3 — pad/check, С токеном (проверяем сам X-Service-Token канал)
curl -s -X POST http://127.0.0.1:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: <ЗНАЧЕНИЕ_ИЗ_.env>" \
  -d "{\"correlation_id\":\"smoke-test-$(date +%s)\",\"transaction_type\":\"sale\",\"transaction_ref\":\"smoke:0\",\"face_photo\":\"$(base64 -w0 /path/to/face.jpg)\"}"
# ожидаем verdict: "live"/"spoof"/"low_quality", НЕ HTTP 401

# 7.4 — без токена ДОЛЖЕН вернуть 401 (проверка, что auth реально включена)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8090/pad/check \
  -H "Content-Type: application/json" \
  -d '{"correlation_id":"smoke-noauth","transaction_type":"sale","transaction_ref":"smoke:1","face_photo":"AAAA"}'
# ожидаем: 401

# 7.5 — снаружи хоста порт быть НЕ должен виден (проверка биндинга на loopback)
#       выполнить с ДРУГОЙ машины (не с самого хоста):
curl -sS --max-time 3 http://<IP_ХОСТА>:8090/health
# ожидаем: connection refused / timeout — если ответ пришёл, биндинг неверный,
# см. docker-compose.yml — порт должен быть "127.0.0.1:8090:8090", НЕ "8090:8090"
```

Все пункты 7.1–7.5 должны пройти, прежде чем считать миграцию успешной и
переходить к финальной уборке (шаг 8).

---

## 8. Финальная уборка (только после успешного smoke-теста)

```bash
# Убедиться, что systemd-юнит не автозапустится при ребуте
sudo systemctl is-enabled antispoof.service   # ожидаем: disabled

# (опционально, через несколько дней стабильной работы Docker-стека)
# sudo systemctl mask antispoof.service
```

Симлинк `/usr/local/bin/spoof` (управлял старым процессом через `ctl.sh`)
можно оставить — он безвреден, но больше не относится к реальному сервису;
управление теперь через `docker compose {up,down,restart,logs} -d`.

---

## 9. Откат на systemd (если Docker-стек не завёлся)

```bash
# 9.1 — остановить Docker-стек
cd /var/lib/mysql/logs/yoyo/antispoof
docker compose down

# 9.2 — убедиться, что 8090 снова свободен (Docker publish снят вместе с down)
ss -ltnp | grep ':8090' || echo "✅ порт свободен"

# 9.3 — поднять старый systemd-сервис обратно
sudo systemctl enable antispoof.service
sudo systemctl start antispoof.service
sudo systemctl status antispoof.service

# 9.4 — smoke-тест на старом пути (те же curl из шага 7, тот же порт 8090)
curl -sk http://127.0.0.1:8090/health
```

Откат безопасен в любой момент до шага 8 (пока `antispoof.service` не
задизейблен окончательно/замаскирован) — Docker и systemd-путь не могут
работать одновременно только из-за конфликта за порт 8090, но ничего не
удаляет и не портит другой путь.

---

## Контрольная сводка команд (короткая версия для повторного прогона)

```bash
# СТОП + зачистка (шаги 0-2)
sudo systemctl stop antispoof.service && sudo systemctl disable antispoof.service
pkill -f 'uvicorn app.main:app' || true; sleep 2; pkill -9 -f 'uvicorn app.main:app' || true
ss -ltnp | grep ':8090' && echo "❌ порт занят" || echo "✅ порт свободен"

# ЗАГРУЗКА ОБРАЗА + ЗАПУСК (шаги 4-6)
gunzip -c antispoof-1.0.0-cpu.tar.gz | docker load
cp .env.example .env   # заполнить SERVICE_TOKEN!
chmod 600 .env
docker compose up -d
docker compose ps

# SMOKE (шаг 7)
curl -s http://127.0.0.1:8090/health
```
