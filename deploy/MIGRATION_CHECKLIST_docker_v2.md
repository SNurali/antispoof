# Обновление antispoof: новые гейты качества кадра (v2)

Пошаговая инструкция для обновления уже развёрнутого Docker-стека `antispoof`
(`antispoof-app` + `antispoof-redis`, CPU-only, `127.0.0.1:8090` + X-Service-Token,
без mTLS — см. `deploy/MIGRATION_CHECKLIST_docker_v1.md`, если systemd→Docker
ещё не сделан на этом хосте, сначала пройти v1, потом этот документ).

**Что меняется:** три новых слоя-гейта перед passive-PAD (все bbox-независимые
или переиспользуют уже вычисленный bbox, без новых моделей/сетевых вызовов):
пропорция кадра (3:4/4:5 vs экранные 9:16), дедуп повторного кадра между
разными продажами, минимальное разрешение/вес файла. Подробный разбор —
`docs/plans/HANDOFF-2026-07-21-cross-transaction-face-reuse.md` (тот же коммит).

**⚠️ КОРРЕКТИРОВКА по итогам ревью 2PAC (2026-07-21):** в бой едут только
ДВА из трёх гейтов — пропорция кадра и разрешение/вес файла. **Дедуп
(`DEDUP_ENABLED`) ПОКА НЕ ВКЛЮЧАЕТСЯ** — см. §2 и §2.1 ниже, причина и
условие включения расписаны отдельно. Не путать с более ранней версией
рекомендации 50 CENT (включить все три) — она отменена этой правкой.

**Автор пакета:** 50 CENT (деплой), код — RZA, ворота — 2PAC. Проверено
локально (CPU-only Docker build + смоук ДВАЖДЫ — с исходным набором флагов
и повторно с финальным набором после правки 2PAC, см. §4 и приложение).
Собран, но **НЕ запушен и НЕ смёржен в main** — ждёт команды владельца
(Нурали).

---

## 0. Коммит/ветка — какую версию брать

Код на ветке `crew/antispoof-blur-angle-gate`, актуальный коммит
**`c849ec81ee95a1b24cb52e090ccdd8e16bb2bd87`** (3 коммита поверх текущего main,
включая ветку целиком):

- `c2f5f82` — blur (sharpness) + face-angle (pose) гейты — **оба остаются
  выключены**, см. §2.
- `bb4dafa` — гейт разрешения/веса файла + фикс бага маппинга verdict для
  blurry-метки.
- `c849ec8` — гейт по пропорции кадра (aspect-ratio) + усиленный дедуп + этот
  хендофф-документ.

### Вариант А — тянуть ветку напрямую (рекомендуется, быстрее)

Не требует мержа в main, разворачивает ровно то, что проверено локально:

```bash
cd /var/lib/mysql/logs/yoyo/antispoof   # рабочая директория деплоя (см. v1 §4)
git fetch origin
git checkout crew/antispoof-blur-angle-gate
git pull origin crew/antispoof-blur-angle-gate
git log -1 --oneline   # должен показать c849ec8 ...
```

### Вариант Б — через main (если владелец решит смёржить сначала)

**Мерж в main НЕ выполнен и не будет выполнен без явной команды Нурали** — это
осознанно необратимый шаг (меняет боевую ветку репозитория для всех). Если
команда придёт, тогда:

```bash
# на стороне 50 CENT/Нурали, ДО этого шага
git checkout main
git merge --no-ff crew/antispoof-blur-angle-gate
git push origin main

# на боевом хосте (egaz-02) — после того как main обновлён
cd /var/lib/mysql/logs/yoyo/antispoof
git fetch origin
git checkout main
git pull origin main
```

**Пока используем Вариант А.** Если к моменту разворота main уже обновлён —
проверить `git log -1 --oneline` совпадает с `c849ec8` (или более поздним
коммитом на main, включающим его) прежде чем продолжать.

---

## 1. Пересборка образа

```bash
cd /var/lib/mysql/logs/yoyo/antispoof
docker compose build antispoof
# либо, если тегируется отдельно от docker-compose.yml:
# docker build -t antispoof:1.1.0-cpu .
```

Локальная сборка (50 CENT, CPU-only, `python:3.11-slim`) прошла без ошибок —
слои `torch`/`torchvision`/`onnxruntime` кэшируются из предыдущей сборки, если
`requirements.txt` не менялся (не менялся в этих 3 коммитах).

---

## 2. Новые переменные окружения (`.env`) — что включить, что оставить выключенным

**⚠️ `.env.example` в репозитории НЕ обновлён под эти 3 коммита** (последняя
запись в нём — Layer 0a/0b, строка 125). Добавить блок ниже вручную в боевой
`.env` (не в `.env.example`, если не хотите отдельно чинить шаблон — это не
блокирует разворот).

### ВКЛЮЧИТЬ (GO от 2PAC, надёжные гейты, ложных блокировок живых не найдено):

```bash
# --- Layer 0g — гейт по пропорции кадра (aspect-ratio) -------------------
# Ловит фейк-фото 9:16 (экран/скрин), которое реальная камера телефона
# физически не снимает (снимает 3:4=0.75 или 4:5=0.80). Реальный
# подтверждённый фрод-образец real_fake_01.jpg — ровно 9:16 (0.5625).
ASPECT_RATIO_CHECK_ENABLED=true
ASPECT_RATIO_MIN=0.70
ASPECT_RATIO_MAX=0.85

# --- Гейт минимального разрешения/веса файла -----------------------------
# Отсекает пересжатые/telegram-preview-подобные кадры (низкое разрешение
# скрывает часть сигналов, на которые опираются другие гейты).
RESOLUTION_CHECK_ENABLED=true
MIN_IMAGE_MIN_SIDE_PX=700
MIN_IMAGE_MEGAPIXELS=0.55
MIN_IMAGE_BYTES=15360
```

### НЕ ВКЛЮЧАТЬ ПОКА (условие GO от 2PAC) — `DEDUP_ENABLED`

```bash
# --- Frame-reuse dedup — ПОКА ВЫКЛЮЧЕН, это НЕ default кода, это осознанное
# условие 2PAC для этого рcollout'а -------------------------------------
DEDUP_ENABLED=false
```

**Причина (обязательно довести до сведения Умида ака, буквально этими
словами):** если покупатель берёт несколько баллонов за одну продажу, одно и
то же `face_photo` может ЛЕГИТИМНО уйти в `/pad/check` несколько раз под
РАЗНЫМИ `transaction_ref` (разные `id_ballon`, тот же `id_request`) — сегодняшний
дедуп сравнивает именно `transaction_ref`, а не `id_request`, и заблокирует
честную продажу второго и последующих баллонов как `DUPLICATE_PHOTO`.

**Включать дедуп ТОЛЬКО после того, как Умид ака подтвердит семантику
multi-ballon.** Если подтвердится, что одно фото легитимно идёт на несколько
`id_ballon` в рамках одной продажи — правило дедупа надо привязать к
`id_request`, а не к `transaction_ref` (то есть дублирующим считать совпадение
фото ТОЛЬКО между разными `id_request`, не между `id_ballon` внутри одного
`id_request`) — это требует доработки на стороне antispoof (`app/dedup_store.py`)
и/или уточнения контракта полей с Laravel, **прежде** чем `DEDUP_ENABLED=true`
может безопасно уйти в прод. Пока этот вопрос не закрыт, `DEDUP_ENABLED`
остаётся `false` независимо от остальных двух гейтов.

Локально проверено (§4): при `DEDUP_ENABLED=false` то же самое фото под
вторым `transaction_ref` (моделирующее второй баллон той же продажи) проходит
как обычный `live`-запрос, НЕ блокируется — то есть выключенный флаг
действительно не создаёт риска для multi-ballon сценария сегодня.

### ОСТАВИТЬ ВЫКЛЮЧЕННЫМИ (это уже default в коде, но прописать явно в `.env`
для однозначности — если кто-то потом включит вручную по ошибке, будет видно
в диффе `.env`, что это отклонение от рекомендации):

```bash
# --- Blur/edge-sharpness (Layer 0c) — НЕ ВКЛЮЧАТЬ -------------------------
# Ложно резал живых на реальном фото при проверке (n=1 subject, синтетический
# blur-калибратор — см. app/config.py::FRAME_SHARPNESS_CHECK_ENABLED
# докстринг). Не включать без более широкой калибровки.
FRAME_SHARPNESS_CHECK_ENABLED=false

# --- Face-angle/pose (Layer 0d) — оставить выключенным -------------------
# Требует LIVENESS_ENDPOINTS_ENABLED=true (Phase 2, отдельно выключен на
# этом деплое — нет прод-GPU/провижининга весов) — без него это тихий
# no-op, не гейт. Оставить выключенным без риска что-то сломать.
POSE_CHECK_ENABLED=false
```

**Не трогать** (не входит в этот хендофф, остаются как есть): `DOCUMENT_CHECK_ENABLED`,
`EDGE_SHARPNESS_DIAGNOSTIC_ENABLED` (только диагностика, никогда не блокирует),
`REPLAY_PROTECTION_ENABLED`, `TRUST_PROXY_HEADERS`, `LIVENESS_ENDPOINTS_ENABLED`
— у всех свои отдельные rollout-гейты, описанные в `app/config.py`, не путать с
этим обновлением.

```bash
chmod 600 .env   # если менялось владельцем/правами — на всякий случай перепроверить
```

---

## 3. Перезапуск стека

```bash
docker compose up -d --no-deps antispoof
# redis не трогаем — эти 3 гейта его не используют
docker compose ps
# ожидаем antispoof-app: Up X seconds (healthy)
```

Если после `--no-deps` что-то не так (например, redis тоже пересоздался) —
можно просто `docker compose up -d` без флага, разница не критична здесь.

---

## 4. Смоук-тест ПОСЛЕ разворота (3 обязательные команды + 1 контрольная на dedup=off)

Нужны 2 тестовых файла на хосте:
- фейк 9:16 — тот же образец, что фигурирует в фрод-инциденте
  (`faces-dataset/real-fakes/real_fake_01.jpg` в этом репозитории, 720×1280,
  ~109КБ; если файла нет на хосте — любое фото 9:16 подойдёт для проверки
  самого гейта, вердикт/reason не зависят от содержимого лица);
- обычное фото 3:4 (любое реальное селфи с камеры, min-side ≥700px, ≥15KB).

```bash
# 4.1 — health
curl -s http://127.0.0.1:8090/health
# ожидаем: {"status":"healthy", ..., "models_loaded":true}

# 4.2 — фейк 9:16 -> ДОЛЖЕН вернуть low_quality/NON_CAMERA_GEOMETRY,
#       и до передачи в детектор лица (face_detected:false, быстро, <20ms)
python3 - <<'PY'
import base64, json
d = {
    "correlation_id": "smoke-v2-fake-1",
    "transaction_type": "sale",
    "transaction_ref": "smoke:v2:fake:1",
    "face_photo": base64.b64encode(open("faces-dataset/real-fakes/real_fake_01.jpg", "rb").read()).decode(),
}
json.dump(d, open("/tmp/smoke_fake.json", "w"))
PY
curl -s -X POST http://127.0.0.1:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: <ЗНАЧЕНИЕ_ИЗ_.env>" \
  --data-binary @/tmp/smoke_fake.json
# ожидаем: "verdict":"low_quality","reason":"NON_CAMERA_GEOMETRY"

# 4.3 — обычное фото 3:4 -> ДОЛЖНО пройти (verdict НЕ low_quality/NON_CAMERA_GEOMETRY,
#       НЕ spoof/DUPLICATE_PHOTO при первой отправке)
python3 - <<'PY'
import base64, json
d = {
    "correlation_id": "smoke-v2-real-1",
    "transaction_type": "sale",
    "transaction_ref": "smoke:v2:real:1",
    "face_photo": base64.b64encode(open("/путь/к/реальному/селфи_3x4.jpg", "rb").read()).decode(),
}
json.dump(d, open("/tmp/smoke_real.json", "w"))
PY
curl -s -X POST http://127.0.0.1:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: <ЗНАЧЕНИЕ_ИЗ_.env>" \
  --data-binary @/tmp/smoke_real.json
# ожидаем: "verdict":"live" (или "spoof" по НЕ-geometry причине, если фото реально
# спуф — но НЕ NON_CAMERA_GEOMETRY и НЕ DUPLICATE_PHOTO на первой отправке)

# 4.4 (КОНТРОЛЬНАЯ — dedup выключен, multi-ballon сценарий не должен
#      блокироваться) — тот же файл, ДРУГОЙ transaction_ref, повторно
python3 -c "
import json
d = json.load(open('/tmp/smoke_real.json'))
d['transaction_ref'] = 'smoke:v2:real:2'
d['correlation_id'] = 'smoke-v2-real-2'
json.dump(d, open('/tmp/smoke_real2.json', 'w'))
"
curl -s -X POST http://127.0.0.1:8090/pad/check \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: <ЗНАЧЕНИЕ_ИЗ_.env>" \
  --data-binary @/tmp/smoke_real2.json
# ожидаем: "verdict":"live" — НЕ "spoof"/"DUPLICATE_PHOTO". С DEDUP_ENABLED=false
# (см. §2.1) второй/третий баллон той же продажи с тем же фото ДОЛЖЕН
# проходить как обычный запрос. Если тут прилетел DUPLICATE_PHOTO — значит
# DEDUP_ENABLED в .env реально true, а не false, как договорено — остановиться
# и перепроверить .env ПЕРЕД тем, как считать обновление успешным (это
# заблокирует честные multi-ballon продажи в проде).
```

Локальный прогон 50 CENT на этом же коде (CPU-only Docker, `SESSION_STORE_
BACKEND=memory` для скорости смоука, финальный набор флагов после правки
2PAC — `DEDUP_ENABLED=false`) дал именно такой результат — см. приложение в
конце документа для точного JSON-вывода на сверку.

**Все 4 пункта должны пройти, прежде чем считать обновление успешным.**

---

## 5. Откат, если гейты начнут резать живых

Два уровня, от мягкого к жёсткому:

### 5.1 Мягкий откат — выключить конкретный флаг (без пересборки, без даунтайма)

```bash
# в .env поменять, например:
ASPECT_RATIO_CHECK_ENABLED=false
# (или RESOLUTION_CHECK_ENABLED=false — какой именно гейт режет живых,
# определить по signals.* в /pad/check-ответах отказов. DEDUP_ENABLED уже
# false по этому rollout'у, трогать не нужно — см. §2.1.)

docker compose up -d --no-deps antispoof   # перечитывает .env, пересоздаёт контейнер
docker compose ps   # снова healthy
```

Это самый быстрый и наименее рискованный откат — код не меняется, старый
образ не нужен, просто выключается конкретный гейт по конфигу.

### 5.2 Жёсткий откат — вернуться к предыдущему образу целиком

Если проблема не в конкретном флаге, а в самом коде (например, регрессия,
которую смоук не поймал):

```bash
cd /var/lib/mysql/logs/yoyo/antispoof
git log --oneline -5   # найти коммит ДО этого обновления
git checkout <предыдущий-коммит-или-тег>
docker compose build antispoof
docker compose up -d --no-deps antispoof
docker compose ps
curl -s http://127.0.0.1:8090/health   # health должен быть healthy на старом коде
```

Откат безопасен в любой момент — ни один из этих гейтов не трогает
существующие данные/схему БД Laravel, только вердикт самого `/pad/check`.

---

## 6. Рекомендация — последить первые 1-2 дня

После включения ASPECT_RATIO_CHECK_ENABLED/RESOLUTION_CHECK_ENABLED
**обязательно** последить долю отказов на реальном трафике покупателей:

- Смотреть `docker compose logs antispoof` (или audit-log, если `AUDIT_LOG_
  STDOUT=false`) на предмет всплеска `verdict != "live"` — особенно
  `NON_CAMERA_GEOMETRY` (может резать реальных клиентов, если клиентское
  приложение когда-то отправляет кадр не строго 3:4) и `low_quality` от
  RESOLUTION_CHECK (может резать клиентов на слабом канале/старом устройстве,
  если реальный кадр сжимается сильнее, чем предполагает калибровка).
- Калибровка обоих гейтов сделана на `faces-dataset/` (телеграм-скрейп) +
  1 реальном фрод-образце + клиентских исходниках (`FaceCaptureGeometry.kt`) —
  **НЕ** на живом трафике `egaz-02`. Если доля отказов реальных покупателей
  заметно выросла в первые дни — это сигнал вернуться к §5.1 и выключить
  конкретный флаг, не ждать, пока накопится жалоба.
- Если всё чисто — эти данные (реальная доля срабатывания на живом трафике)
  стоит передать назад RZA для донастройки порогов под реальные цифры, а не
  калибровочный датасет.
- **Отдельно, параллельно:** дождаться ответа Умида ака по multi-ballon
  семантике (§2.1) — как только подтверждено (или опровергнуто), решить
  отдельным заходом судьбу `DEDUP_ENABLED` (включать с привязкой к
  `id_request` вместо `transaction_ref`, либо оставить выключенным, если
  такого сценария нет). Это отдельный, самостоятельный шаг, не блокирует
  прод этих двух гейтов уже сейчас.

---

## Приложение — вывод локального смоука 50 CENT (2026-07-21, для сверки)

Собрано и прогнано локально ДВАЖДЫ (не на egaz-02 — доступа нет), `docker
build` из ветки `crew/antispoof-blur-angle-gate` @ `c849ec8`,
`SESSION_STORE_BACKEND=memory` (для локального смоука без redis-контейнера —
на самом деплое остаётся `redis`, это не меняет поведение этих гейтов).

### Прогон 1 — до правки 2PAC (флаги: ASPECT_RATIO/RESOLUTION/DEDUP всё `true`)

Приведён для истории — **этот набор флагов больше НЕ актуален**, DEDUP_ENABLED
отменён правкой 2PAC (см. врезку в начале документа и §2.1).

```
$ curl -s http://127.0.0.1:18090/health
{"status":"healthy","device":"cpu","gpu":"N/A","models_loaded":true,
 "model_version":"silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1",
 "liveness_endpoints_enabled":false,"liveness_models_loaded":false}

$ # фейк 9:16 (faces-dataset/real-fakes/real_fake_01.jpg, 720x1280)
{"verdict":"low_quality","reason":"NON_CAMERA_GEOMETRY","score":0.5625,
 "threshold":0.7,"face_detected":false,
 "signals":{"aspect_ratio_check":{"width":720,"height":1280,"ratio":0.5625,
 "min_ratio":0.7,"max_ratio":0.85}},"save_frame":false,
 "processing_ms":14.4}

$ # реальное фото 3:4 (960x1280, реальное селфи с телефона)
{"verdict":"live","reason":null,"score":0.8433,"threshold":0.5,
 "face_detected":true,
 "signals":{"signal_scores":{"fft":0.3,"lbp":0.0,"color":0.0,"moire":0.0,
 "sharpness":0.0,"jpeg":0.0,"recapture":0.3148},"spoof_probability":0.1567,
 "signals_triggered":["fft","recapture"],"nn_label":"spoof","nn_score":0.5542,
 "combined_label":"real","combined_score":0.8433,
 "image_phash":"bbcfc40952693b23"},"save_frame":false,
 "processing_ms":161.6}

$ # то же фото, другой transaction_ref, DEDUP_ENABLED=true (старый набор) —
$ # заблокировано. ЭТО ИМЕННО ТОТ РИСК, который снял 2PAC для multi-ballon:
{"verdict":"spoof","reason":"DUPLICATE_PHOTO","score":1.0,"threshold":0.875,
 "face_detected":false,
 "signals":{"dedup_check":{"phash_match":true,"hamming_distance":0,
 "matched_correlation_id":"smoke-...-smoke:real:1",
 "matched_transaction_ref":"smoke:real:1","matched_age_s":9.1},
 "image_phash":"bbcfc40952693b23"},"save_frame":true,
 "processing_ms":28.1}
```

### Прогон 2 — ФИНАЛЬНЫЙ набор флагов (после правки 2PAC, актуально для деплоя)

`ASPECT_RATIO_CHECK_ENABLED=true`, `RESOLUTION_CHECK_ENABLED=true`,
`DEDUP_ENABLED=false`, `FRAME_SHARPNESS_CHECK_ENABLED=false`,
`POSE_CHECK_ENABLED=false`.

```
$ curl -s http://127.0.0.1:18091/health
{"status":"healthy","device":"cpu","gpu":"N/A","models_loaded":true,
 "model_version":"silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1",
 "liveness_endpoints_enabled":false,"liveness_models_loaded":false}

$ # фейк 9:16 (faces-dataset/real-fakes/real_fake_01.jpg, 720x1280)
{"verdict":"low_quality","reason":"NON_CAMERA_GEOMETRY","score":0.5625,
 "threshold":0.7,"face_detected":false,
 "signals":{"aspect_ratio_check":{"width":720,"height":1280,"ratio":0.5625,
 "min_ratio":0.7,"max_ratio":0.85}},"save_frame":false,
 "processing_ms":14.4}

$ # реальное фото 3:4 (960x1280, реальное селфи с телефона)
{"verdict":"live","reason":null,"score":0.8433,"threshold":0.5,
 "face_detected":true,
 "signals":{"signal_scores":{"fft":0.3,"lbp":0.0,"color":0.0,"moire":0.0,
 "sharpness":0.0,"jpeg":0.0,"recapture":0.3148},"spoof_probability":0.1567,
 "signals_triggered":["fft","recapture"],"nn_label":"spoof","nn_score":0.5542,
 "combined_label":"real","combined_score":0.8433,
 "image_phash":"bbcfc40952693b23"},"save_frame":false,
 "processing_ms":156.1}

$ # то же фото, другой transaction_ref (эмулирует 2й баллон той же продажи),
$ # DEDUP_ENABLED=false -> проходит как обычный live, НЕ блокируется:
{"verdict":"live","reason":null,"score":0.8433,"threshold":0.5,
 "face_detected":true,
 "signals":{"signal_scores":{"fft":0.3,"lbp":0.0,"color":0.0,"moire":0.0,
 "sharpness":0.0,"jpeg":0.0,"recapture":0.3148},"spoof_probability":0.1567,
 "signals_triggered":["fft","recapture"],"nn_label":"spoof","nn_score":0.5542,
 "combined_label":"real","combined_score":0.8433,
 "image_phash":"bbcfc40952693b23"},"save_frame":false,
 "processing_ms":34.3}
```

Прогон 2 — это то, что реально едет в бой. Все 4 результата совпадают с
ожидаемыми из §4 — обновление готово к передаче.
