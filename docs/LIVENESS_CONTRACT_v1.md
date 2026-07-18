# E-GAZ — Финальный контракт `POST /liveness/challenge` + `POST /liveness/verdict`

**Автор:** RZA
**Дата:** 2026-07-17
**Статус:** ФАКТИЧЕСКИЙ контракт уже реализованного кода (не спека "как хотелось бы"). Извлечён напрямую из Pydantic-моделей `app/main.py`, `app/liveness_session.py`, `app/active_challenge.py`, `app/config.py` на текущем HEAD. Эндпоинты работают за фиче-флагом `LIVENESS_ENDPOINTS_ENABLED=False` — сейчас в проде выключены, ждут RzaLivenessGate на стороне Laravel.
**Запрошено:** Наташа (agent-mesh egaz-mobile, сообщение 1784298258990-0) — поля вердикта, форма challenge_spec, привязка к `correlation_id`.
**Контекст:** `docs/plans/FACEID_LIVENESS_ML_CORE_v1.md` (Layer 0/2/3), `docs/plans/FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md` §1.2 (Phase 2). Продолжение `docs/plans/FACEID_PHASE1_PAD_GATE.md` (`POST /pad/check`, Phase 1, уже в проде отдельно от этих двух эндпоинтов).

---

## 0. Общее для обоих эндпоинтов

- **Транспорт:** внутренние ML-периметр эндпоинты, тот же паттерн, что и `/pad/check` — вызываются Laravel'ем 127.0.0.1-only, НЕ клиент-facing напрямую.
- **Аутентификация:** заголовок `X-Service-Token`, сверяется с `settings.SERVICE_TOKEN`. Если `SERVICE_TOKEN` пуст — авторизация ВЫКЛЮЧЕНА (dev-режим, в проде так быть не должно — стартап логирует `WARNING`).
- **Готовность:** оба эндпоинта возвращают **`503`** с телом `{"detail": "Active-liveness endpoints are disabled or models failed to load"}` (НЕ форма `LivenessChallengeResponse`/`LivenessVerdictResponse`!), если `settings.LIVENESS_ENDPOINTS_ENABLED=False` ИЛИ модели (SCRFD-детектор ландмарок + AdaFace) не загрузились на старте. Laravel должен явно разбирать этот 503 отдельно от бизнес-вердиктов.
- **`model_version`** для этих двух эндпоинтов (не путать с `MODEL_VERSION` из `/pad/check`):
  ```
  silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1+scrfd-buffalo_l-det+landmark_3d_68+adaface-ir101-webface12m-onnx+active_challenge-turn_only_v1
  ```

---

## 1. `POST /liveness/challenge`

### Request — `LivenessChallengeRequest`

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `correlation_id` | `str` | да | UUID, минтит Laravel в `POST /liveness/start`. Эхается без изменений через `/liveness/challenge` → `/liveness/verdict`. Это **R2 SESSION-BINDING ключ** (см. §3) + id для трассировки логов. |
| `transaction_type` | `str` | да | **Passthrough ONLY** — в отличие от `/pad/check`, где `transaction_type` типизирован как `Literal["sale"]` и валидируется, здесь это свободная строка, сервис её не проверяет (сознательное решение — идентити/транзакционная семантика вне зоны этого сервиса). |
| `transaction_ref` | `str` | да | Натуральный ключ (`id_request:id_ballon`) — **passthrough only** для аудита/логов. Может ещё не быть финальным на момент challenge и **НЕ участвует** в session-binding проверке. |

```json
{
  "correlation_id": "7f3a1c9e-4b2d-4e11-9a6f-2c8d0e5b1f77",
  "transaction_type": "sale",
  "transaction_ref": "10245:88931"
}
```

### Response — `LivenessChallengeResponse`

| Поле | Тип | Описание |
|---|---|---|
| `session_id` | `str` | UUID4, минтит ЭТОТ сервис (`SessionStore.create`). |
| `challenge_spec.steps` | `list[str]` | Рандомизированный подмножество+порядок из пула. **Текущий пул: `["TURN_LEFT", "TURN_RIGHT"]`** (`LIVENESS_CHALLENGE_STEPS_POOL`), сэмплится `LIVENESS_CHALLENGE_STEP_COUNT=2` шага — при пуле из 2 элементов это **всегда оба шага, в случайном порядке** (2 перестановки = 1 бит энтропии против video-replay). `BLINK` в проде НЕ участвует (см. §4). |
| `challenge_spec.min_frames` | `int` | `4` (`LIVENESS_MIN_FRAMES`) |
| `challenge_spec.max_frames` | `int` | `6` (`LIVENESS_MAX_FRAMES`) |
| `t_instruction_shown` | `float` | Unix timestamp выдачи — для проверки тайминга окна на клиенте. |
| `expires_at` | `float` | Unix timestamp — `session_id` невалиден после этого момента. `TTL=90s` (`LIVENESS_SESSION_TTL_S`) — это **потолок с запасом на сеть/ретраи**, НЕ целевая длительность UX (целевая — 5-6с по ML_CORE §8, ещё не подтверждена владельцем). |
| `model_version` | `str` | см. §0. |

```json
{
  "session_id": "9d4e2a1b-6f30-4c8a-8e21-5b7c9a3d0e11",
  "challenge_spec": {
    "steps": ["TURN_RIGHT", "TURN_LEFT"],
    "min_frames": 4,
    "max_frames": 6
  },
  "t_instruction_shown": 1784298300.512,
  "expires_at": 1784298390.512,
  "model_version": "silentface-2.7_80x80_MiniFASNetV2+4_0_0_80x80_MiniFASNetV1SE+multisignal-v1+scrfd-buffalo_l-det+landmark_3d_68+adaface-ir101-webface12m-onnx+active_challenge-turn_only_v1"
}
```

**Важно:** `session_id` — единственный источник истины для `challenge_spec` этой сессии. `/liveness/verdict` смотрит `steps` по `session_id` из своего стора, а НЕ из того, что пришлёт клиент/Laravel повторно — подмена шагов в запросе на verdict невозможна.

---

## 2. `POST /liveness/verdict`

### Request — `LivenessVerdictRequest`

| Поле | Тип | Обязательное | Описание |
|---|---|---|---|
| `correlation_id` | `str` | да | Должен совпадать с `correlation_id`, под которым был создан `session_id` — **это и есть R2-проверка** (см. §3), сравнивается ИМЕННО это поле. |
| `session_id` | `str` | да | Из ответа `/liveness/challenge`. |
| `transaction_type` | `str` | да | Passthrough, не валидируется. |
| `transaction_ref` | `str` | да | Passthrough only — **не участвует** в binding-проверке. |
| `frames` | `list[LivenessFrame]` | да | См. ниже. |

`LivenessFrame`:

| Поле | Тип | Описание |
|---|---|---|
| `seq` | `int` (`>=0`) | Порядковый номер кадра. |
| `base64` | `str` | Base64 сырых байт JPEG/PNG (`base64.b64decode` → `cv2.imdecode`) — тот же формат, что `face_photo` в `/pad/check`. **Без `data:image/jpeg;base64,`-префикса** — с ним декод упадёт (см. ниже, деградирует per-frame, не роняет весь запрос). |
| `captured_at` | `str \| null` | Опционально, не участвует в вычислениях сейчас (нет проверки временного окна между кадрами на сегодня). |

**Валидация на уровне запроса (до бизнес-логики):**
- `1 <= len(frames) <= 6` (`LIVENESS_MAX_FRAMES`), иначе `HTTP 422` со стандартным FastAPI-телом ошибки (`{"detail": "..."}`) — **НЕ** форма `LivenessVerdictResponse`.
- `seq` не должны повторяться, иначе тоже `422`.
- **`LIVENESS_MIN_FRAMES=4` НЕ проверяется на этом уровне.** Он проверяется ПОСЛЕ Layer 0 QC — как количество кадров, ПРОШЕДШИХ QC (`assess_frame`: резкость/поза/окклюзия), а не количество присланных. Т.е. клиент технически может прислать 1 кадр и не получить 422, но с высокой вероятностью получит `verdict=incomplete, reason=LOW_QUALITY_FRAMES`. **Рекомендация Laravel/клиенту: всегда слать `min_frames..max_frames` кадров из `challenge_spec`, не меньше.**

```json
{
  "correlation_id": "7f3a1c9e-4b2d-4e11-9a6f-2c8d0e5b1f77",
  "session_id": "9d4e2a1b-6f30-4c8a-8e21-5b7c9a3d0e11",
  "transaction_type": "sale",
  "transaction_ref": "10245:88931",
  "frames": [
    {"seq": 0, "base64": "/9j/4AAQSkZJRgAB...", "captured_at": "2026-07-17T14:32:00.100Z"},
    {"seq": 1, "base64": "/9j/4AAQSkZJRgAB...", "captured_at": "2026-07-17T14:32:00.450Z"},
    {"seq": 2, "base64": "/9j/4AAQSkZJRgAB...", "captured_at": "2026-07-17T14:32:00.800Z"},
    {"seq": 3, "base64": "/9j/4AAQSkZJRgAB...", "captured_at": "2026-07-17T14:32:01.150Z"}
  ]
}
```

### Response — `LivenessVerdictResponse`

| Поле | Тип | Описание |
|---|---|---|
| `verdict` | `Literal["live","spoof","incomplete"]` | **Ровно 3 значения.** ⚠️ **BREAKING CHANGE (P0-4, 2026-07-18):** `low_quality` убран из схемы этого эндпоинта — он никогда фактически не выставлялся `_run_liveness_verdict` (все "недостаточно данных" случаи используют `incomplete`), это была мёртвая ветка enum'а. Убран ТОЛЬКО здесь — `/pad/check.verdict` по-прежнему включает `low_quality` и продолжает его реально выставлять, это другой эндпоинт с другой response-моделью. Партнёр (Наташа/Умид) должен быть предупреждён ДО того, как эта правка попадёт в прод, если где-то на их стороне уже жёстко ожидается 4-е значение. |
| `reason` | `str \| null` | **Детальная внутренняя причина.** Прямая цитата из кода: "этот ответ потребляет только Laravel, никогда не транслируется клиенту дословно — Laravel обязан замэппить в generic client-facing код/сообщение, чтобы атакующий не мог откалибровать обход по построчному фидбеку". См. полный список кодов в §2.1 ниже. |
| `score` | — | **В этом эндпоинте отдельного числового `score` НЕТ** (в отличие от `/pad/check.score`). Числовой сигнал — `frame_consistency_score` (ниже). |
| `frame_consistency_score` | `float` | Layer 3: минимальное попарное косинусное сходство эмбеддингов AdaFace между key-кадрами. Диапазон практически `[-1..1]`, для одного человека ожидается ближе к `[0.4..1.0]`. **`-1.0` = "не вычислялось"** (вердикт разрешился раньше Layer 3 — session-ошибка, geometry-гейт, LOW_QUALITY_FRAMES, провал Layer 2, TIMEOUT). |
| `best_frame_seq` | `int \| null` | **Только при `verdict="live"`.** Один из `seq` присланных кадров — сервис его НЕ пересэмплирует, только выбирает лучший из уже прошедших QC (фронтальность + резкость). Это кадр, который Laravel должен пересылать дальше (напр. в Adliya). |
| `session_id` | `str \| null` | Эхо `req.session_id` — эхается **всегда**, включая пути ошибок (`SESSION_NOT_FOUND` и т.п.), т.к. берётся прямо из запроса, не из найденной сессии. |
| `correlation_id` | `str \| null` | Эхо `req.correlation_id`, аналогично всегда. |
| `transaction_type` | `str \| null` | Эхо passthrough. |
| `transaction_ref` | `str \| null` | Эхо passthrough. |
| `model_version` | `str` | См. §0. |
| `processing_ms` | `float` | Серверное время обработки. |
| `signals` | `dict` | **Internal-only, не для показа клиенту.** Состав зависит от того, на каком слое остановился вердикт — см. §2.2. |

### 2.1 Полный список `reason` (по коду, ничего не придумано)

| `reason` | Когда | `verdict` |
|---|---|---|
| `SESSION_NOT_FOUND` | `session_id` не найден в сторе (не создавался / протух и был вычищен sweep'ом) | `incomplete` |
| `SESSION_EXPIRED` | `now > expires_at` | `incomplete` |
| `SESSION_ALREADY_USED` | Сессия уже была consumed другим вызовом `/liveness/verdict` (сессия одноразовая) | `incomplete` |
| `SESSION_CORRELATION_MISMATCH` | `correlation_id` запроса ≠ `correlation_id`, под которым была создана сессия — см. §3 | `incomplete` |
| `DOCUMENT_PHOTO` | Layer 0a geometry-гейт словил документ/паспортное фото хотя бы на одном кадре — валит ВСЮ сессию, приоритет выше даже проверки `LIVENESS_MIN_FRAMES` | `spoof` |
| `LOW_QUALITY_FRAMES` | Меньше 4 кадров прошли Layer 0 QC (резкость/поза/окклюзия) | `incomplete` |
| `ACTIVE_CHALLENGE_NOT_IMPLEMENTED` | Layer 2 вернул `UNSUPPORTED_STEP` (шаг не из `{TURN_LEFT, TURN_RIGHT, BLINK}`) — в бою недостижимо, т.к. пул шагов сервис сам генерирует из поддерживаемых | `incomplete` |
| `NO_FRONTAL_REFERENCE` | Ни один валидный кадр не попал в `|yaw| <= 10°` (`LIVENESS_YAW_FRONTAL_MAX_DEG`) — нет опорного фронтального кадра для оценки поворота | `incomplete` |
| `CHALLENGE_FAILED` | Layer 2: запрошенный `TURN_LEFT`/`TURN_RIGHT` не обнаружен ни в одном кадре (`STEP_NOT_DETECTED`) | `spoof` |
| `IDENTITY_SWAP_MID_SESSION` | Layer 3: `min_similarity < IDENTITY_MIN(0.40)` — подозрение на подмену лица посреди сессии | `spoof` |
| `PASSIVE_PAD_SPOOF` | Layer 1 (переиспользованный passive-PAD движок из `/pad/check`): хотя бы один валидный кадр размечен как `spoof` (агрегация `any_frame_spoof`, консервативная — см. §4) | `spoof` |
| `TIMEOUT` | Обработка превысила `LIVENESS_INFERENCE_TIMEOUT_S=8.0s` | `incomplete` |
| `INTERNAL_ERROR` | Необработанное исключение — fail-closed (см. §2.3, там есть нюанс формы ответа) | `incomplete` |
| `null` | Всё прошло — `verdict="live"` | `live` |

### 2.2 Состав `signals` по слоям (накопительно)

- `layer0_frame_qc`: `{seq: {"valid": bool, "reason": str, "metrics": {...}}}` — на КАЖДОМ пути, где успели декодироваться кадры.
- `layer0a_geometry_check`: только при `DOCUMENT_PHOTO` — `{seq: {"face_area_ratio", "face_width_ratio", "frame_aspect_ratio"}}`.
- `layer2_active_challenge`: `{"passed", "reason", "detail", "requested_steps"}` — появляется, если дошли до Layer 2.
- `layer3_identity_consistency`: `{"passed", "min_similarity", "reference_seq", "pairwise", "threshold"}` — появляется, если дошли до Layer 3.
- `layer1_passive_pad`: `{"frames": [{"seq","label","score"}], "aggregate": "any_frame_spoof"}` — появляется, если дошли до Layer 1.
- При `SESSION_CORRELATION_MISMATCH`: `{"session_correlation_id": "...", "request_correlation_id": "..."}` — оба значения для дебага несовпадения.
- Внутреннее поле `_n_valid` (счётчик валидных кадров) существует только в аудит-логе, **из внешнего `signals` вырезается** перед отправкой ответа — в теле HTTP-ответа его не будет.

### 2.3 `INTERNAL_ERROR` (fail-closed путь) — ИСПРАВЛЕНО (P0-1, 2026-07-18)

**Обновление 2026-07-18:** три расхождения, задокументированные здесь ранее, исправлены. Теперь два разных пути отдают `INTERNAL_ERROR`, в зависимости от того, успел ли распарситься запрос:

**А) Тело запроса УЖЕ распарсено** (обычный случай — исключение произошло где-то в обработке кадров/моделей внутри `liveness_verdict()`): локальный `try/except` в самом роуте (`app/main.py::liveness_verdict`, НЕ глобальный handler) ловит исключение, когда `req` уже в скоупе:

```json
{
  "verdict": "incomplete",
  "reason": "INTERNAL_ERROR",
  "model_version": "<полный LIVENESS_MODEL_VERSION из §0>",
  "frame_consistency_score": -1.0,
  "best_frame_seq": null,
  "session_id": "<эхо req.session_id>",
  "correlation_id": "<эхо req.correlation_id>",
  "transaction_type": "<эхо req.transaction_type>",
  "transaction_ref": "<эхо req.transaction_ref>",
  "processing_ms": <реальное время обработки>,
  "signals": {}
}
```

Полный `LIVENESS_MODEL_VERSION`, документированный `-1.0` (не вычислялось), и настоящее эхо всех 4 полей — форма совпадает с таблицей выше по всем полям, кроме `signals={}` (Layer 0/2/3 не успели отработать).

**Б) Тело запроса ЕЩЁ НЕ распарсилось** (редкий случай — напр. malformed JSON, отказ до валидации Pydantic-модели): глобальный `app.exception_handler(Exception)` (`app/main.py::global_exception_handler`) остаётся fallback'ом для этого случая — `model_version`/`frame_consistency_score` тоже исправлены (полный `LIVENESS_MODEL_VERSION`, `-1.0`), но `session_id`/`correlation_id`/`transaction_type`/`transaction_ref` там честно `null`, т.к. `req` физически не существует в этой точке (не баг, а единственно возможное поведение).

Оба пути покрыты тестом `tests/test_liveness_endpoints.py::TestLivenessVerdictInternalError`.

---

## 3. Привязка `correlation_id` (то, что конкретно запросила Наташа)

**Подтверждаю явно: `correlation_id`, НЕ `sale_ref`/`transaction_ref`.**

- `correlation_id` **минтится Laravel** в `POST /liveness/start` (публичный клиент-facing эндпоинт, вне зоны этого сервиса).
- Laravel передаёт его в `POST /liveness/challenge.correlation_id` при создании сессии — сервис сохраняет его в `ChallengeSession.correlation_id`.
- Laravel передаёт тот же `correlation_id` в `POST /liveness/verdict.correlation_id`.
- Сервис на входе `/liveness/verdict` сравнивает `session.correlation_id == req.correlation_id` (`app/main.py::_run_liveness_verdict`, сразу после успешного `session_store.consume(session_id)`).
- **При несовпадении:** `verdict="incomplete"`, `reason="SESSION_CORRELATION_MISMATCH"`, `signals` содержит оба значения (`session_correlation_id` — то, с чем сессия создавалась, `request_correlation_id` — то, что пришло в verdict-запросе) для дебага на стороне Laravel/аудита.
- `transaction_ref` (натуральный ключ `id_request:id_ballon`) **сознательно исключён** из binding-проверки — комментарий в коде объясняет почему: он может легитимно быть ещё не финальным на момент выдачи challenge (ссылка на продажу может подтверждаться позже, чем стартует liveness-проверка), поэтому хранится только как passthrough для аудита/логов, как и `transaction_type`.

**Обновление 2026-07-18 (P0-5):** добавлен реальный сквозной тест на уровне HTTP-контракта сервиса — `tests/test_liveness_endpoints.py::TestVerdictEndpoint::test_correlation_id_mismatch_rejected`: реальный вызов `POST /liveness/challenge` с `correlation_id=A` → реальный вызов `POST /liveness/verdict` с намеренно другим `correlation_id=B` → проверяется `verdict="incomplete"`, `reason="SESSION_CORRELATION_MISMATCH"`, и что ОБА значения (A и B) попадают в `signals`. Это честный прогон настоящего пути кода этого сервиса (не мок session_store), но это **всё ещё не сквозной тест против реального Laravel** — приведённая ниже оговорка из кода остаётся в силе в части "не проверено против реального вызова Laravel", закрыта только часть "логика не гонялась даже интеграционным тестом внутри сервиса".

**⚠️ Честная оговорка из самого кода** (`app/main.py`, комментарий у проверки): эта привязка помечена как *"Soft in this increment — logged and rejected, but not yet proven against a real Laravel integration test"* — т.е. логика реализована и теперь покрыта интеграционным тестом на уровне сервиса (см. выше), но **всё ещё не была прогнана end-to-end против реального вызова Laravel**. Рекомендую первым делом после подключения `RzaLivenessGate` прогнать именно этот кейс (намеренно рассинхронизированный `correlation_id` между `/liveness/start`→`/liveness/challenge` и `/liveness/verify`→`/liveness/verdict`) как настоящий межсервисный интеграционный тест против Laravel, а не полагаться на этот внутрисервисный тест.

---

## 4. Известные ограничения — то, что НЕ стоит закладывать в план как готовое

1. **`BLINK`-шаг реализован механически, но НЕ в боевом пуле.** EAR-индексация (36-41/42-47 точек) подтверждена на реальном фото, но `LIVENESS_EAR_BLINK_MAX=0.20` (порог "закрытый глаз") — литературная константа, НЕ откалиброванная на реальных морганиях этого домена (камера/освещение E-GAZ). Собственная sanity-проверка на 3 реальных ОТКРЫТЫХ глазах нашла EAR=0.214 у одного из них — на 0.014 выше порога закрытого глаза. Это тревожный сигнал, что порог слишком близко к шумовому полу модели, а не косметика. Пул шагов сегодня: `TURN_LEFT, TURN_RIGHT` только — **не добавлять `BLINK` в прод без реального корпуса открытый+закрытый глаз** (нужен сбор: несколько человек, реальное моргание, продовое освещение).
2. **`IDENTITY_MIN=0.40` (Layer 3) НЕ откалиброван на реальных данных.** Попытка калибровки на `docs/plans/calibration/incident_urgut` провалилась — датасет оказался одиночными фото РАЗНЫХ людей, а не multi-frame сессиями одного человека (bonafide-bonafide cosine там near-zero, mean=0.058 — не тот вопрос, на который датасет может ответить). Значение оставлено на литературном "полу" из ML_CORE, это **placeholder, не проверенный security-порог**. Нужен реальный multi-frame корпус одной и той же сессии одного человека для честной калибровки.
3. **Latency AdaFace IR-101 на CPU: ~342мс/кадр (тёплый прогон) — ~524мс/кадр (холодный/единичный вызов)**, измерено на i5-11400, 12 потоков. При 4-6 key-кадрах это ~1.4-3.1с ТОЛЬКО на эмбеддинги, плюс SCRFD-детекция (~100-160мс/кадр) плюс Layer 1 passive-PAD (~20мс/кадр). `LIVENESS_INFERENCE_TIMEOUT_S=8.0` — это **стопгэп-потолок под измеренный худший случай, а не принятый UX-бюджет**. Целевые 5-6с полного клиентского флоу (ML_CORE §8) архитектурно под риском при текущей CPU-модели — IR-18/IR-50 чекпоинт легче не существует в репо сегодня, GPU-перенос не сделан. Это открытый архитектурный риск, не решённая задача.
4. **Geometry-гейт (Layer 0a) подключён** и переиспользован без изменений из `/pad/check` (`app/geometry_check.py`) — но его собственная калибровка построена на **n=1 образце спуф-атаки** (см. комментарий в `app/main.py` про `face_width_ratio`), эвадируем более умной атакой, которая не заполняет весь кадр лицом. Работает как есть, доверять как окончательному порогу нельзя.
5. **Формальные PAD-метрики (APCER/BPCER по ISO/IEC 30107-3) для ЭТОГО multi-frame `/liveness/*`-пайплайна НЕ измерены.** Есть только цифры Phase 1 `/pad/check` (single-frame, APCER=0%/BPCER=0% на 21 фото инцидента `incident_urgut`) — они относятся к другому, более простому эндпоинту и НЕ переносятся автоматически на связку Layer 0/2/3 здесь. Прежде чем считать этот контракт production-ready по качеству распознавания (не по форме API), нужен отдельный прогон на реальном multi-frame bonafide+spoof корпусе.
6. **Sign-convention `TURN_LEFT`/`TURN_RIGHT` не подтверждён на реальном устройстве.** Предположение "положительный yaw = поворот в правую сторону от зрителя" — стандартная конвенция для этой pose-модели, но не проверена на размеченной реальной съёмке "повернись налево по команде" с устройства E-GAZ. Если знак перепутан, безопасное свойство (реальный поворот произошёл) сохраняется, но текст инструкции на экране может быть перепутан местами (UX-баг, не дыра в безопасности).
7. **~~Сессии — только in-memory, в одном процессе~~ — ЗАКРЫТО (JAY-Z, 2026-07-18), при условии что бэкенд реально включён.** `app/liveness_session.py` теперь имеет два бэкенда за одним и тем же интерфейсом (`create`/`get`/`consume`): `SessionStore` (in-memory, дефолт, один процесс) и `RedisSessionStore` (Redis, шарится между любым числом воркеров/реплик). Переключение — явный `SESSION_STORE_BACKEND=memory|redis` + `REDIS_URL` (`app/config.py`), без тихого фолбэка: если `SESSION_STORE_BACKEND=redis`, а Redis недоступен на старте — сервис падает с `RuntimeError`, а не тихо продолжает на in-memory. Атомарность `consume()` (fetch+mark-used) между процессами — Lua-скрипт (`EVAL`), а не клиентский `threading.Lock` (тот работал только в одном процессе). Guard на `WEB_CONCURRENCY>1` в `app/main.py` теперь пропускает multi-worker деплой ТОЛЬКО если `SESSION_STORE_BACKEND=redis`, иначе всё ещё падает на старте, как раньше. **Условие, чтобы риск был реально закрыт в проде:** 50 CENT должен поднять сам Redis-инстанс на egaz-02.uz и выставить `SESSION_STORE_BACKEND=redis`/`REDIS_URL` в `antispoof.service` — до этого прод по-прежнему на in-memory-бэкенде и ограничение №7 действует буквально как было. Тесты: `tests/test_redis_session_store.py` (12 тестов, включая cross-thread атомарность и pытие через `fakeredis[lua]`) + `tests/test_liveness_endpoints.py` прогнан живьём против реального `redis-server` (не только fakeredis) — оба набора зелёные.

---

## 5. Что нужно от владельца, прежде чем считать контракт production-ready

Это не блокер для СВЕРКИ формы контракта с Наташей (форма API стабильна и её можно подключать), но блокер для доверия к качеству вердикта:

1. **Датасет для калибровки Layer 3 (`IDENTITY_MIN`)** — multi-frame сессии одного и того же человека (не одиночные фото), в идеале с реальных E-GAZ-камер/условий.
2. **Датасет для калибровки `BLINK`** (если решено включать в пул) — несколько человек, реальное моргание, продовое освещение.
3. **GPU vs лёгкая модель для AdaFace** — решение владельца: либо найти/обучить IR-18/IR-50 чекпоинт под CPU-бюджет, либо выделить GPU этому сервису под latency-цель 5-6с.
4. **Реальный multi-frame bonafide+spoof корпус** для формального APCER/BPCER этого пайплайна (не переиспользовать цифры Phase 1).
5. ~~**Redis для session store**, если горизонтальное масштабирование стоит в плане раскатки.~~ Код готов (см. §4 п.7) — от владельца/50 CENT нужно только фактически поднять Redis-инстанс на egaz-02.uz и выставить `SESSION_STORE_BACKEND=redis` + `REDIS_URL` в `antispoof.service`, после чего можно снимать guard `WEB_CONCURRENCY=1` и раскатывать >1 воркер.
