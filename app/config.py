"""Pydantic settings loaded from environment variables."""

import logging
from pathlib import Path
from typing import Annotated, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration from env vars."""

    LIVENESS_THRESHOLD: float = 0.5
    HOST: str = "0.0.0.0"
    PORT: int = 8090
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "models"
    DEVICE: str = "auto"
    MAX_BATCH: int = 16

    # Deploy environment marker (P0-3, 2026-07-18). "dev" (default) keeps the
    # existing dev-mode warning-only behavior for an empty SERVICE_TOKEN.
    # "prod" makes an empty SERVICE_TOKEN a hard startup failure — see
    # app/main.py's SERVICE_TOKEN check right after Settings() is constructed.
    # 50 CENT must set ENVIRONMENT=prod in antispoof.service on egaz-02.uz
    # BEFORE this change is deployed, or the service will refuse to start.
    ENVIRONMENT: str = "dev"

    # Reverse-proxy topology (BUSTA RHYMES, deploy/mtls/, 2026-07-18).
    # DEFAULT FALSE: today uvicorn listens directly on HOST:PORT (0.0.0.0 by
    # default) and request.client.host in app/main.py's IP allowlist is the
    # real caller's address. Once deploy/mtls/nginx-antispoof-mtls.conf is
    # rolled out (nginx :443 TLS+mTLS -> uvicorn on 127.0.0.1 only, external
    # access to PORT blocked by firewall), EVERY external caller's
    # request.client.host becomes nginx's own loopback address — the
    # allowlist would silently stop filtering anyone (127.0.0.0/8 is itself
    # allowed). Set this to true ONLY when that nginx topology is actually
    # live, so the allowlist reads the real client IP from X-Forwarded-For
    # instead — see app/main.py::_effective_client_ip for the trust logic
    # (X-Forwarded-For is only honored when the physical TCP peer is
    # loopback; a direct, non-loopback connection — i.e. nginx bypassed —
    # never trusts the header). 50 CENT flips this on at the same time the
    # nginx config is deployed, not before.
    TRUST_PROXY_HEADERS: bool = False

    # Phase 1 PAD-gate integration (BACKEND_REQUIREMENTS_2026-07-06)
    SERVICE_TOKEN: str = ""  # X-Service-Token shared secret with Laravel; empty = auth disabled
    RATE_LIMIT_BURST: int = 20  # max concurrent requests (per-second burst)
    RATE_LIMIT_SUSTAINED: float = 5.0  # sustained requests per second
    SAVE_FRAME_VERDICTS: str = "spoof"  # comma-separated verdicts that trigger save_frame=true

    # Anti-replay timestamp window (KENDRICK security analysis, 2026-07-18;
    # wired in alongside BUSTA RHYMES's mTLS transport layer, deploy/mtls/).
    # mTLS authenticates the CHANNEL ("who is talking") but does not stop a
    # captured request from being replayed verbatim within its validity
    # window — this is a deliberately lightweight control layered ON TOP of
    # the existing X-Service-Token + IP-allowlist (both UNCHANGED by this),
    # requested explicitly in lieu of a nonce-store/Redis-dedup: the client
    # sends X-Request-Timestamp (unix seconds, current time at request
    # creation) and the server rejects anything outside
    # +/-REPLAY_TOLERANCE_S of its own clock. This does NOT detect a replay
    # of the SAME request within the window — it only bounds how long a
    # captured request stays usable, trading precision for zero new
    # infrastructure. Applies to the three money-path endpoints only
    # (/pad/check, /liveness/challenge, /liveness/verdict) — see
    # app/main.py::_verify_replay_protection and its call sites.
    #
    # DEFAULT DISABLED: the partner (Laravel/Umid's team) must start sending
    # X-Request-Timestamp on every money-path call FIRST — flipping this on
    # before they do turns every one of their existing requests into a 401.
    # 50 CENT flips this on only after that is confirmed live, same rollout
    # pattern as TRUST_PROXY_HEADERS above.
    #
    # DELIBERATELY LEFT FALSE (RZA, 2026-07-20 fraud-incident hardening
    # pass): re-checked this specific flag against the most recent status —
    # docs/plans/HANDOFF-2026-07-18-egaz2-mtls-staging.md (2 days before this
    # pass) states plainly that egaz-02.uz DNS still does not resolve, mTLS
    # is NOT deployed, and the partner has NOT yet been sent even the
    # staging URL/token, let alone confirmed sending X-Request-Timestamp.
    # Flipping this default to True now would 401 every legitimate
    # money-path request the moment this code reaches prod — the exact
    # outage this rollout gate exists to prevent. The mechanism itself is
    # fully implemented and tested (tests/test_replay_protection.py, 218
    # green as of the 07-18 handoff, unaffected by this pass) — only the
    # DEFAULT is intentionally unchanged pending the partner's confirmation
    # this handoff document says is still outstanding.
    REPLAY_PROTECTION_ENABLED: bool = False
    # Clock-skew + network/retry tolerance, in seconds. 120s is deliberately
    # generous — wide enough to absorb NTP drift and a slow mobile network
    # retry, narrow enough that a captured request stops being replayable
    # within ~2 minutes. Configurable via env, not hardcoded, so it can be
    # tightened once real production round-trip-time data exists.
    REPLAY_TOLERANCE_S: int = 120

    # Layer 0 — document/passport-photo pre-filter (RZA, 2026-07-16).
    # DEFAULT DISABLED: calibration on incident_urgut (n=1 spoof / n=12 bonafide,
    # see app/document_check.py module docstring for full numbers) found 2 of 4
    # tested bonafide with a plain/simple background (indoor wall; painted wood
    # door) get flagged as a studio/document background by minicpm-v — an
    # ordinary home-selfie scenario, not a rare edge case. Latency (~50-90s/call
    # observed, worse under shared-GPU contention) is also far beyond the
    # existing /pad/check budget (INFERENCE_TIMEOUT_S=2s). Do not enable without
    # a larger calibration pass first.
    DOCUMENT_CHECK_ENABLED: bool = False
    # Vision model tag (Ollama). Kept configurable — a newer minicpm-v4.6 is
    # under parallel evaluation as of 2026-07-16; switching should be an env
    # change, not a code change.
    DOCUMENT_CHECK_MODEL: str = "minicpm-v:latest"
    DOCUMENT_CHECK_OLLAMA_URL: str = "http://127.0.0.1:11434/api/generate"
    # Confidence in "this is a document/studio photo" (0..1) required to
    # short-circuit to verdict=spoof BEFORE passive-PAD runs.
    DOCUMENT_REJECT_THRESHOLD: float = 0.70
    # Per-call timeout for the Ollama HTTP request. Observed single-flight
    # latency in testing was 50-90s (no GPU contention) — 20s is NOT long
    # enough to reliably complete a call under normal conditions on this
    # hardware; kept configurable so it can be raised, or the layer left
    # disabled, without a code change.
    DOCUMENT_CHECK_TIMEOUT_S: float = 20.0

    # Layer 0a — deterministic face-to-frame geometry gate (RZA, 2026-07-16,
    # RE-CALIBRATED 2026-07-16 evening after a second real incident slipped
    # through the original 0.35 threshold — see app/geometry_check.py module
    # docstring for the full numbers). Reuses the SAME RetinaFace bbox
    # passive-PAD already computes — no extra model, no network call,
    # microseconds.
    #
    # Calibrated on incident_urgut + the 2026-07-16 19:41 incident photo, all
    # with the REAL FaceDetector bbox:
    #   bonafide (12 files):     face_area_ratio 0.043-0.215
    #   document spoof (n=2):    face_area_ratio 0.322/0.472
    # 0.27 sits ~26% above the bonafide area-ratio max and ~16% below the
    # weaker of the two known spoofs (was 0.35 — too high, let the 0.3224
    # incident through). This is the ONLY ratio wired into the reject
    # decision (see app/main.py::_run_geometry_gate). DEFAULT ENABLED: unlike
    # the minicpm-v layer this is free (no latency/availability risk) — but
    # calibration is still n=12 bonafide / n=2 document-spoof phone photos,
    # NOT verified sale-transaction camera frames; production camera may sit
    # closer to the customer and shift the bonafide baseline upward. Re-check
    # against real sale-flow frames before trusting the FRR this implies.
    GEOMETRY_CHECK_ENABLED: bool = True
    FACE_RATIO_REJECT: float = 0.27

    # DIAGNOSTIC ONLY — NOT wired into the reject decision (2PAC review,
    # 2026-07-16: `face_width_ratio` is empirically ~1.09*sqrt(face_area_ratio)
    # on every one of the 14 calibration samples measured so far, so gating
    # on width in addition to area does not catch any attack area misses —
    # it only tightens the effective margin against a real customer standing
    # close to the camera, i.e. pure FRR cost with no FAR benefit on current
    # data). `face_width_ratio` is still computed and reported in
    # `signals.geometry_check` for every request (like `frame_aspect_ratio`)
    # so a future, larger, independently-collected sample can re-evaluate it
    # as a genuinely orthogonal signal. This constant is kept only so that
    # value is available if/when that recalibration wires it back in — do
    # not read it as an active threshold today.
    FACE_WIDTH_RATIO_REJECT: float = 0.55

    # Layer 0c — deterministic frame-sharpness gate (RZA, 2026-07-21). See
    # app/blur_check.py module docstring for the full rationale (a printed/
    # screen photo held at an angle AND deliberately motion-blurred was
    # observed passing /pad/check as verdict=live) and calibration numbers.
    # Dependency-free like GEOMETRY_CHECK_ENABLED (reuses the RetinaFace bbox
    # already computed, pure OpenCV, no extra model/network call) — but
    # DEFAULT DISABLED, UNLIKE that gate: GEOMETRY_CHECK_ENABLED had n=14
    # calibration samples across bonafide+2 spoof profiles before defaulting
    # on; this gate has n=1 subject / n=8 bona fide frames from ONE staged
    # capture, and the blur values it was checked against are SYNTHETIC
    # (motion-blur kernel applied in this repo), not the real production
    # attack photo. Also: this repo's existing /pad/check + /spoof-server
    # test suites (tests/test_pad_check.py, test_spoof_server.py,
    # test_dedup.py) reuse a flat-color synthetic circle image across many
    # tests that measures well below MIN_FACE_SHARPNESS_224 — the SAME
    # "would break the existing test suite's shared fixture on an unrelated
    # PR" reasoning DEDUP_ENABLED below is held to. Recommend: enable after
    # this report is reviewed AND after collecting a broader bona fide
    # corpus (n>1 subject) to check MIN_FACE_SHARPNESS_224 against, and
    # after fixing the synthetic test fixtures — same "не занижай FAR" bar,
    # applied to not silently raising FRR without real data behind it either.
    FRAME_SHARPNESS_CHECK_ENABLED: bool = False
    # Laplacian variance floor on a 224x224 resize of the face bbox crop
    # (same crop scale app/multisignal.py::recapture_spoof_score already
    # uses). Below this -> reject as low_quality/BLURRY, before passive-PAD
    # runs. See app/blur_check.py docstring for the 93.0 (sharp bona fide
    # floor) vs 25.6-89.4 (same frames, synthetic 9px motion blur) numbers
    # this sits between.
    MIN_FACE_SHARPNESS_224: float = 60.0

    # Layer 0d — face-angle (yaw/pitch) gate (RZA, 2026-07-21). See
    # app/pose_check.py module docstring for the full rationale and
    # calibration numbers (s001, n=1 subject).
    # DEFAULT DISABLED, unlike the sharpness gate above: this one requires
    # LandmarkDetector (SCRFD + landmark_3d_68, insightface) which is ONLY
    # loaded when LIVENESS_ENDPOINTS_ENABLED=True — flipping this on without
    # that flag (and without buffalo_l weights actually provisioned on the
    # host) is a silent no-op, not a security control; see app/main.py::
    # _run_pose_gate for the exact guard. Even with both flags true, this
    # adds ~100-160ms CPU (a second detector pass) inside /pad/check's
    # existing 2.0s INFERENCE_TIMEOUT_S budget — confirm real p95 latency on
    # egaz-02 before enabling in prod.
    POSE_CHECK_ENABLED: bool = False
    # Max |yaw| degrees before a frame is rejected as off-angle. s001 bona
    # fide "30-degree" turns measured ~32-33 actual — 40.0 leaves ~7 degrees
    # margin, THIN on n=1 subject. See app/pose_check.py docstring.
    POSE_YAW_REJECT_DEG: float = 40.0
    # Max |pitch| degrees before a frame is rejected as off-angle. s001 bona
    # fide up/down tilts measured ~35-37 actual — 45.0 leaves ~8 degrees
    # margin for an ordinary checkout glance. See app/pose_check.py.
    POSE_PITCH_REJECT_DEG: float = 45.0

    # CORS (MF DOOM review, 2026-07-16): DEFAULT EMPTY = middleware not
    # attached at all — no CORS headers, no wildcard attack surface. The
    # real production caller is server-side Laravel (not a browser), so CORS
    # is not needed in prod. Set to a list of origins (e.g.
    # ["http://127.0.0.1:8090"]) only for local manual browser testing via
    # testpage/.
    CORS_ALLOW_ORIGINS: Annotated[list[str], NoDecode] = []

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """Accept a plain comma-separated string (e.g. '*' or
        'http://a,http://b') from env, so ctl.sh/env files can set one or
        more origins without JSON-quoting them. NoDecode above stops
        pydantic-settings from trying (and failing) to json.loads() the raw
        env string before this validator ever runs."""
        if isinstance(v, str):
            stripped = v.strip()
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return v

    # ------------------------------------------------------------------
    # Active liveness (RZA, 2026-07-17) — POST /liveness/challenge +
    # POST /liveness/verdict, per docs/plans/FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md
    # Phase 2 and docs/plans/FACEID_LIVENESS_ML_CORE_v1.md Layer 0/2/3.
    #
    # DEFAULT DISABLED. Flipping this on pulls in NEW heavy dependencies
    # (insightface, onnxruntime) and a 260MB ONNX weight file
    # (models/liveness/adaface_ir101_webface12m.onnx) that are not
    # guaranteed to be installed/provisioned on every deploy yet. When
    # False, app/main.py must not import insightface/onnxruntime or touch
    # the weight file at all at startup — /verify, /verify_batch,
    # /spoof-server, /pad/check must keep working unmodified on a host
    # that has not rolled this out.
    LIVENESS_ENDPOINTS_ENABLED: bool = False

    # SCRFD detector input size (insightface buffalo_l, detection +
    # landmark_3d_68 submodules only — NOT the full buffalo_l pipeline, no
    # landmark_2d_106/genderage/recognition load, see app/face_landmarks.py).
    # 320 measured on docs/plans/calibration/incident_urgut (12 bonafide +
    # 12 spoof, single-photo, NOT session frames): ~100-160ms/frame on
    # i5-11400 CPU after model load. See app/face_landmarks.py docstring.
    LIVENESS_DET_SIZE: int = 320

    # AdaFace embedder for Layer 3 cross-frame identity consistency.
    # MEASURED, NOT the model ML_CORE §2.3/§7 recommended: that document
    # explicitly argues for AdaFace IR-18 or IR-50 over IR-101 on CPU
    # latency grounds, but no IR-18/IR-50 checkpoint exists in this repo or
    # the sibling face_id/tracker project today — only the IR-101 ONNX
    # (adaface_ir101_webface12m.onnx, reused from face_id/tracker/weights/,
    # already CPU-turbo-proven there for a DIFFERENT latency budget: a
    # live door-camera stream, not a <2s synchronous HTTP call). Measured
    # HERE on this repo's calibration set (2026-07-17, i5-11400, 12
    # threads, onnxruntime CPUExecutionProvider):
    #   steady-state (warm session, explicit intra_op_num_threads=12):
    #       ~342ms/frame
    #   cold/mixed average across 24 single-shot calls (no warmup
    #       amortization — closer to what an idle low-QPS internal service
    #       actually pays per call): ~524ms/frame
    # At 4-6 key frames/session this is ~1.4-3.1s for embeddings ALONE,
    # before SCRFD detection (~100-160ms/frame) or the existing Layer 1
    # passive-PAD pass (~20ms/frame, already measured in
    # FACEID_PHASE1_PAD_GATE.md §3). Do not treat this as a finished
    # decision — it is the SAME risk ML_CORE flagged, now with real
    # numbers instead of a guess. Swapping to IR-18/50 (or moving this
    # service to GPU) is the clear next step once a lighter checkpoint is
    # available; this constant exists so that swap is an env change, not a
    # code change.
    ADAFACE_ONNX_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "liveness" / "adaface_ir101_webface12m.onnx"

    # Layer 3 cross-frame identity — UNCALIBRATED. ML_CORE §2.3 cites a
    # literature-only working hypothesis (cosine >= ~0.4-0.5 same-person
    # for ArcFace/AdaFace-family embeddings under same-session conditions).
    # Attempted to calibrate this on docs/plans/calibration/incident_urgut
    # (2026-07-17): that dataset turned out to be single photos of
    # DIFFERENT people (not repeat captures of one person across a
    # session) — confirmed by pipeline self-consistency checks (identical
    # input embedded twice -> cosine=1.0, mirror-flip of the same face ->
    # 0.962, same crop re-encoded at JPEG q80 -> 0.980, all sane) followed
    # by cross-file cosine on the actual calibration images, which came
    # back near-zero (bonafide-bonafide pairs: min=-0.079, max=0.203,
    # mean=0.058 — nowhere near a same-person range). This is NOT a
    # pipeline bug; it means the calibration set cannot answer the
    # question Layer 3 needs answered. Kept at the ML_CORE literature
    # floor (0.40) rather than inventing a new number — genuinely
    # UNCALIBRATED, needs a real same-session multi-frame corpus (see
    # ML_CORE §6.2 item 1, "bona fide corpus") before this can be trusted
    # as a security threshold rather than a placeholder.
    IDENTITY_MIN: float = 0.40

    # Layer 2 active challenge — randomization pool. BLINK is now IMPLEMENTED
    # (2026-07-17, app/active_challenge.py + app/face_landmarks.py::
    # eye_aspect_ratios) using EAR from the SAME landmark_3d_68 model already
    # loaded for pose — no landmark_2d_106, no MediaPipe, no new dependency.
    # The EYE-CONTOUR INDEX MAPPING (36-41 right eye, 42-47 left eye — the
    # standard dlib/iBUG-300W 68-point convention landmark_3d_68 is fit to)
    # was VERIFIED against a real photo, not assumed — see
    # FrameFace.landmark_68 docstring in app/face_landmarks.py.
    #
    # BLINK IS DELIBERATELY STILL EXCLUDED FROM THIS POOL, though: index
    # correctness and THRESHOLD calibration (LIVENESS_EAR_BLINK_MAX below)
    # are different questions, and only the first one is solved. ML_CORE
    # §2.2 already flagged its own EAR threshold as "typical value from
    # literature ... NOT calibrated under our camera/resolution —
    # placeholder"; shipping an uncalibrated BLINK gate as an ACTIVE
    # security control (default pool member) is exactly the "don't lower
    # FAR for a good-looking number" trap the honesty rules warn against.
    # Detection stays reachable (a caller can put "BLINK" in a session's
    # own steps) for exactly this future recalibration to build on, without
    # a second implementation pass.
    LIVENESS_CHALLENGE_STEPS_POOL: str = "TURN_LEFT,TURN_RIGHT"
    # How many steps to sample from the pool per session — a RANGE, not a
    # fixed count (Challenge Entropy sprint, CHALLENGE_ENTROPY_SPRINT_v1.md
    # §5.1, requirement dictated by Rustam's review §1 p.1). Replaces the old
    # fixed `LIVENESS_CHALLENGE_STEP_COUNT=2`, which with only 2 supported
    # steps meant "always both, ~1 bit of entropy" — ML_CORE §2.2's own
    # complaint about this service. `app/liveness_session.py::
    # generate_challenge_spec` samples `k = rng.randint(MIN, MAX)` each call,
    # clamped to the CURRENT pool size (see that function's docstring) — with
    # today's 2-step pool this clamps down to k=2 deterministically, i.e.
    # PRODUCTION BEHAVIOR IS UNCHANGED until the pool actually grows past 4
    # (Fase 5, volume rollout, separate from this sprint). The pool itself is
    # NOT expanded here.
    # MEDIUM finding (MF DOOM code review, 2026-07-20): `ge=0` rejects a
    # negative count outright at startup (Pydantic ValidationError, fail
    # fast). An INVERTED range (MIN > MAX) is deliberately NOT rejected here
    # — `app/liveness_session.py::generate_challenge_spec` already clamps
    # `lo = min(step_count_min, hi)` before `rng.randint`, so a misconfigured
    # inverted range degrades to a narrower-but-still-valid range rather than
    # crashing; rejecting it here too would be redundant defense that could
    # itself reject a legitimate env-var rollout ordering.
    LIVENESS_CHALLENGE_STEP_COUNT_MIN: int = Field(default=3, ge=0)
    LIVENESS_CHALLENGE_STEP_COUNT_MAX: int = Field(default=4, ge=0)

    # Required yaw deviation (degrees) from the frontal reference for a
    # TURN_LEFT/TURN_RIGHT step to count as satisfied. ML_CORE §2.2's own
    # number (+/-20 deg), carried over UNVERIFIED against real device/pose
    # convention — see app/active_challenge.py module docstring for the
    # sign-convention caveat (which physical direction maps to positive
    # yaw has not been confirmed with a labeled real capture).
    LIVENESS_YAW_TURN_MIN_DEG: float = 20.0
    # Max |yaw| for a frame to count as the frontal reference/return-to-
    # center. Placeholder, not calibrated.
    LIVENESS_YAW_FRONTAL_MAX_DEG: float = 10.0

    # NOD_UP/NOD_DOWN (CHALLENGE_ENTROPY_SPRINT_v1.md §4.1, Фаза 1) —
    # required |pitch| deviation (degrees) for a nod step to count as
    # satisfied, symmetric to LIVENESS_YAW_TURN_MIN_DEG above but on the
    # pitch axis (`FrameFace.pose_pitch`, same landmark_3d_68 pass, no new
    # inference — see app/active_challenge.py module docstring). Carried
    # over the SAME 20.0 literature/ML_CORE number used for yaw, UNVERIFIED
    # against a real device — the sign-convention caveat already documented
    # for TURN_LEFT/TURN_RIGHT (app/active_challenge.py) applies here
    # identically: which physical direction ("chin up" vs "chin down") maps
    # to positive pitch has NOT been confirmed with a labeled real capture.
    # If the sign is backwards, NOD_UP/NOD_DOWN swap meaning but the
    # security property (some real pitch rotation happened) still holds.
    # No separate "frontal" threshold is introduced for pitch — the
    # existing global has_frontal gate below stays yaw-only, same as it
    # already is for BLINK (this is a deliberate scope decision, not an
    # oversight: NOD's own evidence-frame check does not require
    # frontality, mirroring how TURN_LEFT/TURN_RIGHT's evidence frame is
    # never itself "frontal" either).
    LIVENESS_PITCH_NOD_MIN_DEG: float = 20.0

    # BLINK closed-eye cutoff for min(right_ear, left_ear) — see
    # app/active_challenge.py + app/face_landmarks.py::eye_aspect_ratios.
    # UNCALIBRATED: 0.20 is the widely-cited literature value (Soukupová &
    # Čech 2016; the pyimagesearch EAR-blink tutorial most implementations
    # trace back to), NOT measured on this camera/landmark-model domain — no
    # real closed-eye frame exists in this repo to calibrate against (only
    # single bonafide/spoof photos, none mid-blink). A same-domain sanity
    # check on 3 real OPEN-eye frontal selfies (2026-07-17, i5-11400 CPU)
    # found EAR from THIS landmark_3d_68-derived measurement ranging
    # 0.214-0.317 — i.e. one genuinely-open eye already sits at 0.214, only
    # 0.014 above this cutoff. That is a real warning sign the literature
    # number may sit too close to this model's open-eye noise floor, not
    # cosmetic — do NOT add BLINK to LIVENESS_CHALLENGE_STEPS_POOL on the
    # strength of this constant alone; a real open+closed-eye session
    # corpus is needed first (see final report / owner handoff for the
    # concrete ask: a handful of people each captured performing a real
    # blink under production-like lighting).
    LIVENESS_EAR_BLINK_MAX: float = 0.20

    # SMILE (CHALLENGE_ENTROPY_SPRINT_v1.md §4.1, Фаза 1) — minimum
    # width/height ratio of the mouth's outer contour
    # (app/face_landmarks.py::mouth_aspect_ratio) for a frame to count as
    # smile evidence. UNCALIBRATED, and WEAKER than LIVENESS_EAR_BLINK_MAX's
    # placeholder: EAR's 0.20 at least traces to a cited literature source
    # (Soukupová & Čech 2016) for a well-studied signal; there is no
    # equivalent widely-cited "MAR value = smiling" constant for this
    # width/height formulation — no such number is invented here either.
    # The ONLY real data point behind this placeholder is a same-domain
    # sanity check (2026-07-20, same photo/method as the BLINK check above,
    # see face_landmarks.py mouth-index verification comment): a NEUTRAL
    # (non-smiling) frontal mouth on a real photo measured
    # mouth_aspect_ratio()=2.56. No real smiling photo was available in
    # this environment to measure the other end of the range, so 3.0 is
    # chosen ONLY to clear that one neutral baseline with a small margin —
    # it is NOT derived from any smiling-face measurement and could easily
    # be wrong in either direction (too low -> false SMILE on a relaxed
    # face close to 2.56; too high -> real smiles never clear it). Detection
    # stays reachable (SUPPORTED_STEPS) for future recalibration, but per
    # the same rule already applied to BLINK: do NOT add SMILE to
    # LIVENESS_CHALLENGE_STEPS_POOL on the strength of this constant alone
    # — a real neutral+smiling session corpus is needed first (Фаза 5,
    # Волна 2, CHALLENGE_ENTROPY_SPRINT_v1.md §8).
    LIVENESS_MAR_SMILE_MIN: float = 3.0

    # Фаза 2 (CHALLENGE_ENTROPY_SPRINT_v1.md §5.3) — диапазон, из которого
    # `app/liveness_session.py::generate_step_windows` сэмплирует случайное
    # окно задержки на КАЖДЫЙ шаг challenge (ChallengeSpec.step_windows,
    # новое аддитивное поле). 400/1500 — ПРЕДВАРИТЕЛЬНЫЕ значения, ничего не
    # придумано под "красивое число": это не согласовано с Рустамом/UX и не
    # проверено против реального CPU-бюджета инференса (LIVENESS_INFERENCE_
    # TIMEOUT_S=8.0s уже под риском по латентности — см. §4 п.3
    # LIVENESS_CONTRACT_v1.md), см. §9 п.2 плана — открытый вопрос владельцу.
    # Timing-валидация на основе этих окон (LIVENESS_TIMING_VALIDATION_ENABLED
    # ниже) идёт мягким rollout'ом именно поэтому — жёстко резать вердикт по
    # несогласованным цифрам нельзя.
    # MEDIUM finding (MF DOOM code review, 2026-07-20): same `ge=0` fail-fast
    # for a negative delay; an inverted range (MIN > MAX) is likewise
    # deliberately left to `generate_step_windows`'s own
    # `min(delay_min_ms, delay_max_ms)`/`max(...)` clamp (see that function's
    # docstring) rather than duplicated here as a hard validator.
    LIVENESS_STEP_DELAY_MIN_MS: int = Field(default=400, ge=0)
    LIVENESS_STEP_DELAY_MAX_MS: int = Field(default=1500, ge=0)

    # Фаза 3.2 (§6.2) — серверная проверка `captured_at` (окно
    # [t_instruction_shown, expires_at] + неубывание по seq) как ПЕРВЫЙ
    # контур, независимый от M2-валидации, которую партнёр (Laravel) уже
    # реализовал у себя как ВТОРОЙ контур (требование Рустама §1 п.3).
    # МЯГКИЙ rollout, тот же паттерн, что уже прижился в этом репозитории
    # для REPLAY_PROTECTION_ENABLED/TRUST_PROXY_HEADERS: DEFAULT DISABLED —
    # `captured_at` остаётся Optional в схеме, партнёр должен СНАЧАЛА
    # подтвердить, что стабильно шлёт его на каждом кадре, ПРЕЖДЕ чем этот
    # флаг переключится в True и начнёт реально валить вердикт
    # (`reason="CAPTURED_AT_INVALID"`). Пока False — аномалия (если
    # `captured_at` вообще присутствует) только логируется в audit-log, ни
    # один честный клиент, ещё не отправляющий это поле стабильно, не
    # пострадает.
    LIVENESS_CAPTURED_AT_VALIDATION_ENABLED: bool = False

    # Фаза 3.3 (§6.3) — та же мягкая механика, но для соблюдения
    # `step_windows` (Фаза 2 выше). Зависит от Фазы 2 (существование самих
    # окон) И от партнёра (клиент должен реально начать их уважать) —
    # включать раньше времени означает резать честный трафик, у которого
    # просто ещё нет данных для соблюдения ещё не отправленных окон.
    LIVENESS_TIMING_VALIDATION_ENABLED: bool = False

    # Фаза 4 (§7) — рамка `quality_certified`, требование Рустама §1 п.2
    # ("статус quality_certified допустим только при числовых целях
    # APCER/BPCER"). ЭТО МЕХАНИЗМ, НЕ ЧИСЛА — значения ниже НЕ придуманы "для
    # галочки", они намеренно `None`/`False` до реального согласования
    # Нурали+Рустама (см. §9 п.1 плана) и реального замера на multi-frame
    # корпусе (§7.1, CALIBRATION_REPORT.md, ещё не существует).
    LIVENESS_TARGET_APCER: Optional[float] = None  # согласование Нурали+Рустам
    LIVENESS_TARGET_BPCER: Optional[float] = None  # согласование Нурали+Рустам
    # РУЧНОЙ флаг sign-off (НЕ авто-вычисляемый ни из каких метрик в этом
    # коде) — переключается владельцем ПОСЛЕ того, как замеренные APCER/BPCER
    # из CALIBRATION_REPORT.md пройдены против целей выше (§7.3).
    LIVENESS_QUALITY_CERTIFIED: bool = False
    # `model_version`, на котором был прогнан сертифицирующий отчёт — должен
    # совпадать с LIVENESS_MODEL_VERSION деплоя, когда LIVENESS_QUALITY_
    # CERTIFIED=True (защита от "сертифицировали одну версию, задеплоили
    # другую"), см. стартап-проверку в app/main.py рядом с определением
    # LIVENESS_MODEL_VERSION.
    LIVENESS_CERTIFIED_MODEL_VERSION: Optional[str] = None

    # Frame count bounds per FACEID_ANTIBYPASS_UNIFIED_PLAN_v1.md §1.2
    # ("4-6 ключевых кадров"). Below MIN -> verdict=incomplete; above MAX
    # is rejected as a malformed request (protects against a client
    # uploading many more frames than the protocol calls for).
    LIVENESS_MIN_FRAMES: int = 4
    LIVENESS_MAX_FRAMES: int = 6

    # Challenge-session validity window. Session generated by
    # POST /liveness/challenge must be consumed by POST /liveness/verdict
    # before this many seconds elapse (matches ML_CORE §2.2 "полная сессия
    # должна уложиться в общее окно" — exact UX duration is an open owner
    # decision per ML_CORE §8 item 2; 90s is a generous placeholder ceiling
    # covering the 5-6s UX target plus network/retry slack, NOT the target
    # UX duration itself).
    LIVENESS_SESSION_TTL_S: float = 90.0

    # Challenge SessionStore backend (app/liveness_session.py::build_session_store).
    # "memory" (default): in-memory dict, single-process only — fine for a
    # single-worker dev/smoke deploy, matches the WEB_CONCURRENCY=1 guard in
    # app/main.py. "redis": shared across any number of worker
    # processes/replicas — REQUIRED before running with WEB_CONCURRENCY>1
    # (see FACEID_PHASE1_PAD_GATE.md §2 item 10 and
    # docs/LIVENESS_CONTRACT_v1.md §4 item 7, both now closed by this
    # backend when selected + deployed). Explicit switch, never a silent
    # fallback — build_session_store() raises at startup if backend=redis
    # is set but Redis is unreachable.
    SESSION_STORE_BACKEND: str = "memory"

    # Redis connection URL, only used when SESSION_STORE_BACKEND=redis.
    # Default targets a local dev Redis (redis-server on the default port,
    # db 0). Prod (egaz-02.uz) needs an actual Redis instance provisioned
    # before flipping SESSION_STORE_BACKEND=redis there — see 50 CENT
    # deploy notes. Sessions are short-lived (TTL-bounded, see
    # LIVENESS_SESSION_TTL_S below) and disposable — Redis persistence
    # (RDB/AOF) is NOT required; a restart losing in-flight challenge
    # sessions just makes the next /liveness/verdict for those sessions
    # come back SESSION_NOT_FOUND, same user-facing failure mode as today's
    # in-memory store restarting.
    REDIS_URL: str = "redis://localhost:6379/0"

    # Inference timeout for POST /liveness/verdict — deliberately larger
    # than /pad/check's INFERENCE_TIMEOUT_S=2.0. Derived from the measured
    # numbers above: up to LIVENESS_MAX_FRAMES=6 key frames x
    # (~524ms AdaFace + ~160ms SCRFD + ~20ms passive-PAD) ~= 4.2s, plus
    # margin for JSON decode/base64 of up to 6 frames. This is a STOPGAP,
    # not an accepted UX budget — ML_CORE §8 item 2 wants total user-facing
    # challenge time at 5-6s, and a multi-second SERVER compute tail on top
    # of that is a real architecture risk to escalate (see IR-101 note on
    # ADAFACE_ONNX_PATH above), not something to quietly accept.
    LIVENESS_INFERENCE_TIMEOUT_S: float = 8.0

    # ------------------------------------------------------------------
    # Frame-reuse dedup + inspector/abonent fraud-pattern alerting
    # (RZA, 2026-07-20) — see app/dedup_store.py module docstring for the
    # full design. Built in direct response to a real production fraud
    # incident: the SAME photo accepted for TWO DIFFERENT abonents on one
    # sale request, 46s apart, same inspector — the stateless service had
    # nothing to catch that with.
    #
    # DEFAULT DISABLED (unlike GEOMETRY_CHECK_ENABLED, which had 14 real
    # calibration samples before defaulting on): this repo has ZERO real
    # duplicate-photo pairs from egaz-02 traffic to verify DEDUP_PHASH_
    # HAMMING_MAX against — the value below is the literature default for
    # "very likely the same source image" (imagehash-family pHash
    # comparisons), not measured on this camera/compression pipeline. It is
    # ALSO the reason the existing test suite (tests/test_pad_check.py and
    # friends) reuses one fixed synthetic image across many tests with
    # DIFFERENT transaction_ref values — flipping this on by default would
    # make those tests fail on an unrelated PR, not just the ones testing
    # dedup. Recommend: enable after this report is reviewed AND after
    # collecting a small known-duplicate vs known-different photo-pair
    # sample from real traffic to sanity-check the threshold, same
    # "не занижай FAR" bar every other threshold in this file is held to.
    DEDUP_ENABLED: bool = False
    # Hamming distance (out of 64 bits) below which two pHashes count as
    # "the same photo" for the HARD BLOCK path. UNCALIBRATED on this
    # service's real frames (see above) — 4 is the literature default.
    DEDUP_PHASH_HAMMING_MAX: int = 4
    # Retention window for both the pHash dedup table and the
    # inspector-activity table (see app/dedup_store.py). 90 days chosen to
    # match the task's own retention ask; NOT independently derived from a
    # documented fraud-investigation SLA — open question for the owner.
    DEDUP_TTL_DAYS: float = 90.0
    # SQLite file path. Lives under MODEL_DIR (like ADAFACE_ONNX_PATH) —
    # deliberately NOT a bare filename in the repo root, and deliberately a
    # real on-disk file (not ":memory:") in production so the 90-day window
    # survives a service restart/deploy; tests override this to ":memory:"
    # via the DEDUP_DB_PATH env var (see tests/test_dedup.py) for isolation.
    DEDUP_DB_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "dedup_store.sqlite3"

    # AdaFace-embedding-based dedup — ALERT ONLY, never blocks (see
    # app/dedup_store.py module docstring §2 for why this is deliberately
    # WEAKER than the reviewed spec's original "reject on face match"
    # proposal: the same real customer legitimately buys gas again on a
    # different day, so a same-person match across two different
    # transaction_ref's is the EXPECTED case, not fraud).
    #
    # DEFAULT DISABLED, and gated on LIVENESS_ENDPOINTS_ENABLED being True
    # too (app/main.py's call site) — computing an AdaFace embedding on the
    # /pad/check path requires the SAME SCRFD+landmark_3d_68 detection
    # (~100-160ms CPU) + AdaFace IR-101 embedding (~342-524ms CPU, see
    # ADAFACE_ONNX_PATH docstring above) already measured as expensive for
    # the ACTIVE-liveness service's own 8s budget — /pad/check's budget is
    # 2.0s (INFERENCE_TIMEOUT_S, app/main.py), tighter than that. Enabling
    # this without a lighter checkpoint (IR-18/50, same open item as
    # ADAFACE_ONNX_PATH) or GPU risks pushing real requests into TIMEOUT.
    # Left implemented and tested, but NOT wired to run by default.
    DEDUP_EMBEDDING_ALERT_ENABLED: bool = False
    # UNCALIBRATED literature placeholder — same "no same-domain corpus"
    # caveat as IDENTITY_MIN above, kept at the same working-hypothesis
    # value rather than inventing a different number.
    DEDUP_EMBEDDING_COSINE_ALERT: float = 0.40

    # Inspector/abonent fraud-pattern heuristic — SOFT signal only, never
    # blocks a verdict (Laravel's own hard/soft fraud-escalation contract
    # decides what to do with it downstream). Requires the CALLER to send
    # the new optional `abonent_id`/`inspector_id` fields on /pad/check — a
    # complete no-op for any caller that does not send them (today: every
    # existing caller), so this is safe to default ON unlike DEDUP_ENABLED
    # above — there is no existing traffic pattern it could disrupt.
    FRAUD_INSPECTOR_ALERT_ENABLED: bool = True
    # Sliding window (seconds) the distinct-abonent count is measured over.
    # 300s (5 min) chosen to comfortably cover the actual incident's 46s gap
    # with margin — NOT derived from a documented normal-inspector-workflow
    # baseline (how many DIFFERENT abonents a legitimate inspector visits in
    # 5 minutes during a real route) because no such baseline exists yet in
    # this repo; open question for the owner/50 CENT once real audit-log
    # volume exists to check this against.
    FRAUD_INSPECTOR_WINDOW_S: float = 300.0
    # Same caveat: 3 distinct abonents in the window is a guess at "unusual",
    # not a measured baseline.
    FRAUD_INSPECTOR_DISTINCT_ABONENT_MAX: int = 3

    model_config = {"env_prefix": ""}


def resolve_device(requested: str) -> str:
    """Map device string to actual torch device name."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def onnx_providers(device: str) -> list[str]:
    """Map a RESOLVED device string ("cuda"/"cpu", i.e. already passed through
    resolve_device() above) to an onnxruntime provider list — the same
    auto|cpu|cuda knob app/liveness.py's torch-based LivenessEngine already
    gets, extended to this service's onnxruntime/insightface consumers
    (app/adaface.py::AdaFaceEmbedder, app/face_landmarks.py::LandmarkDetector
    — Layer 0/2/3 of the active-liveness pipeline; see
    Settings.ADAFACE_ONNX_PATH docstring for the measured CPU latency
    (342-524ms/frame) this exists to cut down).

    CRITICAL (prod is CPU-only today, egaz-02.uz has no GPU): this is
    deliberately conservative and NEVER assumes a GPU-capable onnxruntime is
    actually installed just because device=="cuda" was requested. It always
    checks onnxruntime's OWN ort.get_available_providers() first —
    requirements.txt keeps `onnxruntime-gpu` an OPTIONAL install (the base
    `onnxruntime` CPU wheel stays mandatory); on a host that only has the CPU
    wheel, "CUDAExecutionProvider" simply is not in that list and this
    silently returns CPU-only, no exception, no GPU package required. The
    caller (AdaFaceEmbedder / LandmarkDetector) additionally wraps session
    creation in a try/except and retries CPU-only if a CUDA provider that
    LOOKED available still fails to actually initialize (e.g. cuDNN/CUDA
    runtime version mismatch) — this function only handles the "package not
    installed at all" case, not every possible runtime failure.
    """
    if device != "cuda":
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        log.debug("onnxruntime provider check failed, falling back to CPU", exc_info=True)
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
