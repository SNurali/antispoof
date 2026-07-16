"""Layer 0 — document/passport-photo pre-filter (runs BEFORE passive-PAD).

Calls a local Ollama vision-language model (default: minicpm-v) to flag a
presentation-attack class that the existing multi-signal passive-PAD
(app/multisignal.py) is not designed to catch: a studio/ID-style document
photo (plain white/grey/blue backdrop, matted cutout edges, frontal passport
pose) held up to the camera. This is an ORTHOGONAL signal — it looks at
composition/background, not texture/frequency recapture artifacts — and does
NOT touch `_fuse()` / `NN_TRUST_REAL` in app/liveness.py.

Calibration finding (2026-07-16, incident_urgut dataset, n=1 spoof + n=12
bonafide, see docs/plans/calibration/incident_urgut/README.md) — DEFAULT
DISABLED (DOCUMENT_CHECK_ENABLED=False), this is NOT production-validated:
  - The obvious combined prompt ("classify DOCUMENT vs LIVE" citing pose +
    lighting + matting + background together) produced 6/12 (50%) false
    positives on bonafide selfies with this quantized 7.6B model — NOT usable.
  - Narrowing the prompt to a single cue (background only: studio backdrop
    vs real-world scene, used below) correctly separated the spoof from the
    2 "easy" bonafide (varied outdoor scenes) but STILL false-positived on
    2/4 tested bonafide that happen to have a plain/simple background (a
    beige indoor wall; a painted blue/white wood-plank door) — ordinary,
    non-rare home-selfie scenarios, not edge cases. 2/4 on this small sample
    is not a usable false-accept rate for gating a real transaction.
  - Net: neither prompt variant is safe to enable by default. Do not raise
    DOCUMENT_CHECK_ENABLED to True without a larger, more diverse bonafide
    set (plain-wall/simple-background selfies specifically) and a re-run of
    the EXACT shipped prompt below (the CONFIDENCE= field was added after
    the free-text variant was tested; its numeric behavior is unverified).

Fail-open, always: if Ollama is unreachable, times out, or returns something
the parser cannot understand, `check()` returns DocumentCheckResult(ran=False,
...) and the caller MUST fall through to passive-PAD unchanged. Availability
of the liveness service must never depend on Ollama being up, and this layer
must never turn a real user away due to an infra hiccup.
"""

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger("antispoof.document_check")

# Coordinator note (2026-07-16): a newer minicpm-v4.6 is being evaluated in
# parallel as a candidate replacement — keep the model tag env-configurable,
# never hardcode it, so switching is a config change, not a code change.
DEFAULT_MODEL = "minicpm-v:latest"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Narrow, single-cue prompt. Calibration (see module docstring) showed the
# small quantized model reasons far more reliably about ONE composition cue
# (background) than about a multi-criteria "is this a document photo"
# question — the combined prompt confused ~50% of real outdoor selfies.
_PROMPT = (
    "Look at the BACKGROUND behind the person's head in this photo. "
    "Is the background a plain solid white/grey/blue studio backdrop (like a "
    "passport or ID photo), or is it a real-world scene (room, street, "
    "outdoors, furniture, vehicle, etc)? "
    "Respond with a line starting with LABEL=STUDIO_BACKGROUND or "
    "LABEL=REAL_BACKGROUND, followed by CONFIDENCE=<integer 0-100> "
    "(your confidence in that label), then a short reason."
)

# Matches LABEL=STUDIO_BACKGROUND / LABEL=REAL_BACKGROUND anywhere in the text
# (the model does not reliably restrict output to a single line). `STUDIO_BACK\w*`
# (not a literal "BACKGROUND") because a live e2e run against the real spoof
# image returned "LABEL=STUDIO_BACKDROP" — the model does not consistently
# reuse the exact word from the prompt; a strict literal match on that one
# real call would have discarded a correct, high-confidence (85%) spoof
# detection as UNPARSEABLE and failed open past an actual attack photo.
_LABEL_RE = re.compile(r"LABEL\s*[:=]?\s*(STUDIO_BACK\w*|REAL_BACKGROUND)", re.IGNORECASE)
# Loose CONF-prefix match: `[A-Z]*` absorbs the rest of "CONFIDENCE" (or a
# quantized-model typo like "CONFIENCE") of any length, then optional
# punctuation/space before the number. A bug in an earlier fixed-width
# `\D{0,6}` version matched the *typo* "CONFIENCE=" (6 chars after CONF) but
# NOT the correctly-spelled "CONFIDENCE=" (7 chars after CONF) — caught by
# a unit test against literal 'CONFIDENCE=90' input.
_CONF_RE = re.compile(r"CONF[A-Z]*\s*[:=]\s*(\d{1,3})", re.IGNORECASE)

# Fallback keyword search used only if the model ignores the LABEL= format
# entirely (observed in testing: full free-text paragraphs starting with the
# bare keyword). Order matters: check STUDIO first since REAL_BACKGROUND is
# not a substring of it, avoids one matching inside the other's context text.
# NOTE (2026-07-16 e2e finding): this keyword list is based only on wording
# actually observed so far (STUDIO_BACKGROUND / STUDIO_BACKDROP / REAL_BACKGROUND).
# The model is not guaranteed to stick to these — other plausible synonyms
# (e.g. "PLAIN_BACKGROUND", "ID_PHOTO", "PASSPORT_STYLE") have NOT been
# observed or tested and are not covered; broadening further without a real
# example would be guessing, not measuring.
_STUDIO_KEYWORDS = ("STUDIO_BACKGROUND", "STUDIO_BACKDROP")
_REAL_KEYWORDS = ("REAL_BACKGROUND",)


@dataclass
class DocumentCheckResult:
    """Outcome of the Layer 0 document-photo check. Always constructible without
    raising — failure states are represented as data, not exceptions."""

    ran: bool  # False => this layer did not produce a usable verdict; caller MUST fail open
    is_document: bool = False
    confidence: float = 0.0  # 0..1, confidence in the "this is a document/studio photo" call
    raw_label: Optional[str] = None
    raw_response: str = ""
    error: Optional[str] = None
    elapsed_s: float = 0.0


def _parse_response(text: str) -> tuple[Optional[bool], Optional[float]]:
    """Best-effort parse of the model's verdict.

    Returns (is_document, confidence) or (None, None) if nothing recognizable
    is present. `confidence` is normalized to be "confidence that this IS a
    document/studio photo" regardless of which label the model actually wrote
    the number next to (testing showed the model sometimes states confidence
    in its own label, not always in the DOCUMENT class specifically).
    """
    label_match = _LABEL_RE.search(text)
    label: Optional[str] = None
    if label_match:
        matched = label_match.group(1).upper()
        # Normalize any STUDIO_BACK* variant (BACKGROUND, BACKDROP, ...) to one
        # canonical label — the regex intentionally accepts wording variants
        # (see _LABEL_RE comment) but downstream logic needs a fixed constant.
        label = "STUDIO_BACKGROUND" if matched.startswith("STUDIO_BACK") else "REAL_BACKGROUND"

    if label is None:
        upper = text.upper()
        if any(k in upper for k in _STUDIO_KEYWORDS):
            label = "STUDIO_BACKGROUND"
        elif any(k in upper for k in _REAL_KEYWORDS):
            label = "REAL_BACKGROUND"

    if label is None:
        return None, None

    conf_match = _CONF_RE.search(text)
    stated_conf: Optional[float] = None
    if conf_match:
        try:
            stated_conf = max(0.0, min(100.0, float(conf_match.group(1)))) / 100.0
        except ValueError:
            stated_conf = None

    is_document = label == "STUDIO_BACKGROUND"
    if stated_conf is None:
        # No parseable number — degrade to a conservative fixed confidence
        # rather than failing open entirely, since the label itself IS
        # unambiguous. Set below DOCUMENT_REJECT_THRESHOLD default (0.70) so
        # a bare unqualified label alone cannot reject anyone; a real number
        # from the model is required to cross the reject threshold.
        stated_conf = 0.5
    # stated_conf is "confidence in `label`" — convert to "confidence in
    # STUDIO_BACKGROUND" specifically.
    doc_confidence = stated_conf if is_document else (1.0 - stated_conf)
    return is_document, doc_confidence


class DocumentPhotoChecker:
    """Calls a local Ollama vision model to flag document/studio-style photos.

    Construct once (like LivenessEngine) and reuse; holds no model state of
    its own, just HTTP config, so it is cheap and thread-safe to share.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout_s: float = 20.0,
    ) -> None:
        self._model = model
        self._url = ollama_url
        self._timeout_s = timeout_s

    def check(self, image_bgr: np.ndarray) -> DocumentCheckResult:
        """Run the document/studio-background check on a BGR numpy frame.

        Never raises. Every failure mode (network, timeout, decode, bad model
        output) is captured into DocumentCheckResult(ran=False, error=...) so
        the caller can unconditionally fall through to passive-PAD.
        """
        t0 = time.monotonic()
        try:
            import cv2  # local import: keeps module import light for unit tests

            ok, buf = cv2.imencode(".jpg", image_bgr)
            if not ok:
                return DocumentCheckResult(ran=False, error="ENCODE_FAILED")

            b64 = base64.b64encode(buf.tobytes()).decode()
            payload = json.dumps(
                {"model": self._model, "prompt": _PROMPT, "images": [b64], "stream": False}
            ).encode()
            req = urllib.request.Request(
                self._url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            elapsed = time.monotonic() - t0
            log.warning(
                "document_check: Ollama unavailable/timeout after %.1fs (%s) — "
                "failing open to passive-PAD", elapsed, exc,
            )
            return DocumentCheckResult(ran=False, error=f"OLLAMA_UNAVAILABLE: {exc}", elapsed_s=elapsed)
        except Exception as exc:  # noqa: BLE001 - must never crash the request path
            elapsed = time.monotonic() - t0
            log.warning(
                "document_check: unexpected error after %.1fs (%s) — failing open to passive-PAD",
                elapsed, exc,
            )
            return DocumentCheckResult(ran=False, error=f"UNEXPECTED: {type(exc).__name__}: {exc}", elapsed_s=elapsed)

        elapsed = time.monotonic() - t0
        text = str(data.get("response", "")).strip()
        is_document, confidence = _parse_response(text)

        if is_document is None:
            log.warning(
                "document_check: unparseable model response after %.1fs: %r — "
                "failing open to passive-PAD", elapsed, text[:200],
            )
            return DocumentCheckResult(
                ran=False, raw_response=text, error="UNPARSEABLE_RESPONSE", elapsed_s=elapsed,
            )

        return DocumentCheckResult(
            ran=True,
            is_document=is_document,
            confidence=confidence,
            raw_label="STUDIO_BACKGROUND" if is_document else "REAL_BACKGROUND",
            raw_response=text,
            elapsed_s=elapsed,
        )
