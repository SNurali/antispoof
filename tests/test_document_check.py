"""Tests for app/document_check.py — Layer 0 document/passport-photo pre-filter.

Focus: (1) fail-open on every failure mode (network, timeout, bad model
output) — this layer must NEVER block a request when Ollama is unavailable;
(2) parser correctness on the response shapes actually observed from
minicpm-v during calibration (free text, missing CONFIDENCE, typos).
"""

import json
import socket
import urllib.error

import numpy as np
import pytest

from app.document_check import DocumentPhotoChecker, _parse_response


def _make_test_image() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _parse_response — format tolerance
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_well_formed_document_response(self):
        is_document, confidence = _parse_response(
            "LABEL=STUDIO_BACKGROUND CONFIDENCE=90 plain white wall"
        )
        assert is_document is True
        assert confidence == pytest.approx(0.9)

    def test_correctly_spelled_confidence_is_not_dropped(self):
        """Regression: an earlier fixed-width `CONF\\D{0,6}` regex matched the
        typo 'CONFIENCE=' (6 chars after CONF) but NOT the correctly spelled
        'CONFIDENCE=' (7 chars after CONF) — the common case silently fell
        back to the conservative default instead of using the model's actual
        number. Caught here against the literal well-formed string."""
        is_document, confidence = _parse_response("LABEL=STUDIO_BACKGROUND CONFIDENCE=95")
        assert confidence == pytest.approx(0.95)

    def test_typo_confidence_spelling_tolerated(self):
        is_document, confidence = _parse_response("LABEL=STUDIO_BACKGROUND CONFIENCE=90")
        assert is_document is True
        assert confidence == pytest.approx(0.9)

    def test_real_background_confidence_is_inverted(self):
        """CONFIDENCE=N is the model's confidence in ITS OWN label, not always
        specifically "confidence in DOCUMENT" — must be converted to
        "confidence this IS a document" regardless of which label it followed."""
        is_document, confidence = _parse_response("LABEL=REAL_BACKGROUND CONFIDENCE=80")
        assert is_document is False
        assert confidence == pytest.approx(0.2)

    def test_free_text_without_label_prefix_falls_back_to_keyword_search(self):
        """Observed in calibration: the model ignores the 'respond with LABEL='
        instruction and returns a free-text paragraph starting with the bare
        keyword instead."""
        is_document, confidence = _parse_response(
            "STUDIO_BACKGROUND, the plain wall suggests a studio backdrop."
        )
        assert is_document is True

    def test_missing_confidence_number_uses_conservative_default(self):
        """No parseable number => degrade to 0.5, which sits BELOW the default
        DOCUMENT_REJECT_THRESHOLD (0.70) so a bare unqualified label alone can
        never reject a real user — only a real number crossing the threshold can."""
        is_document, confidence = _parse_response("LABEL=STUDIO_BACKGROUND")
        assert is_document is True
        assert confidence == 0.5

    def test_unparseable_response_returns_none_none(self):
        is_document, confidence = _parse_response("The weather today is nice.")
        assert is_document is None
        assert confidence is None

    def test_case_insensitive_label(self):
        is_document, confidence = _parse_response("label=studio_background confidence=88")
        assert is_document is True
        assert confidence == pytest.approx(0.88)

    def test_studio_backdrop_wording_variant_is_recognized(self):
        """Regression: a live e2e run against the real incident_urgut spoof
        photo returned 'LABEL=STUDIO_BACKDROP' (not the literal 'BACKGROUND'
        from the prompt). An earlier strict-literal regex discarded this as
        UNPARSEABLE and failed open past a real, correctly-detected spoof."""
        text = (
            "LABEL=STUDIO_BACKDROP, CONFIDENCE=85. The background is plain and "
            "lacks any texture or depth indicative of an actual environment "
            "like a room or street. Its evenness suggests it has been digitally alte"
        )
        is_document, confidence = _parse_response(text)
        assert is_document is True
        assert confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# DocumentPhotoChecker.check — fail-open behavior
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestDocumentPhotoCheckerFailOpen:
    def test_connection_refused_fails_open(self, monkeypatch):
        """Ollama not running => ran=False, never raises."""
        def _raise(*a, **kw):
            raise urllib.error.URLError(ConnectionRefusedError())

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        checker = DocumentPhotoChecker()
        result = checker.check(_make_test_image())
        assert result.ran is False
        assert result.is_document is False
        assert "OLLAMA_UNAVAILABLE" in result.error

    def test_timeout_fails_open(self, monkeypatch):
        def _raise(*a, **kw):
            raise socket.timeout("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        checker = DocumentPhotoChecker(timeout_s=1.0)
        result = checker.check(_make_test_image())
        assert result.ran is False
        assert result.error is not None

    def test_unexpected_exception_fails_open_not_raises(self, monkeypatch):
        """Any other exception path must also degrade to ran=False, not propagate —
        the request-handling call site relies on this to never crash /pad/check."""
        def _raise(*a, **kw):
            raise RuntimeError("something exploded")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        checker = DocumentPhotoChecker()
        result = checker.check(_make_test_image())
        assert result.ran is False
        assert "UNEXPECTED" in result.error

    def test_unparseable_model_output_fails_open(self, monkeypatch):
        def _fake_urlopen(*a, **kw):
            return _FakeResponse({"response": "I cannot answer that."})

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        checker = DocumentPhotoChecker()
        result = checker.check(_make_test_image())
        assert result.ran is False
        assert result.error == "UNPARSEABLE_RESPONSE"

    def test_well_formed_document_response_flags_document(self, monkeypatch):
        def _fake_urlopen(*a, **kw):
            return _FakeResponse({"response": "LABEL=STUDIO_BACKGROUND CONFIDENCE=92 plain wall"})

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        checker = DocumentPhotoChecker()
        result = checker.check(_make_test_image())
        assert result.ran is True
        assert result.is_document is True
        assert result.confidence == pytest.approx(0.92)

    def test_well_formed_live_response_does_not_flag_document(self, monkeypatch):
        def _fake_urlopen(*a, **kw):
            return _FakeResponse({"response": "LABEL=REAL_BACKGROUND CONFIDENCE=90 outdoor scene"})

        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
        checker = DocumentPhotoChecker()
        result = checker.check(_make_test_image())
        assert result.ran is True
        assert result.is_document is False
