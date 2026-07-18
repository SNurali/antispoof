"""Tests for the IP allowlist's proxy-awareness (settings.TRUST_PROXY_HEADERS).

Context (BUSTA RHYMES, deploy/mtls/): once nginx starts reverse-proxying
external traffic to uvicorn on 127.0.0.1, request.client.host for EVERY
external caller becomes nginx's own loopback address. Without an explicit
opt-in, app/main.py's IP allowlist (app/main.py::_effective_client_ip) would
silently stop filtering anyone at all, since 127.0.0.0/8 is itself allowed.

These tests exercise app/main.py::_effective_client_ip + the
security_and_rate_limit middleware directly, covering:
  (a) TRUST_PROXY_HEADERS off (default) — X-Forwarded-For is ignored even if
      sent; only request.client.host (the TCP peer) is judged.
  (b) TRUST_PROXY_HEADERS on + request arriving from loopback (the
      legitimate nginx-proxied case) + X-Forwarded-For carrying a
      non-allowlisted real client IP — the request must be rejected, since
      the allowlist should now judge the REAL client, not nginx.
  (c) TRUST_PROXY_HEADERS on + request arriving from a NON-loopback peer
      (someone bypassed nginx and hit uvicorn directly) with a forged
      X-Forwarded-For claiming an allowlisted IP — the header must be
      ignored; the real (non-loopback, presumably non-allowlisted) peer
      address is what gets judged.
"""

import os

os.environ.setdefault("SERVICE_TOKEN", "")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("RATE_LIMIT_BURST", "1000")
os.environ.setdefault("RATE_LIMIT_SUSTAINED", "1000.0")
os.environ["ANTISPOOF_SKIP_MODELS"] = "1"

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_middleware_state():
    """Prevent startup model loading, reset rate limiter + TRUST_PROXY_HEADERS
    between tests so no test leaks state into another."""
    import app.main as m

    original_handlers = m.app.router.on_startup.copy()
    m.app.router.on_startup.clear()
    m._rate_limiter._windows.clear()
    m._rate_limiter._burst = 1000
    m._rate_limiter._sustained = 1000.0
    original_trust = m.settings.TRUST_PROXY_HEADERS
    m.settings.TRUST_PROXY_HEADERS = False
    yield m
    m.settings.TRUST_PROXY_HEADERS = original_trust
    m.app.router.on_startup = original_handlers


NOT_ALLOWLISTED_IP = "8.8.8.8"       # public IP, not in any ALLOWED_NETWORKS entry
ALLOWLISTED_LAN_IP = "192.168.0.42"  # inside the 192.168.0.0/24 allowlisted range


def _get(client: TestClient, **headers) -> "object":
    return client.get("/health", headers=headers or None)


class TestTrustProxyHeadersOff:
    """(a) Default behavior — X-Forwarded-For must be fully ignored."""

    def test_client_host_outside_allowlist_is_rejected(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        with TestClient(m.app, client=(NOT_ALLOWLISTED_IP, 12345)) as c:
            resp = _get(c)
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Access denied"}

    def test_forged_xff_does_not_rescue_a_disallowed_peer(self, _isolated_middleware_state):
        """Even a real client.host from loopback with an X-Forwarded-For
        claiming to be allowlisted must NOT be trusted while the flag is off:
        the allowlist should keep judging request.client.host as-is."""
        m = _isolated_middleware_state
        with TestClient(m.app, client=(NOT_ALLOWLISTED_IP, 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": ALLOWLISTED_LAN_IP})
        assert resp.status_code == 403

    def test_client_host_inside_allowlist_passes_through(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        with TestClient(m.app, client=("127.0.0.1", 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": NOT_ALLOWLISTED_IP})
        # X-Forwarded-For must be ignored entirely — the real peer (loopback)
        # is allowlisted, so the request must NOT be blocked by the allowlist
        # (health may still be 503 because models aren't loaded in tests).
        assert resp.status_code != 403


class TestTrustProxyHeadersOnFromLoopback:
    """(b) TRUST_PROXY_HEADERS=true, request genuinely proxied via nginx
    (peer == loopback) — the REAL client from X-Forwarded-For must be judged."""

    def test_bad_real_client_behind_loopback_proxy_is_rejected(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        with TestClient(m.app, client=("127.0.0.1", 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": NOT_ALLOWLISTED_IP})
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Access denied"}

    def test_good_real_client_behind_loopback_proxy_passes_through(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        with TestClient(m.app, client=("127.0.0.1", 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": ALLOWLISTED_LAN_IP})
        assert resp.status_code != 403

    def test_multi_hop_xff_uses_last_entry_as_nginx_wrote_it(self, _isolated_middleware_state):
        """nginx's $proxy_add_x_forwarded_for APPENDS to any client-supplied
        X-Forwarded-For rather than replacing it — so an attacker prepending
        a fake "trusted" IP must NOT help; only the LAST (nginx-appended)
        hop is authoritative."""
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        forged_chain = f"{ALLOWLISTED_LAN_IP}, {NOT_ALLOWLISTED_IP}"
        with TestClient(m.app, client=("127.0.0.1", 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": forged_chain})
        # last entry (NOT_ALLOWLISTED_IP) must be what's judged -> rejected
        assert resp.status_code == 403


class TestTrustProxyHeadersOnButNotFromLoopback:
    """(c) TRUST_PROXY_HEADERS=true but the connection did NOT come from
    loopback (nginx bypassed, uvicorn hit directly) — X-Forwarded-For must
    be ignored regardless of what it claims."""

    def test_forged_xff_from_non_loopback_peer_is_ignored(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        # Real TCP peer is NOT allowlisted and NOT loopback; X-Forwarded-For
        # lies and claims an allowlisted address — must still be rejected,
        # because the header is not trusted for a non-loopback peer.
        with TestClient(m.app, client=(NOT_ALLOWLISTED_IP, 12345)) as c:
            resp = _get(c, **{"X-Forwarded-For": ALLOWLISTED_LAN_IP})
        assert resp.status_code == 403
        assert resp.json() == {"detail": "Access denied"}

    def test_genuine_non_loopback_allowlisted_peer_still_passes(self, _isolated_middleware_state):
        """Sanity check: a direct (non-loopback) but genuinely allowlisted
        peer keeps working normally when TRUST_PROXY_HEADERS is on — this
        flag only changes how X-Forwarded-For is trusted, not the allowlist
        itself."""
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        with TestClient(m.app, client=(ALLOWLISTED_LAN_IP, 12345)) as c:
            resp = _get(c)
        assert resp.status_code != 403


class TestEffectiveClientIpUnit:
    """Direct unit coverage of app/main.py::_effective_client_ip, independent
    of the ASGI/TestClient plumbing above."""

    def _fake_request(self, m, peer_ip: str, xff: str | None = None):
        from starlette.requests import Request

        headers = [(b"x-forwarded-for", xff.encode())] if xff else []
        scope = {
            "type": "http",
            "client": (peer_ip, 12345),
            "headers": headers,
        }
        return Request(scope)

    def test_disabled_returns_raw_peer_ip(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = False
        req = self._fake_request(m, "127.0.0.1", xff=NOT_ALLOWLISTED_IP)
        assert m._effective_client_ip(req) == "127.0.0.1"

    def test_enabled_loopback_no_xff_returns_raw_peer_ip(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        req = self._fake_request(m, "127.0.0.1")
        assert m._effective_client_ip(req) == "127.0.0.1"

    def test_enabled_loopback_with_xff_returns_last_hop(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        req = self._fake_request(m, "127.0.0.1", xff=f"{ALLOWLISTED_LAN_IP}, {NOT_ALLOWLISTED_IP}")
        assert m._effective_client_ip(req) == NOT_ALLOWLISTED_IP

    def test_enabled_non_loopback_ignores_xff(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        req = self._fake_request(m, NOT_ALLOWLISTED_IP, xff=ALLOWLISTED_LAN_IP)
        assert m._effective_client_ip(req) == NOT_ALLOWLISTED_IP

    def test_enabled_loopback_malformed_xff_falls_back_to_peer(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        req = self._fake_request(m, "127.0.0.1", xff="not-an-ip")
        assert m._effective_client_ip(req) == "127.0.0.1"

    def test_ipv6_loopback_is_recognized(self, _isolated_middleware_state):
        m = _isolated_middleware_state
        m.settings.TRUST_PROXY_HEADERS = True
        req = self._fake_request(m, "::1", xff=NOT_ALLOWLISTED_IP)
        assert m._effective_client_ip(req) == NOT_ALLOWLISTED_IP
