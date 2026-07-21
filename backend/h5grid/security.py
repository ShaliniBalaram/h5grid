"""Session token and host validation for the local API.

Binding to 127.0.0.1 keeps the network out but not the user's own browser: any
website they visit can issue requests to http://127.0.0.1:<port> and, without a
check, read local files through this API. Two guards close that:

  * a random per-launch token, required on every /api request. A cross-origin
    page cannot read it, so it cannot forge an accepted request.
  * a Host header allowlist, which blocks DNS rebinding (an attacker pointing
    their own domain at 127.0.0.1 to make the browser treat us as same-origin).
"""

from __future__ import annotations

import hmac
import ipaddress
import secrets

from fastapi import HTTPException, Request

TOKEN_HEADER = "x-h5grid-token"
TOKEN_QUERY = "token"

_ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"}


class SessionAuth:
    """Holds the token for one server launch."""

    def __init__(self, token: str | None = None, *, enabled: bool = True) -> None:
        self.token = token or secrets.token_urlsafe(32)
        self.enabled = enabled

    def check(self, request: Request) -> None:
        """Raise 401/403 unless the request carries the token from a local host."""
        if not self.enabled:
            return

        self._check_host(request)

        presented = request.headers.get(TOKEN_HEADER) or request.query_params.get(
            TOKEN_QUERY
        )
        if not presented or not hmac.compare_digest(presented, self.token):
            raise HTTPException(
                status_code=401,
                detail=(
                    "Missing or invalid session token. Open H5Grid using the URL "
                    "printed by the `h5grid` command."
                ),
            )

    def _check_host(self, request: Request) -> None:
        host = request.headers.get("host", "")
        hostname = _hostname_of(host)
        if hostname and not _is_local(hostname):
            raise HTTPException(
                status_code=403,
                detail=f"Refusing request for non-local host {host!r}.",
            )


def _hostname_of(host_header: str) -> str:
    """Strip the port from a Host header, handling bracketed IPv6."""
    host = host_header.strip()
    if not host:
        return ""
    if host.startswith("["):  # [::1]:8000
        end = host.find("]")
        if end != -1:
            return host[: end + 1]
        return host
    return host.rsplit(":", 1)[0] if ":" in host else host


def _is_local(hostname: str) -> bool:
    if hostname.lower() in _ALLOWED_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(hostname.strip("[]")).is_loopback
    except ValueError:
        return False
