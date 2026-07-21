"""Token and host guards on the local API.

These protect against a website the user happens to be visiting issuing
requests to http://127.0.0.1:<port> and reading local files through this API.
"""

from __future__ import annotations

import pytest


class TestToken:
    def test_api_requires_a_token(self, auth_client):
        response = auth_client.get("/api/browse")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorized"

    def test_header_token_accepted(self, auth_client, fixture_files):
        response = auth_client.get(
            f"/api/browse?dir={fixture_files}",
            headers={"X-H5Grid-Token": auth_client.h5grid_token},
        )
        assert response.status_code == 200

    def test_query_token_accepted(self, auth_client, fixture_files):
        # Downloads are plain browser navigations and cannot set a header.
        response = auth_client.get(
            f"/api/browse?dir={fixture_files}&token={auth_client.h5grid_token}"
        )
        assert response.status_code == 200

    def test_wrong_token_rejected(self, auth_client):
        response = auth_client.get(
            "/api/browse", headers={"X-H5Grid-Token": "not-the-token"}
        )
        assert response.status_code == 401

    def test_open_endpoint_is_guarded(self, auth_client, fixture_files):
        response = auth_client.post(
            "/api/files/open", json={"path": str(fixture_files / "plain.h5")}
        )
        assert response.status_code == 401

    def test_every_data_route_is_guarded(self, auth_client):
        for url in (
            "/api/browse",
            "/api/files/abc/tree",
            "/api/files/abc/node/meta?path=/x",
            "/api/files/abc/node/data?path=/x",
            "/api/files/abc/node/stats?path=/x&col=y",
            "/api/files/abc/node/search?path=/x&col=y&q=1",
            "/api/files/abc/node/plotdata?path=/x",
            "/api/files/abc/node/export?path=/x",
        ):
            assert auth_client.get(url).status_code == 401, url
        assert auth_client.post("/api/files/abc/close").status_code == 401


class TestHostGuard:
    def test_foreign_host_header_rejected(self, auth_client):
        # DNS rebinding: an attacker's domain resolving to 127.0.0.1.
        response = auth_client.get(
            "/api/health".replace("health", "browse"),
            headers={
                "X-H5Grid-Token": auth_client.h5grid_token,
                "Host": "evil.example.com",
            },
        )
        assert response.status_code == 403
        assert response.json()["error"] == "forbidden"

    @pytest.mark.parametrize(
        "host", ["localhost:8765", "127.0.0.1:8765", "127.0.0.1", "[::1]:8765"]
    )
    def test_local_hosts_allowed(self, auth_client, fixture_files, host):
        response = auth_client.get(
            f"/api/browse?dir={fixture_files}",
            headers={"X-H5Grid-Token": auth_client.h5grid_token, "Host": host},
        )
        assert response.status_code == 200, host


class TestHostnameParsing:
    @pytest.mark.parametrize(
        "header,expected",
        [
            ("localhost:8765", "localhost"),
            ("127.0.0.1", "127.0.0.1"),
            ("[::1]:8765", "[::1]"),
            ("", ""),
        ],
    )
    def test_hostname_of(self, header, expected):
        from h5grid.security import _hostname_of

        assert _hostname_of(header) == expected

    def test_loopback_detection(self):
        from h5grid.security import _is_local

        assert _is_local("127.0.0.1")
        assert _is_local("127.0.0.5")
        assert _is_local("localhost")
        assert not _is_local("evil.example.com")
        assert not _is_local("192.168.1.10")


class TestTokenGeneration:
    def test_tokens_are_random_and_long(self):
        from h5grid.security import SessionAuth

        first, second = SessionAuth(), SessionAuth()
        assert first.token != second.token
        assert len(first.token) >= 32

    def test_disabled_auth_lets_everything_through(self, client):
        assert client.get("/api/browse").status_code in (200, 404)
