"""Shared fixtures. Generates the HDF5 files on first run if they are missing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
FIXTURE_DIR = TESTS_DIR / "fixtures"
BACKEND_DIR = TESTS_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

FIXTURE_NAMES = (
    "plain.h5",
    "pandas_fixed.h5",
    "pandas_table.h5",
    "pywr_style.h5",
    "pywr_scenarios.h5",
    "pywr_combinations.h5",
)


@pytest.fixture(scope="session", autouse=True)
def fixture_files() -> Path:
    missing = [n for n in FIXTURE_NAMES if not (FIXTURE_DIR / n).exists()]
    if missing:
        import make_fixtures

        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        make_fixtures.make_plain(FIXTURE_DIR / "plain.h5")
        make_fixtures.make_pandas_fixed(FIXTURE_DIR / "pandas_fixed.h5")
        make_fixtures.make_pandas_table(FIXTURE_DIR / "pandas_table.h5")
        make_fixtures.make_pywr_style(FIXTURE_DIR / "pywr_style.h5")
        make_fixtures.make_pywr_scenarios(FIXTURE_DIR / "pywr_scenarios.h5")
        make_fixtures.make_pywr_combinations(FIXTURE_DIR / "pywr_combinations.h5")
    return FIXTURE_DIR


@pytest.fixture
def registry():
    from h5grid.files import FileRegistry

    reg = FileRegistry()
    yield reg
    reg.close_all()


@pytest.fixture
def plain(registry, fixture_files):
    return registry.open(fixture_files / "plain.h5")


@pytest.fixture
def pandas_fixed(registry, fixture_files):
    return registry.open(fixture_files / "pandas_fixed.h5")


@pytest.fixture
def pandas_table(registry, fixture_files):
    return registry.open(fixture_files / "pandas_table.h5")


@pytest.fixture
def pywr(registry, fixture_files):
    return registry.open(fixture_files / "pywr_style.h5")


@pytest.fixture
def client(registry):
    """A TestClient with token auth disabled, for endpoint behaviour tests."""
    from fastapi.testclient import TestClient

    from h5grid.main import create_app
    from h5grid.security import SessionAuth

    app = create_app(
        auth=SessionAuth(enabled=False), registry=registry, serve_static=False
    )
    # base_url sets the Host header; the default "testserver" would trip the
    # loopback-only guard before any token check ran.
    with TestClient(app, base_url="http://127.0.0.1:8765") as test_client:
        yield test_client


@pytest.fixture
def auth_client(registry):
    """A TestClient with token auth enabled; the token is on the client."""
    from fastapi.testclient import TestClient

    from h5grid.main import create_app
    from h5grid.security import SessionAuth

    auth = SessionAuth(token="test-token-abc123")
    app = create_app(auth=auth, registry=registry, serve_static=False)
    # base_url sets the Host header; the default "testserver" would trip the
    # loopback-only guard before any token check ran.
    with TestClient(app, base_url="http://127.0.0.1:8765") as test_client:
        test_client.h5grid_token = auth.token
        yield test_client


@pytest.fixture
def opened(client, fixture_files):
    """Open every fixture through the API; returns {name: file_id}."""
    ids = {}
    for name in FIXTURE_NAMES:
        response = client.post(
            "/api/files/open", json={"path": str(fixture_files / name)}
        )
        assert response.status_code == 200, response.text
        ids[name] = response.json()["file_id"]
    return ids
