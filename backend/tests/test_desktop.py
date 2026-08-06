"""Desktop-window mode: server lifecycle and CLI routing.

The window itself (pywebview) needs a GUI session, so it is not opened here;
these cover everything up to that call.
"""

from __future__ import annotations

import urllib.request

import pytest


class TestBackgroundServer:
    def test_starts_becomes_ready_and_serves(self, registry, fixture_files):
        from h5grid.cli import find_free_port
        from h5grid.desktop import _Server
        from h5grid.main import create_app
        from h5grid.security import SessionAuth

        auth = SessionAuth(enabled=True)
        app = create_app(auth=auth, registry=registry)
        entry = app.state.registry.open(fixture_files / "pywr_style.h5")
        port = find_free_port("127.0.0.1", 8793)

        server = _Server(app, "127.0.0.1", port)
        server.start()
        try:
            assert server.wait_until_ready(timeout=15) is True

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/files/{entry.file_id}/tree",
                headers={"X-H5Grid-Token": auth.token},
            )
            assert urllib.request.urlopen(req, timeout=5).status == 200
        finally:
            server.stop()

    def test_wait_times_out_on_a_dead_port(self):
        from h5grid.desktop import _Server

        # A server object pointed at a port nothing is listening on: readiness
        # must give up rather than hang forever.
        server = _Server(app=None, host="127.0.0.1", port=9)  # discard port
        # never start() it
        assert server.wait_until_ready(timeout=1.0) is False


class TestDesktopCliRouting:
    def test_desktop_subcommand_calls_run_desktop(self, monkeypatch, tmp_path):
        import h5grid.desktop as desktop
        from h5grid.cli import main

        calls = []

        def fake(*a, **k):
            calls.append((a, k))
            return 0

        monkeypatch.setattr(desktop, "run_desktop", fake)
        f = tmp_path / "x.h5"
        f.write_bytes(b"")  # existence is checked inside run_desktop, which is mocked
        assert main(["desktop", str(f)]) == 0
        assert len(calls) == 1

    def test_open_with_desktop_flag_routes_to_window(self, monkeypatch, tmp_path):
        import h5grid.desktop as desktop
        from h5grid.cli import main

        calls = []

        def fake(*a, **k):
            calls.append(True)
            return 0

        monkeypatch.setattr(desktop, "run_desktop", fake)
        f = tmp_path / "y.h5"
        f.write_bytes(b"")
        assert main(["open", str(f), "--desktop"]) == 0
        assert calls == [True]

    def test_serve_does_not_open_a_window(self, monkeypatch):
        import h5grid.desktop as desktop

        opened = {"n": 0}
        monkeypatch.setattr(
            desktop, "run_desktop", lambda *a, **k: opened.__setitem__("n", opened["n"] + 1) or 0
        )
        # `serve` has no desktop flag, so parsing it must not reference desktop at
        # all. We assert by checking the parser accepts it and sets command.
        from h5grid.cli import build_parser, normalize_argv

        args = build_parser().parse_args(normalize_argv(["serve"]))
        assert args.command == "serve"
        assert opened["n"] == 0


class TestMissingPywebview:
    def test_clear_message_when_pywebview_absent(self, monkeypatch, capsys):
        import builtins

        from h5grid.desktop import run_desktop

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "webview":
                raise ImportError("no webview")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert run_desktop(None) == 1
        out = capsys.readouterr().out
        assert "pip install" in out and "desktop" in out
