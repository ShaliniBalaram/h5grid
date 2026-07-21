"""Argument handling for the `h5grid` command."""

from __future__ import annotations

import pytest

from h5grid.cli import build_parser, find_free_port, normalize_argv


def parse(argv: list[str]):
    return build_parser().parse_args(normalize_argv(argv))


class TestNormalizeArgv:
    def test_bare_path_becomes_open(self):
        assert normalize_argv(["file.h5"]) == ["open", "file.h5"]

    def test_explicit_open_left_alone(self):
        assert normalize_argv(["open", "file.h5"]) == ["open", "file.h5"]

    def test_serve_left_alone(self):
        assert normalize_argv(["serve", "--port", "9000"]) == [
            "serve",
            "--port",
            "9000",
        ]

    def test_no_arguments_starts_the_server(self):
        assert normalize_argv([]) == ["serve"]

    def test_flags_before_the_path(self):
        assert normalize_argv(["--no-browser", "file.h5"]) == [
            "--no-browser",
            "open",
            "file.h5",
        ]

    def test_only_flags(self):
        assert normalize_argv(["--no-browser"]) == ["--no-browser", "serve"]

    def test_help_and_version_untouched(self):
        assert normalize_argv(["--help"]) == ["--help"]
        assert normalize_argv(["--version"]) == ["--version"]


class TestParsing:
    def test_path_survives_parsing(self):
        # The regression this guards: a second top-level positional ran after
        # the subparser and reset `file` to None, so nothing was ever opened.
        assert parse(["open", "data.h5"]).file == "data.h5"
        assert parse(["data.h5"]).file == "data.h5"

    def test_path_with_flags(self):
        args = parse(["open", "data.h5", "--port", "9100", "--no-browser"])
        assert args.file == "data.h5"
        assert args.port == 9100
        assert args.no_browser is True

    def test_serve_has_no_file(self):
        args = parse(["serve"])
        assert args.file is None

    def test_defaults(self):
        args = parse(["data.h5"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765
        assert args.no_browser is False
        assert args.no_token is False


class TestPortSelection:
    def test_returns_a_bindable_port(self):
        import socket

        port = find_free_port("127.0.0.1", 0)
        assert port > 0
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))

    def test_falls_back_when_preferred_is_taken(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            busy = taken.getsockname()[1]
            chosen = find_free_port("127.0.0.1", busy)
            assert chosen != busy


class TestMainErrors:
    def test_missing_file_exits_nonzero(self, capsys):
        from h5grid.cli import main

        assert main(["open", "/nope/missing.h5"]) == 1
        assert "no such file" in capsys.readouterr().err
