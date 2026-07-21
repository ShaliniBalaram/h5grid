"""`h5grid open file.h5` — start the local server and open the browser."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__


def find_free_port(host: str = "127.0.0.1", preferred: int = 8765) -> int:
    """Take the preferred port if free, otherwise let the OS choose one."""
    for candidate in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Could not find a free port to bind to.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h5grid",
        description="A lightweight HDF5 viewer for water resource model files.",
    )
    parser.add_argument("--version", action="version", version=f"h5grid {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)
    open_cmd = sub.add_parser("open", help="open a file (or just start the server)")
    open_cmd.add_argument("file", nargs="?", help="path to an .h5 file")
    _add_server_args(open_cmd)

    serve_cmd = sub.add_parser("serve", help="start the server without opening a file")
    serve_cmd.set_defaults(file=None)
    _add_server_args(serve_cmd)
    return parser


SUBCOMMANDS = {"open", "serve"}


def normalize_argv(argv: list[str]) -> list[str]:
    """Let `h5grid file.h5` mean `h5grid open file.h5`.

    Done here rather than with a second positional on the main parser: argparse
    fills every positional in order, so a top-level `file` would run after the
    subparser and overwrite the path it had just parsed with None.
    """
    for i, token in enumerate(argv):
        if token in ("-h", "--help", "--version"):
            return argv
        if token.startswith("-"):
            continue
        return argv if token in SUBCOMMANDS else [*argv[:i], "open", *argv[i:]]
    return [*argv, "serve"]


def _add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, default=8765, help="preferred port")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: localhost only)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="disable the session token (only for local development)",
    )


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(normalize_argv(raw))

    from .main import create_app
    from .security import SessionAuth

    target: Path | None = None
    if args.file:
        target = Path(args.file).expanduser()
        if not target.exists():
            print(f"h5grid: no such file: {target}", file=sys.stderr)
            return 1

    auth = SessionAuth(enabled=not args.no_token)
    app = create_app(auth=auth)

    # Pre-open the file so the browser lands straight on it and any problem is
    # reported here, in the terminal, rather than as a toast in the UI.
    query = f"?token={auth.token}" if auth.enabled else ""
    if target is not None:
        try:
            entry = app.state.registry.open(target)
            query += f"{'&' if query else '?'}file={entry.file_id}"
        except Exception as exc:
            print(f"h5grid: could not open {target}: {exc}", file=sys.stderr)
            return 1

    port = find_free_port(args.host, args.port)
    url = f"http://{args.host}:{port}/{query}"

    if not (Path(__file__).parent / "static" / "index.html").exists():
        print(
            "h5grid: no built frontend found in h5grid/static.\n"
            "        Run `npm install && npm run build` in frontend/ first.",
            file=sys.stderr,
        )

    print(f"h5grid {__version__}")
    print(f"  serving on {url}")
    if target is not None:
        print(f"  file       {target}")
    print("  press Ctrl+C to stop")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
