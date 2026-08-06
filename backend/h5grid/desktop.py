"""Run H5Grid as a native desktop window instead of a browser tab.

The web UI is unchanged — this just hosts it in an OS window (WebView2 on
Windows, WKWebView on macOS, WebKitGTK on Linux) via pywebview, so the packaged
executable feels like an app rather than "start a server, open a browser".

Threading matters: pywebview's event loop must own the main thread, so uvicorn
runs in a background thread and we wait for it to actually accept connections
before creating the window (otherwise the window loads before the server is up
and shows a connection error).
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from . import __version__


class _Server:
    """A uvicorn server running in a background thread, with a readiness wait."""

    def __init__(self, app, host: str, port: int) -> None:
        import uvicorn

        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(self._config)
        self.host = host
        self.port = port
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout: float = 20.0) -> bool:
        """Block until the port accepts connections, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                return True
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.25)
                try:
                    probe.connect((self.host, self.port))
                    return True
                except OSError:
                    time.sleep(0.1)
        return False

    def stop(self) -> None:
        self._server.should_exit = True


def run_desktop(
    file: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "H5Grid",
) -> int:
    """Launch the desktop window. Returns a process exit code."""
    try:
        import webview
    except ImportError:
        print(
            "h5grid: desktop mode needs pywebview.\n"
            "        Install it with:  pip install 'h5grid[desktop]'\n"
            "        Or run the browser version with:  h5grid serve",
        )
        return 1

    from .cli import find_free_port
    from .main import create_app
    from .security import SessionAuth

    auth = SessionAuth(enabled=True)
    app = create_app(auth=auth)

    query = f"?token={auth.token}"
    if file is not None:
        target = Path(file).expanduser()
        if not target.exists():
            print(f"h5grid: no such file: {target}")
            return 1
        try:
            entry = app.state.registry.open(target)
            query += f"&file={entry.file_id}"
        except Exception as exc:
            print(f"h5grid: could not open {target}: {exc}")
            return 1

    port = find_free_port(host, port)
    server = _Server(app, host, port)
    server.start()

    if not server.wait_until_ready():
        print("h5grid: the server did not start in time.")
        server.stop()
        return 1

    url = f"http://{host}:{port}/{query}"

    # min_size keeps the three-pane layout usable; the window is resizable.
    webview.create_window(
        f"{title} {__version__}",
        url,
        width=1280,
        height=820,
        min_size=(900, 560),
    )
    try:
        webview.start()
    finally:
        server.stop()
    return 0
