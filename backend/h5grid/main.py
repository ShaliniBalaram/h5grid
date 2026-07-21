"""FastAPI application: thin endpoints over the service layer."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import __version__, export, service, stats
from .files import (
    FileChangedError,
    FileRegistry,
    NodeNotFoundError,
    list_directory,
    list_roots,
)
from .security import SessionAuth

STATIC_DIR = Path(__file__).parent / "static"


class OpenFileRequest(BaseModel):
    path: str = Field(..., description="Absolute path to an .h5 file on this machine")


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def create_app(
    *,
    auth: SessionAuth | None = None,
    registry: FileRegistry | None = None,
    serve_static: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        application.state.registry.close_all()

    app = FastAPI(
        title="H5Grid",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.auth = auth or SessionAuth()
    app.state.registry = registry or FileRegistry()

    async def require_token(request: Request) -> None:
        request.app.state.auth.check(request)

    guarded = [Depends(require_token)]

    # -- error shaping ---------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error = getattr(exc, "error_code", None) or _default_error_code(exc.status_code)
        return _error(exc.status_code, error, str(exc.detail))

    @app.exception_handler(FileChangedError)
    async def file_changed_handler(request: Request, exc: FileChangedError):
        return _error(409, "file_changed", str(exc))

    @app.exception_handler(NodeNotFoundError)
    async def node_missing_handler(request: Request, exc: NodeNotFoundError):
        return _error(404, "node_not_found", str(exc))

    @app.exception_handler(service.RequestTooLargeError)
    async def too_large_handler(request: Request, exc: service.RequestTooLargeError):
        return _error(413, "request_too_large", str(exc))

    @app.exception_handler(export.ExportTooLargeError)
    async def export_too_large_handler(request: Request, exc: export.ExportTooLargeError):
        return _error(413, "export_too_large", str(exc))

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return _error(400, "bad_request", str(exc))

    # -- helpers ---------------------------------------------------------

    def get_file(fid: str):
        try:
            entry = app.state.registry.get(fid)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Unknown file id. It expires when the file changes on disk or "
                    "the server restarts; open the file again."
                ),
            )
        app.state.registry.sweep()
        return entry

    # -- endpoints -------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/browse/roots", dependencies=guarded)
    async def browse_roots() -> dict[str, Any]:
        return {"roots": await asyncio.to_thread(list_roots)}

    @app.get("/api/browse", dependencies=guarded)
    async def browse(dir: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(list_directory, dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))

    @app.post("/api/files/open", dependencies=guarded)
    async def open_file(body: OpenFileRequest) -> dict[str, Any]:
        try:
            entry = await asyncio.to_thread(app.state.registry.open, body.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not open as HDF5: {exc}",
            )
        return {
            "file_id": entry.file_id,
            "path": str(entry.path),
            "name": entry.path.name,
            "size_bytes": entry.size_bytes,
            "mtime": entry.mtime_ns / 1e9,
        }

    @app.get("/api/files/{fid}/tree", dependencies=guarded)
    async def get_tree(fid: str, raw: bool = False) -> dict[str, Any]:
        entry = get_file(fid)
        return await entry.run(service.tree_payload, entry, raw=raw)

    @app.get("/api/files/{fid}/node/meta", dependencies=guarded)
    async def get_meta(
        fid: str, path: str, slice: str | None = None
    ) -> dict[str, Any]:
        entry = get_file(fid)
        return await entry.run(service.node_meta, entry, path, slice)

    @app.get("/api/files/{fid}/node/data", dependencies=guarded)
    async def get_data(
        fid: str,
        path: str,
        start: int = 0,
        stop: int | None = None,
        cols: str | None = None,
        slice: str | None = None,
        use_time_index: bool = False,
    ) -> dict[str, Any]:
        entry = get_file(fid)
        return await entry.run(
            service.node_data,
            entry,
            path,
            start=start,
            stop=stop,
            cols_spec=cols,
            dim_slice_spec=slice,
            use_time_index=use_time_index,
        )

    @app.get("/api/files/{fid}/node/stats", dependencies=guarded)
    async def get_stats(
        fid: str, path: str, col: str, slice: str | None = None
    ) -> dict[str, Any]:
        entry = get_file(fid)
        cache = entry.stats_cache()
        key = (f"{path}|{slice or ''}", col)
        if key in cache:
            return cache[key]

        def compute():
            reader = entry.reader(path)
            dim_slice = (
                service.parse_dim_slice(slice, reader.ndim) if reader.ndim > 2 else None
            )
            return stats.column_stats(reader, col, dim_slice)

        try:
            result = await entry.run(compute)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        cache[key] = result
        return result

    @app.get("/api/files/{fid}/node/search", dependencies=guarded)
    async def search(
        fid: str,
        path: str,
        col: str,
        q: str,
        limit: int = 500,
        slice: str | None = None,
    ) -> dict[str, Any]:
        entry = get_file(fid)

        def run_search():
            reader = entry.reader(path)
            dim_slice = (
                service.parse_dim_slice(slice, reader.ndim) if reader.ndim > 2 else None
            )
            return stats.search_column(
                reader, col, q, limit=max(1, min(limit, 5000)), dim_slice=dim_slice
            )

        try:
            return await entry.run(run_search)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/files/{fid}/node/plotdata", dependencies=guarded)
    async def plotdata(
        fid: str,
        path: str,
        cols: str | None = None,
        max_points: int = 4000,
        slice: str | None = None,
        use_time_index: bool = True,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, Any]:
        entry = get_file(fid)
        return await entry.run(
            service.plot_data,
            entry,
            path,
            cols_spec=cols,
            max_points=max_points,
            dim_slice_spec=slice,
            use_time_index=use_time_index,
            start=start,
            stop=stop,
        )

    @app.get("/api/files/{fid}/node/export", dependencies=guarded)
    async def export_node(
        fid: str,
        path: str,
        format: str = "csv",
        start: int = 0,
        stop: int | None = None,
        cols: str | None = None,
        slice: str | None = None,
        use_time_index: bool = False,
    ):
        entry = get_file(fid)
        target, filename, media_type = await entry.run(
            export.export_rows,
            entry,
            path,
            fmt=format,
            start=start,
            stop=stop,
            cols_spec=cols,
            dim_slice_spec=slice,
            use_time_index=use_time_index,
        )
        return FileResponse(
            target,
            media_type=media_type,
            filename=filename,
            background=BackgroundTask(lambda: target.unlink(missing_ok=True)),
        )

    @app.post("/api/files/{fid}/close", dependencies=guarded)
    async def close_file(fid: str) -> dict[str, Any]:
        closed = app.state.registry.close(fid)
        return {"closed": closed}

    if serve_static and (STATIC_DIR / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


class SPAStaticFiles(StaticFiles):
    """Serve the built SPA, falling back to index.html for client-side routes."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except Exception:
            return await super().get_response("index.html", scope)


def _default_error_code(status: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "file_changed",
        413: "request_too_large",
    }.get(status, "error")


app = create_app()
