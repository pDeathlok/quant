"""Pin a request to one committed output generation."""

from pathlib import Path

from starlette.responses import FileResponse, JSONResponse

from quant.infrastructure.publication import PublicationError, publication_context
from quant.infrastructure.artifact_registry import publication_read_lease


class PublicationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from quant.webapp.services import _publication_store

        try:
            store = _publication_store()
            headers = dict(scope.get("headers", []))
            requested = headers.get(b"x-quant-generation", b"").decode("ascii", errors="ignore") or None
            control = scope["path"].startswith("/api/selector/refresh-latest") or scope["path"] == "/api/health"
            with publication_read_lease(store.root, lambda: store.view(None if control else requested)) as view, publication_context(view):
                async def send_pinned(message):
                    if message["type"] == "http.response.start":
                        message = {**message, "headers": [
                            *message.get("headers", []),
                            (b"x-quant-generation", (view.generation or "legacy").encode()),
                        ]}
                    await send(message)

                if scope["path"].startswith("/data/") and view.generation:
                    relative = Path(scope["path"].lstrip("/"))
                    if ".." in relative.parts:
                        return await JSONResponse({"detail": "Invalid path"}, 400)(scope, receive, send)
                    logical = store.root / "web" / relative
                    resolved = view.resolve(logical)
                    if resolved != logical:
                        response = FileResponse(resolved) if resolved.is_file() else JSONResponse(
                            {"detail": "No committed result"}, 404
                        )
                        return await response(scope, receive, send_pinned)

                await self.app(scope, receive, send_pinned)
        except PublicationError as exc:
            await JSONResponse({"detail": str(exc)}, status_code=409)(scope, receive, send)
