"""HTTP delivery policy for the static strategy workspace."""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class StaticAssetCacheMiddleware:
    """Set explicit cache policy for HTML and versioned static resources."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")

        async def send_with_cache(message: Message) -> None:
            if (
                message["type"] == "http.response.start"
                and int(message["status"]) == 200
            ):
                headers = MutableHeaders(scope=message)
                if path == "/" or path.endswith(".html"):
                    headers["Cache-Control"] = "no-cache"
                elif path.endswith((".js", ".css")):
                    headers["Cache-Control"] = "public, max-age=3600"
            await send(message)

        await self.app(scope, receive, send_with_cache)
