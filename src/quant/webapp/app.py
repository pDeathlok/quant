from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from quant.core.paths import PROJECT_ROOT
from quant.webapp.api import router
from quant.webapp.static_delivery import StaticAssetCacheMiddleware


WEB_DIR = PROJECT_ROOT / "web"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant Strategy Web API",
        version="0.1.0",
        description="B1/B-family strategy dashboard and routine API.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1", "http://localhost"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
    app.add_middleware(StaticAssetCacheMiddleware)
    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
