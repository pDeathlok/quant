from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quant.routine.paths import PROJECT_ROOT
from quant.webapp.api import router


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
    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()

