"""Fabrica do app FastAPI (ponto de entrada do backend Tauri/web)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from brainbridge_v2.bootstrap.container import AppContainer, build_app_container
from brainbridge_v2.server import routers, ws
from brainbridge_v2.server.state import ServerState


def create_app(container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title="BrainBridge API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.brainbridge = ServerState(
        container=container or build_app_container())
    app.include_router(routers.router)
    app.include_router(ws.router)
    return app


app = None


def get_app() -> FastAPI:
    """Singleton preguiçoso para `uvicorn brainbridge_v2.server.app:get_app`."""
    global app
    if app is None:
        app = create_app()
    return app
