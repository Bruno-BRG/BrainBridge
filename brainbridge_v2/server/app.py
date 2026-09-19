"""Fabrica do app FastAPI (ponto de entrada do backend Tauri/web)."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from brainbridge_v2.bootstrap.container import AppContainer, build_app_container
from brainbridge_v2.server import routers, ws
from brainbridge_v2.server.state import ServerState

logger = logging.getLogger(__name__)


def create_app(container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title="BrainBridge API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    resolved = container or build_app_container()
    try:
        # Warm-up do TensorFlow no startup para o primeiro load nao pagar
        # o custo do import sob a requisicao.
        resolved.inference_controller.preload()
    except Exception as exc:
        logger.warning("Preload do TensorFlow falhou no startup: %s", exc)
    app.state.brainbridge = ServerState(container=resolved)
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
