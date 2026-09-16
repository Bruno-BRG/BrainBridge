"""Helpers compartilhados das rotas (erros dominio -> HTTP)."""

import dataclasses
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [asdict(item) for item in obj]
    return obj


def run_controller(fn: Callable[[], Any]) -> Any:
    """Executa controller e traduz ValueError/FileNotFoundError p/ HTTP."""
    try:
        return fn()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def ok(data: Any = None, status_code: int = 200) -> JSONResponse:
    return JSONResponse({"ok": True, "data": asdict(data)}, status_code=status_code)
