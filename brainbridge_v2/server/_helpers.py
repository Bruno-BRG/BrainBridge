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
    """Executa controller e traduz excecoes de dominio/infra p/ HTTP JSON.

    Respostas via HTTPException passam pelo middleware CORS, entao o
    frontend recebe o detalhe em vez de "Failed to fetch" (que ocorre
    quando a excecao escapa ate o handler 500 texto puro do Starlette,
    fora do CORS).
    """
    try:
        return fn()
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc) or "Dependencia de inferencia indisponivel.",
        )
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=str(exc) or "Falha ao carregar o modelo."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=str(exc) or "Erro interno do servidor."
        )


def ok(data: Any = None, status_code: int = 200) -> JSONResponse:
    return JSONResponse({"ok": True, "data": asdict(data)}, status_code=status_code)
