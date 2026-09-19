#!/usr/bin/env python3
"""
BrainBridge - ponto de entrada do pacote.

A antiga interface grafica PyQt5 foi removida: o frontend agora e o
aplicativo desktop Tauri (`webui/`), que fala com o backend FastAPI
(`python -m brainbridge_v2.server`).
"""

import argparse
import sys


def check_environment():
    """Verifica o ambiente e dependências (sem Qt)."""
    print("=== Verificação do Ambiente ===")
    print(f"Python: {sys.version}")

    dependencies = (
        "numpy",
        "scipy",
        "tensorflow",
        "fastapi",
        "uvicorn",
    )
    for module in dependencies:
        try:
            __import__(module)
            print(f"OK {module}: instalado")
        except Exception as exc:
            print(f"X {module}: não instalado ou falhou ao inicializar ({exc})")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="BrainBridge - a interface Qt foi removida; use o app Tauri.",
    )
    parser.add_argument(
        "--check-env", action="store_true",
        help="Verificar ambiente e dependências",
    )
    parser.add_argument("--version", action="version", version="BrainBridge 2.0.0")
    args = parser.parse_args()

    if args.check_env:
        check_environment()
        return 0

    print(
        "A interface grafica PyQt5 foi removida do BrainBridge.\n"
        "Para usar o sistema, abra o aplicativo desktop Tauri ou suba o "
        "backend com: python -m brainbridge_v2.server"
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)
