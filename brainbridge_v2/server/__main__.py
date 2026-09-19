"""Executa o backend: `python -m brainbridge_v2.server [--port 8000]`."""

import argparse
import sys

# O sidecar Tauri tem stdout/stderr em pipe com encoding cp1252 no Windows:
# qualquer `print()` com acento/emoji (ex.: "[ESP32] ✗ ...") estourava a
# requisicao com UnicodeEncodeError. Forca UTF-8 antes de qualquer output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(description="BrainBridge API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("brainbridge_v2.server.app:get_app", factory=True,
                host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
