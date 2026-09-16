"""Executa o backend: `python -m brainbridge_v2.server [--port 8000]`."""

import argparse

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
