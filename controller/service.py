"""Command-line service runner for the controller application."""

from __future__ import annotations

import argparse

import uvicorn

from controller.app import ControllerApp


def build_fastapi_app() -> ControllerApp:
    controller = ControllerApp()
    return controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AITuber controller service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host for the API server")
    parser.add_argument("--port", type=int, default=8001, help="Bind port for the API server")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = build_fastapi_app()
    uvicorn.run(
        controller.fastapi,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
