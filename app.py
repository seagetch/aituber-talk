#!/usr/bin/env python3
"""Unified launcher for controller service and Gradio UI."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Tuple

import uvicorn

from controller.app import ControllerApp
from ui.web.app import build_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AITuber desktop suite")
    parser.add_argument("--controller-host", default="127.0.0.1", help="Controller bind host")
    parser.add_argument("--controller-port", type=int, default=8001, help="Controller bind port")
    parser.add_argument("--ui-host", default="127.0.0.1", help="UI bind host")
    parser.add_argument("--ui-port", type=int, default=8000, help="UI bind port")
    parser.add_argument("--open-browser", action="store_true", help="Open browser on UI start")
    parser.add_argument("--log-level", default="info", help="uvicorn log level for the controller")
    return parser.parse_args()


def start_controller(host: str, port: int, log_level: str) -> Tuple[uvicorn.Server, threading.Thread]:
    controller = ControllerApp()
    config = uvicorn.Config(
        app=controller.fastapi,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="ControllerServer", daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    args = parse_args()
    controller_url = f"http://{args.controller_host}:{args.controller_port}"

    server, server_thread = start_controller(args.controller_host, args.controller_port, args.log_level)
    ui = build_ui(controller_url=controller_url)
    ui.launch(
        server_name=args.ui_host,
        server_port=args.ui_port,
        inbrowser=args.open_browser,
        share=False,
        prevent_thread_lock=True,
    )

    print("AITuber Desktop Suite running.")
    print(f"  Controller API: {controller_url}")
    print(f"  UI: http://{args.ui_host}:{args.ui_port}/")
    print("Press Ctrl+C to stop.")

    try:
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.should_exit = True
        try:
            ui.close()
        except Exception:
            pass
        server_thread.join(timeout=5)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Failed to launch suite: {exc}", file=sys.stderr)
        sys.exit(1)
