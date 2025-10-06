#!/usr/bin/env python3
"""Unified launcher for controller service and Gradio UI."""

from __future__ import annotations

import argparse
import sys
import threading
import time

try:
    import torch
except ImportError:
    torch = None
from typing import Tuple

import uvicorn

from controller.app import ControllerApp
from core.talk_engine import TalkEngineConfig
from ui.web.app import build_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AITuber desktop suite")
    parser.add_argument("--controller-host", default="127.0.0.1", help="Controller bind host")
    parser.add_argument("--controller-port", type=int, default=8001, help="Controller bind port")
    parser.add_argument("--ui-host", default="127.0.0.1", help="UI bind host")
    parser.add_argument("--device", default="auto", help="PyTorch device (auto, cpu, cuda, cuda:0, mps, etc.)")
    parser.add_argument("--list-devices", action="store_true", help="List detected PyTorch devices and exit")
    parser.add_argument("--ui-port", type=int, default=8000, help="UI bind port")
    parser.add_argument("--open-browser", action="store_true", help="Open browser on UI start")
    parser.add_argument("--log-level", default="info", help="uvicorn log level for the controller")
    return parser.parse_args()




def detect_available_devices() -> list[str]:
    devices = ["cpu"]
    if torch is None:
        return devices
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for idx in range(count):
                devices.append(f"cuda:{idx}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


def resolve_device(choice: str) -> str:
    choice = (choice or "auto").strip()
    if choice.lower() == "auto":
        devices = detect_available_devices()
        return devices[1] if len(devices) > 1 else "cpu"
    return choice

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

    deadline = time.time() + 10
    while time.time() < deadline:
        if server.started:
            break
        if not thread.is_alive():
            break
        time.sleep(0.1)

    if not server.started:
        server.should_exit = True
        thread.join(timeout=1)
        raise RuntimeError(f"Controller failed to start on {host}:{port}. Is the port already in use?")

    return server, thread


def main() -> None:
    args = parse_args()
    if args.list_devices:
        print("Available devices:")
        for dev in detect_available_devices():
            print(f"  - {dev}")
        sys.exit(0)

    controller_url = f"http://{args.controller_host}:{args.controller_port}"

    device_choice = resolve_device(args.device)
    print(f"Using TalkEngine device: {device_choice}")
    engine_config = TalkEngineConfig(device=device_choice)

    server, server_thread = start_controller(args.controller_host, args.controller_port, args.log_level, engine_config)
    ui = build_ui(controller_url=controller_url)
    try:
        ui.launch(
            server_name=args.ui_host,
            server_port=args.ui_port,
            inbrowser=args.open_browser,
            share=False,
            prevent_thread_lock=True,
        )
    except OSError as exc:
        server.should_exit = True
        server_thread.join(timeout=5)
        raise RuntimeError(
            f"UI failed to start on {args.ui_host}:{args.ui_port}: {exc}. Try specifying a different --ui-port."
        ) from exc

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
