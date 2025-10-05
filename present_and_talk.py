#!/usr/bin/env python3
"""CLI helper that triggers the controller's presentation mode."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

DEFAULT_CONTROLLER_URL = "http://127.0.0.1:8001"


def submit_presentation(
    controller_url: str,
    script_path: Path,
    ppt_path: Optional[Path],
    style_id: Optional[int],
    wait_seconds: float,
    timeout_sec: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "mode": "present",
        "payload": {
            "script_path": str(script_path),
            "wait": wait_seconds,
            "timeout": timeout_sec,
        },
    }
    if ppt_path is not None:
        payload["payload"]["ppt_path"] = str(ppt_path)
    if style_id is not None:
        payload["payload"]["style_id"] = style_id
    resp = requests.post(f"{controller_url}/sessions", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def poll_status(controller_url: str, session_id: str, interval: float = 2.0) -> None:
    endpoint = f"{controller_url}/sessions/{session_id}"
    try:
        while True:
            resp = requests.get(endpoint, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            session = data.get("session", {})
            status = session.get("status") or data.get("status")
            print(f"Session {session_id}: status={status} info={data}")
            if status in {"completed", "stopped", "error"}:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Polling interrupted by user.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger presentation mode via the controller service.")
    parser.add_argument("script", help="Path to the Markdown script")
    parser.add_argument("--ppt", dest="pptx", help="Optional PowerPoint file path")
    parser.add_argument("--style_id", type=int, default=None, help="Voice style ID override")
    parser.add_argument("--wait", type=float, default=1.0, help="Delay between slides (seconds)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Speech completion timeout per slide")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER_URL, help="Controller base URL")
    parser.add_argument("--follow", action="store_true", help="Poll the session status until completion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script).expanduser().resolve()
    if not script_path.exists():
        raise SystemExit(f"Script not found: {script_path}")
    ppt_path = Path(args.pptx).expanduser().resolve() if args.pptx else None
    if ppt_path and not ppt_path.exists():
        raise SystemExit(f"PowerPoint not found: {ppt_path}")
    try:
        response = submit_presentation(
            args.controller,
            script_path,
            ppt_path,
            args.style_id,
            args.wait,
            args.timeout,
        )
    except requests.HTTPError as exc:
        raise SystemExit(f"Failed to submit presentation: {exc}") from exc

    session_info = response.get("session", {})
    session_id = session_info.get("id")
    print(f"Controller response: {response}")
    if args.follow and session_id:
        poll_status(args.controller, session_id)


if __name__ == "__main__":
    main()
