#!/usr/bin/env python3
"""Legacy entry point preserved for compatibility.

This module now acts as a thin wrapper that exposes the FastAPI
endpoints backed by the refactored `TalkEngine` class. New development
should integrate with the controller service defined under
`controller/` instead of importing directly from this file.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from core.talk_engine import TalkEngine, TalkEngineConfig

app = FastAPI()
_engine_lock = threading.Lock()
engine: TalkEngine = TalkEngine()


class TalkRequest(BaseModel):
    text: str
    style_id: Optional[int] = None
    sync: bool = False
    timeout_sec: float = 300.0


@app.on_event("startup")
async def _startup() -> None:
    with _engine_lock:
        engine.start()


@app.post("/talk")
async def talk_text(request: TalkRequest):
    with _engine_lock:
        engine.start()
    req_id = engine.submit_text(request.text, style_id=request.style_id)
    if request.sync:
        status = engine.wait_for(req_id, timeout=request.timeout_sec)
        return {"status": status, "text": request.text, "session": req_id}
    return {"status": "queued", "text": request.text, "session": req_id}


@app.post("/pause")
async def pause_playback():
    session = engine.pause()
    if session is None:
        raise HTTPException(status_code=400, detail="No active session to pause")
    return {"status": "paused", "session": session}


@app.post("/resume")
async def resume_playback():
    session = engine.resume()
    if session is None:
        raise HTTPException(status_code=400, detail="No paused session to resume")
    return {"status": "resumed", "session": session}


@app.post("/stop")
async def stop_playback():
    session = engine.stop()
    if session is None:
        raise HTTPException(status_code=400, detail="No session to stop")
    return {"status": "stopped", "session": session}


@app.get("/status")
async def status_playback():
    return engine.status()


def build_engine_from_args(args: argparse.Namespace) -> TalkEngine:
    config = TalkEngineConfig(
        image_path=Path(args.image),
        default_style_id=args.style_id,
        device=args.device,
        pose_style=args.pose_style,
        no_blink=args.no_blink,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
    )
    return TalkEngine(config)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    defaults = TalkEngineConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_id", type=int, default=defaults.default_style_id)
    parser.add_argument("--image", default=str(defaults.image_path))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=defaults.device)
    parser.add_argument("--pose_style", type=int, default=defaults.pose_style)
    parser.add_argument("--no_blink", action="store_true", help="Disable blink animation")
    parser.add_argument("--osc_host", default=defaults.osc_host)
    parser.add_argument("--osc_port", type=int, default=defaults.osc_port)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=34512)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    global engine
    args = parse_args(argv)
    with _engine_lock:
        engine = build_engine_from_args(args)
        engine.start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
