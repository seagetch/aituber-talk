"""API surface for the controller FastAPI app."""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from controller.session import Session
from plugins.base import Mode


class SessionCreateRequest(BaseModel):
    mode: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None


class SessionCommandRequest(BaseModel):
    command: str
    payload: Dict[str, Any] = Field(default_factory=dict)


def attach_routes(app: FastAPI, controller) -> None:
    router = APIRouter()

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - FastAPI lifecycle hook
        controller.start_engine()

    @router.get("/modes")
    async def list_modes() -> Dict[str, Any]:
        return {"modes": list(controller.registry.list_modes().values())}

    @router.post("/files")
    async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        suffix = Path(file.filename).suffix
        dest = controller.upload_dir / f"{uuid.uuid4().hex}{suffix}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"path": str(dest), "filename": file.filename}

    @router.post("/sessions")
    async def create_session(req: SessionCreateRequest) -> Dict[str, Any]:
        if req.mode not in controller.registry.list_modes():
            raise HTTPException(status_code=404, detail=f"Mode '{req.mode}' not found")
        session = controller.sessions.create(
            req.mode,
            session_id=req.session_id,
            metadata=req.payload.copy() if isinstance(req.payload, dict) else {},
        )
        mode = controller.get_mode(req.mode)
        try:
            result = mode.start(session, req.payload)
        except Exception as exc:
            controller.sessions.update_status(session.id, "error")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status = result.get("status", "running")
        controller.sessions.update_status(session.id, status)
        return {"session": session.as_dict(), "result": result}

    @router.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        return {"sessions": [s.as_dict() for s in controller.sessions.list()]}

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        session = controller.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        mode = controller.get_mode(session.mode)
        return {"session": session.as_dict(), "status": mode.status(session)}

    @router.post("/sessions/{session_id}/commands")
    async def command_session(session_id: str, req: SessionCommandRequest) -> Dict[str, Any]:
        session = controller.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        mode = controller.get_mode(session.mode)
        command = req.command.lower()
        result: Optional[Dict[str, Any]]
        if command == "pause":
            result = mode.pause(session)
            if result is None:
                raise HTTPException(status_code=400, detail="Nothing to pause")
            controller.sessions.update_status(session.id, "paused")
        elif command == "resume":
            result = mode.resume(session)
            if result is None:
                raise HTTPException(status_code=400, detail="Nothing to resume")
            controller.sessions.update_status(session.id, "running")
        elif command == "stop":
            result = mode.stop(session)
            if result is None:
                raise HTTPException(status_code=400, detail="Nothing to stop")
            controller.sessions.update_status(session.id, "stopped")
        elif command == "status":
            result = mode.status(session)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported command: {command}")
        return {"session": session.as_dict(), "result": result}

    @router.get("/events")
    async def stream_events(request: Request):
        q, unsubscribe = controller.event_bus.subscribe_queue("*")

        def _await_event() -> Optional[tuple[str, Dict[str, Any]]]:
            try:
                return q.get(timeout=0.5)
            except queue.Empty:
                return None

        async def event_iterator():
            loop = asyncio.get_running_loop()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    item = await loop.run_in_executor(None, _await_event)
                    if item is None:
                        continue
                    event_type, payload = item
                    chunk = f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    yield chunk
            finally:
                unsubscribe()

        return StreamingResponse(event_iterator(), media_type="text/event-stream")

    app.include_router(router)


__all__ = ["attach_routes", "SessionCreateRequest", "SessionCommandRequest"]

