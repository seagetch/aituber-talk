"""Talk mode plugin wrapping the shared TalkEngine."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from controller.events import EventBus
from controller.session import Session, SessionManager
from core.talk_engine import TalkEngine
from plugins.base import Mode


class TalkMode(Mode):
    name = "talk"
    description = "Queue text for the SadTalker speech + motion pipeline."

    def __init__(
        self,
        engine: TalkEngine,
        event_bus: Optional[EventBus] = None,
        sessions: Optional[SessionManager] = None,
    ) -> None:
        super().__init__(engine=engine, event_bus=event_bus, sessions=sessions)
        self.engine = engine
        self.event_bus = event_bus
        self.sessions = sessions

    def start(self, session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text")
        if not text:
            raise ValueError("'text' is required to start a talk session")
        style_id = payload.get("style_id")
        sync = bool(payload.get("sync", False))
        timeout = float(payload.get("timeout", 300.0))

        request_id = self.engine.submit_text(text, style_id=style_id, request_id=session.id)
        if self.sessions:
            self.sessions.update_status(session.id, "running" if sync else "queued")
        if self.event_bus:
            self.event_bus.publish(
                "speech_queued",
                {"session": session.id, "text": text, "style_id": style_id, "request": request_id},
            )
        if not sync:
            if self.event_bus or self.sessions:
                def _async_wait() -> None:
                    status = self.engine.wait_for(request_id, timeout=timeout)
                    if self.sessions:
                        self.sessions.update_status(session.id, status)
                    if self.event_bus:
                        self.event_bus.publish(
                            "speech_completed",
                            {
                                "session": session.id,
                                "text": text,
                                "style_id": style_id,
                                "request": request_id,
                                "status": status,
                            },
                        )

                threading.Thread(target=_async_wait, name=f"TalkModeWait-{session.id}", daemon=True).start()
            return {"status": "queued", "session": session.id, "request": request_id}
        status = self.engine.wait_for(request_id, timeout=timeout)
        if self.sessions:
            self.sessions.update_status(session.id, status)
        if self.event_bus:
            self.event_bus.publish(
                "speech_completed",
                {"session": session.id, "text": text, "style_id": style_id, "request": request_id, "status": status},
            )
        return {"status": status, "session": session.id, "request": request_id}

    def pause(self, session: Session) -> Optional[Dict[str, Any]]:
        active = self.engine.pause()
        if active is None:
            return None
        if self.sessions:
            self.sessions.update_status(session.id, "paused")
        return {"status": "paused", "session": active}

    def resume(self, session: Session) -> Optional[Dict[str, Any]]:
        resumed = self.engine.resume()
        if resumed is None:
            return None
        if self.sessions:
            self.sessions.update_status(session.id, "running")
        return {"status": "resumed", "session": resumed}

    def stop(self, session: Session) -> Optional[Dict[str, Any]]:
        stopped = self.engine.stop()
        if stopped is None:
            return None
        if self.sessions:
            self.sessions.update_status(session.id, "stopped")
        if self.event_bus:
            self.event_bus.publish("speech_stopped", {"session": stopped})
        return {"status": "stopped", "session": stopped}

    def status(self, session: Session) -> Dict[str, Any]:
        payload = self.engine.status()
        payload.update({"session": session.id})
        return payload


__all__ = ["TalkMode"]
