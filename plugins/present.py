"""Presentation mode plugin integrating PowerPoint and TalkEngine."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from controller.events import EventBus
from controller.session import Session, SessionManager
from core.talk_engine import TalkEngine
from plugins.base import Mode

try:  # Optional Windows dependency
    from lib.ppt_control import PowerPointController
except Exception:  # pragma: no cover - pywin32 not installed or platform mismatch
    PowerPointController = None  # type: ignore


SLIDE_SPLIT_RE = re.compile(r"^(#+)\s*(.*)")
IMAGE_PATTERN = re.compile(r"!?\[.*?\]\(.*?\)")


@dataclass
class _PresentationState:
    thread: threading.Thread
    stop_event: threading.Event
    slides: List[Tuple[str, str]]
    ppt_path: Optional[str]
    style_id: int
    wait_seconds: float
    timeout_sec: float
    current_index: int = 0
    current_title: Optional[str] = None


def _parse_script(script_path: Path) -> List[Tuple[str, str]]:
    lines = script_path.read_text(encoding="utf-8").splitlines(True)
    slides: List[Tuple[str, str]] = []
    current_title = ""
    current_body: List[str] = []
    for line in lines:
        match = SLIDE_SPLIT_RE.match(line)
        if match:
            if current_title or current_body:
                slides.append((current_title, "".join(current_body).strip()))
            current_title = match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_title or current_body:
        slides.append((current_title, "".join(current_body).strip()))
    return slides


class PresentMode(Mode):
    name = "present"
    description = "Drive a slide deck and narrate it via the shared TalkEngine."

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
        self._states: Dict[str, _PresentationState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def start(self, session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
        script_path = payload.get("script_path")
        if not script_path:
            raise ValueError("'script_path' is required")
        path = Path(script_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Script not found: {path}")

        slides = _parse_script(path)
        if not slides:
            raise ValueError("Script has no readable slides")

        style_id = payload.get("style_id") or self.engine.config.default_style_id
        wait_seconds = float(payload.get("wait", 1.0))
        timeout_sec = float(payload.get("timeout", 300.0))
        ppt_path = payload.get("ppt_path")

        if session.metadata is not None:
            session.metadata.update({
                "mode": "present",
                "script_path": str(path),
                "ppt_path": ppt_path,
                "slides_total": len(slides),
            })

        stop_event = threading.Event()
        state = _PresentationState(
            thread=threading.Thread(
                target=self._run_presentation,
                args=(session, slides, stop_event, style_id, wait_seconds, timeout_sec, ppt_path),
                name=f"PresentMode-{session.id}",
                daemon=True,
            ),
            stop_event=stop_event,
            slides=slides,
            ppt_path=ppt_path,
            style_id=style_id,
            wait_seconds=wait_seconds,
            timeout_sec=timeout_sec,
        )
        with self._lock:
            self._states[session.id] = state
        state.thread.start()
        if self.sessions:
            self.sessions.update_status(session.id, "running")
        if self.event_bus:
            self.event_bus.publish(
                "presentation_started",
                {
                    "session": session.id,
                    "slides": len(slides),
                    "ppt_path": ppt_path,
                    "script_path": str(path),
                },
            )
        return {"status": "running", "session": session.id, "slides": len(slides)}

    # ------------------------------------------------------------------
    def stop(self, session: Session) -> Optional[Dict[str, Any]]:
        state = self._states.get(session.id)
        if not state:
            return None
        state.stop_event.set()
        self.engine.stop()
        state.thread.join(timeout=5)
        if self.sessions:
            self.sessions.update_status(session.id, "stopped")
        if self.event_bus:
            self.event_bus.publish("presentation_stopped", {"session": session.id})
        return {"status": "stopped", "session": session.id}

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

    def status(self, session: Session) -> Dict[str, Any]:
        payload = session.as_dict()
        state = self._states.get(session.id)
        if state:
            payload.update(
                {
                    "slides_total": len(state.slides),
                    "current_slide": state.current_index,
                    "ppt_path": state.ppt_path,
                }
            )
        payload.update(self.engine.status())
        if state:
            payload["current_title"] = state.current_title
        payload["session"] = session.id
        return payload

    # ------------------------------------------------------------------
    def _run_presentation(
        self,
        session: Session,
        slides: List[Tuple[str, str]],
        stop_event: threading.Event,
        style_id: int,
        wait_seconds: float,
        timeout_sec: float,
        ppt_path: Optional[str],
    ) -> None:
        ppt = None
        if ppt_path and PowerPointController is not None:
            try:
                ppt = PowerPointController()
                ppt.open_file(ppt_path)
                ppt.start_slideshow()
                time.sleep(wait_seconds)
            except Exception as exc:  # pragma: no cover - depends on PowerPoint
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_error",
                        {
                            "session": session.id,
                            "stage": "ppt_init",
                            "error": str(exc),
                        },
                    )
                ppt = None

        final_status = "completed"
        try:
            for index, (title, body) in enumerate(slides, 1):
                if stop_event.is_set():
                    final_status = "stopped"
                    break
                with self._lock:
                    state = self._states.get(session.id)
                    if state:
                        state.current_index = index
                if ppt is not None:
                    try:
                        ppt.goto_slide(index)
                    except Exception as exc:  # pragma: no cover
                        if self.event_bus:
                            self.event_bus.publish(
                                "presentation_error",
                                {
                                    "session": session.id,
                                    "stage": "ppt_goto",
                                    "slide": index,
                                    "error": str(exc),
                                },
                            )
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_slide_started",
                        {"session": session.id, "slide": index, "title": title},
                    )
                clean_body = IMAGE_PATTERN.sub("", body or "").strip()
                status = "skipped"
                if clean_body:
                    request_id = f"{session.id}-{index}"
                    request_id = self.engine.submit_text(clean_body, style_id=style_id, request_id=request_id)
                    status = self.engine.wait_for(request_id, timeout=timeout_sec)
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_slide_completed",
                        {
                            "session": session.id,
                            "slide": index,
                            "title": title,
                            "status": status,
                        },
                    )
                if stop_event.is_set():
                    final_status = "stopped"
                    break
                time.sleep(max(wait_seconds, 0.0))
        finally:
            if self.sessions:
                self.sessions.update_status(session.id, final_status)
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_finished",
                    {
                        "session": session.id,
                        "status": final_status,
                        "slides": len(slides),
                    },
                )
            with self._lock:
                self._states.pop(session.id, None)
            if session.metadata is not None:
                session.metadata["status"] = final_status


__all__ = ["PresentMode"]
