"""Presentation mode plugin integrating PowerPoint and TalkEngine."""

from __future__ import annotations

import re
import threading
import time
from io import BytesIO
from multiprocessing import Process, Queue
from queue import Empty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

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
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
}


@dataclass
class _PresentationState:
    thread: threading.Thread
    stop_event: threading.Event
    slides: List[Tuple[str, str]]
    ppt_path: Optional[str]
    style_id: int
    wait_seconds: float
    timeout_sec: float
    image_viewer: bool = False
    loop: bool = False
    loop_delay: float = 0.0
    loop_count: int = 0
    current_index: int = 0
    current_title: Optional[str] = None


def _resolve_image_source(src: str, base_dir: Path) -> str:
    src = src.strip()
    if re.match(r"^https?://", src):
        return src
    return str((base_dir / src).expanduser().resolve())


def _has_markdown_images(slides: List[Tuple[str, str]]) -> bool:
    return any(MARKDOWN_IMAGE_PATTERN.search(body or "") for _, body in slides)


def _first_markdown_image_source(slides: List[Tuple[str, str]], base_dir: Path) -> Optional[str]:
    for _, body in slides:
        match = MARKDOWN_IMAGE_PATTERN.search(body or "")
        if match:
            return _resolve_image_source(match.group(1), base_dir)
    return None


def _resolve_image_viewer_setting(value: Any, slides: List[Tuple[str, str]]) -> Tuple[bool, str]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"on", "true", "yes", "1"}:
            return True, "on"
        if normalized in {"off", "false", "no", "0"}:
            return False, "off"
    elif value is not None:
        return bool(value), "on" if bool(value) else "off"
    return _has_markdown_images(slides), "auto"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _image_viewer_worker(commands: "Queue[Optional[str]]", initial_image: Optional[str] = None) -> None:
    import pygame
    from PIL import Image

    def load_image(src: str):
        if re.match(r"^https?://", src):
            resp = requests.get(src, headers=REQUEST_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.content
        else:
            data = Path(src).read_bytes()
        image = Image.open(BytesIO(data)).convert("RGBA")
        image.load()
        return pygame.image.fromstring(image.tobytes(), image.size, image.mode)

    pygame.init()
    original = None
    size_locked = False
    if initial_image:
        try:
            original = load_image(initial_image)
            initial_size = original.get_size()
            size_locked = True
        except Exception as exc:
            print(f"Image viewer failed to load initial image {initial_image}: {exc}")
            initial_size = (800, 600)
    else:
        initial_size = (800, 600)
    screen = pygame.display.set_mode(initial_size, pygame.RESIZABLE)
    pygame.display.set_caption("AITuber Presentation Images")
    running = True

    def redraw() -> None:
        if original is None:
            screen.fill((0, 0, 0))
            pygame.display.flip()
            return
        w, h = screen.get_size()
        sw, sh = original.get_size()
        if sw <= 0 or sh <= 0:
            return
        ratio = min(w / sw, h / sh)
        size = (max(1, int(sw * ratio)), max(1, int(sh * ratio)))
        image = pygame.transform.smoothscale(original, size)
        screen.fill((0, 0, 0))
        screen.blit(image, image.get_rect(center=(w // 2, h // 2)))
        pygame.display.flip()

    try:
        redraw()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    redraw()
            try:
                command = commands.get(timeout=0.05)
            except Empty:
                continue
            if command is None:
                running = False
                continue
            try:
                original = load_image(command)
                if not size_locked:
                    screen = pygame.display.set_mode(original.get_size(), pygame.RESIZABLE)
                    size_locked = True
                redraw()
            except Exception as exc:
                print(f"Image viewer failed to load {command}: {exc}")
    finally:
        pygame.quit()


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
        image_viewer, image_viewer_mode = _resolve_image_viewer_setting(payload.get("image_viewer", "auto"), slides)
        loop = _as_bool(payload.get("loop", False))
        loop_delay = max(float(payload.get("loop_delay", 0.0)), 0.0)

        if session.metadata is not None:
            session.metadata.update({
                "mode": "present",
                "script_path": str(path),
                "ppt_path": ppt_path,
                "image_viewer": image_viewer,
                "image_viewer_mode": image_viewer_mode,
                "loop": loop,
                "loop_delay": loop_delay,
                "slides_total": len(slides),
            })

        stop_event = threading.Event()
        state = _PresentationState(
            thread=threading.Thread(
                target=self._run_presentation,
                args=(
                    session,
                    slides,
                    stop_event,
                    style_id,
                    wait_seconds,
                    timeout_sec,
                    ppt_path,
                    path.parent,
                    image_viewer,
                    loop,
                    loop_delay,
                ),
                name=f"PresentMode-{session.id}",
                daemon=True,
            ),
            stop_event=stop_event,
            slides=slides,
            ppt_path=ppt_path,
            style_id=style_id,
            wait_seconds=wait_seconds,
            timeout_sec=timeout_sec,
            image_viewer=image_viewer,
            loop=loop,
            loop_delay=loop_delay,
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
                    "image_viewer": image_viewer,
                    "image_viewer_mode": image_viewer_mode,
                    "loop": loop,
                    "loop_delay": loop_delay,
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
                    "image_viewer": state.image_viewer,
                    "loop": state.loop,
                    "loop_delay": state.loop_delay,
                    "loop_count": state.loop_count,
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
        script_dir: Path,
        image_viewer: bool,
        loop: bool,
        loop_delay: float,
    ) -> None:
        ppt = None
        image_commands: Optional["Queue[Optional[str]]"] = None
        image_process: Optional[Process] = None
        if image_viewer:
            image_commands = Queue()
            initial_image = _first_markdown_image_source(slides, script_dir)
            image_process = Process(target=_image_viewer_worker, args=(image_commands, initial_image), daemon=True)
            image_process.start()

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
            loop_count = 0
            while True:
                loop_count += 1
                with self._lock:
                    state = self._states.get(session.id)
                    if state:
                        state.loop_count = loop_count
                for index, (title, body) in enumerate(slides, 1):
                    if stop_event.is_set():
                        final_status = "stopped"
                        break
                    with self._lock:
                        state = self._states.get(session.id)
                        if state:
                            state.current_index = index
                            state.current_title = title
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
                            {"session": session.id, "slide": index, "title": title, "loop": loop_count},
                        )
                    if image_commands is not None:
                        match = MARKDOWN_IMAGE_PATTERN.search(body or "")
                        if match:
                            image_src = _resolve_image_source(match.group(1), script_dir)
                            image_commands.put(image_src)
                            if self.event_bus:
                                self.event_bus.publish(
                                    "presentation_image_shown",
                                    {"session": session.id, "slide": index, "image": image_src, "loop": loop_count},
                                )
                    clean_body = IMAGE_PATTERN.sub("", body or "").strip()
                    status = "skipped"
                    if clean_body:
                        request_id = f"{session.id}-{loop_count}-{index}"
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
                                "loop": loop_count,
                            },
                        )
                    if stop_event.is_set():
                        final_status = "stopped"
                        break
                    if stop_event.wait(max(wait_seconds, 0.0)):
                        final_status = "stopped"
                        break
                if final_status == "stopped" or not loop:
                    break
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_loop_waiting",
                        {"session": session.id, "loop": loop_count, "delay": loop_delay},
                    )
                if stop_event.wait(loop_delay):
                    final_status = "stopped"
                    break
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
            if image_commands is not None:
                image_commands.put(None)
            if image_process is not None:
                image_process.join(timeout=3)
                if image_process.is_alive():
                    image_process.terminate()


__all__ = ["PresentMode"]
