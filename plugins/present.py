"""Presentation mode plugin integrating PowerPoint and TalkEngine."""

from __future__ import annotations

import re
import threading
import time
from io import BytesIO
from multiprocessing import Process, Queue
from queue import Empty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from controller.events import EventBus
from controller.session import Session, SessionManager
from core.talk_engine import BufferedSpeechSegment, TalkEngine, split_sentences
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
class _BufferedSlide:
    title: str
    body: str
    clean_body: str
    segments: List[BufferedSpeechSegment]


@dataclass
class _PresentationState:
    stop_event: threading.Event
    slides: List[Tuple[str, str]]
    script_path: str
    script_dir: Path
    ppt_path: Optional[str]
    style_id: int
    wait_seconds: float
    timeout_sec: float
    thread: Optional[threading.Thread] = None
    play_thread: Optional[threading.Thread] = None
    image_viewer: bool = False
    loop: bool = False
    loop_delay: float = 0.0
    loop_count: int = 0
    current_index: int = 0
    current_title: Optional[str] = None
    all_buffered: bool = False
    buffer_status: str = "disabled"
    buffer_total: int = 0
    buffer_done: int = 0
    segments_total: int = 0
    segments_done: int = 0
    generation_steps_done: int = 0
    generation_steps_total: int = 0
    current_segment: int = 0
    current_segment_text: Optional[str] = None
    generation_stage: Optional[str] = None
    buffer_duration: float = 0.0
    buffered_slides: List[_BufferedSlide] = field(default_factory=list)


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


def _image_viewer_worker(
    commands: "Queue[Optional[str]]",
    initial_image: Optional[str] = None,
    ready: Optional["Queue[Dict[str, Any]]"] = None,
) -> None:
    import pygame
    from PIL import Image

    ready_sent = False

    def notify_ready(status: str, **payload: Any) -> None:
        nonlocal ready_sent
        if ready is None or ready_sent:
            return
        ready.put({"status": status, **payload})
        ready_sent = True

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
        notify_ready(
            "ready",
            image=initial_image,
            loaded=original is not None,
            size=screen.get_size(),
        )
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
    except Exception as exc:
        notify_ready("error", error=str(exc), image=initial_image)
        raise
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
        all_buffered = _as_bool(payload.get("all_buffered", False))

        if session.metadata is not None:
            session.metadata.update({
                "mode": "present",
                "script_path": str(path),
                "ppt_path": ppt_path,
                "image_viewer": image_viewer,
                "image_viewer_mode": image_viewer_mode,
                "loop": loop,
                "loop_delay": loop_delay,
                "all_buffered": all_buffered,
                "buffer_status": "buffering" if all_buffered else "disabled",
                "buffer_total": len(slides) if all_buffered else 0,
                "buffer_done": 0,
                "slides_total": len(slides),
            })

        stop_event = threading.Event()
        state = _PresentationState(
            stop_event=stop_event,
            slides=slides,
            script_path=str(path),
            script_dir=path.parent,
            ppt_path=ppt_path,
            style_id=style_id,
            wait_seconds=wait_seconds,
            timeout_sec=timeout_sec,
            image_viewer=image_viewer,
            loop=loop,
            loop_delay=loop_delay,
            all_buffered=all_buffered,
            buffer_status="buffering" if all_buffered else "disabled",
            buffer_total=len(slides) if all_buffered else 0,
        )
        if all_buffered:
            state.thread = threading.Thread(
                target=self._prepare_buffered_presentation,
                args=(session, state),
                name=f"PresentBuffer-{session.id}",
                daemon=True,
            )
        else:
            state.thread = threading.Thread(
                target=self._run_presentation,
                args=(session, state, None),
                name=f"PresentMode-{session.id}",
                daemon=True,
            )
        with self._lock:
            self._states[session.id] = state
        if self.sessions:
            self.sessions.update_status(session.id, "buffering" if all_buffered else "running")
        if state.thread:
            state.thread.start()
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
                    "all_buffered": all_buffered,
                    "script_path": str(path),
                },
            )
        return {"status": "buffering" if all_buffered else "running", "session": session.id, "slides": len(slides)}

    # ------------------------------------------------------------------
    def stop(self, session: Session) -> Optional[Dict[str, Any]]:
        state = self._states.get(session.id)
        if not state:
            return None
        state.stop_event.set()
        self.engine.stop()
        if state.thread:
            state.thread.join(timeout=5)
        if state.play_thread:
            state.play_thread.join(timeout=5)
        with self._lock:
            self._states.pop(session.id, None)
            state.buffered_slides.clear()
            state.buffer_status = "stopped"
        if session.metadata is not None:
            session.metadata["buffer_status"] = "stopped"
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

    def play(self, session: Session) -> Optional[Dict[str, Any]]:
        state = self._states.get(session.id)
        if not state or not state.all_buffered:
            return None
        if state.buffer_status == "running":
            return {"status": "running", "session": session.id}
        if state.buffer_status not in {"ready", "completed"}:
            return {"status": state.buffer_status, "session": session.id}
        with self._lock:
            state.stop_event.clear()
            state.current_index = 0
            state.current_title = None
            state.loop_count = 0
            state.buffer_status = "running"
            state.play_thread = threading.Thread(
                target=self._run_presentation,
                args=(session, state, state.buffered_slides),
                name=f"PresentPlay-{session.id}",
                daemon=True,
            )
        if session.metadata is not None:
            session.metadata["buffer_status"] = "running"
        if self.sessions:
            self.sessions.update_status(session.id, "running")
        if self.event_bus:
            self.event_bus.publish("presentation_buffered_play_started", {"session": session.id})
        if state.play_thread:
            state.play_thread.start()
        return {"status": "running", "session": session.id}

    def status(self, session: Session) -> Dict[str, Any]:
        payload = session.as_dict()
        state = self._states.get(session.id)
        if state:
            buffer_progress = (
                state.generation_steps_done / state.generation_steps_total
                if state.generation_steps_total
                else state.segments_done / state.segments_total
                if state.segments_total
                else ((state.buffer_done / state.buffer_total) if state.buffer_total else 0.0)
            )
            payload.update(
                {
                    "slides_total": len(state.slides),
                    "current_slide": state.current_index,
                    "ppt_path": state.ppt_path,
                    "image_viewer": state.image_viewer,
                    "loop": state.loop,
                    "loop_delay": state.loop_delay,
                    "loop_count": state.loop_count,
                    "all_buffered": state.all_buffered,
                    "buffer_status": state.buffer_status,
                    "buffer_total": state.buffer_total,
                    "buffer_done": state.buffer_done,
                    "segments_total": state.segments_total,
                    "segments_done": state.segments_done,
                    "generation_steps_total": state.generation_steps_total,
                    "generation_steps_done": state.generation_steps_done,
                    "current_segment": state.current_segment,
                    "current_segment_text": state.current_segment_text,
                    "generation_stage": state.generation_stage,
                    "buffer_duration": state.buffer_duration,
                    "buffer_progress": buffer_progress,
                    "play_ready": state.all_buffered and state.buffer_status in {"ready", "completed"},
                    "buffering": {
                        "enabled": state.all_buffered,
                        "status": state.buffer_status,
                        "play_ready": state.all_buffered and state.buffer_status in {"ready", "completed"},
                        "slides_total": state.buffer_total,
                        "slides_done": state.buffer_done,
                        "segments_total": state.segments_total,
                        "segments_done": state.segments_done,
                        "generation_steps_total": state.generation_steps_total,
                        "generation_steps_done": state.generation_steps_done,
                        "current_segment": state.current_segment,
                        "current_segment_text": state.current_segment_text,
                        "stage": state.generation_stage,
                        "progress": buffer_progress,
                        "script_path": state.script_path,
                        "current_slide": state.current_index,
                        "current_title": state.current_title,
                        "duration": state.buffer_duration,
                    },
                }
            )
        payload.update(self.engine.status())
        if state:
            payload["current_title"] = state.current_title
        payload["session"] = session.id
        return payload

    # ------------------------------------------------------------------
    def _prepare_buffered_presentation(self, session: Session, state: _PresentationState) -> None:
        try:
            clean_bodies = [IMAGE_PATTERN.sub("", body or "").strip() for _, body in state.slides]
            segments_total = sum(len(split_sentences(clean_body)) for clean_body in clean_bodies)
            with self._lock:
                state.segments_total = segments_total
                state.generation_steps_total = segments_total * 2
                state.buffer_status = "buffering"
            if session.metadata is not None:
                session.metadata.update(
                    {
                        "buffer_status": "buffering",
                        "buffer_total": len(state.slides),
                        "buffer_done": 0,
                        "segments_total": segments_total,
                        "segments_done": 0,
                        "generation_steps_total": segments_total * 2,
                        "generation_steps_done": 0,
                    }
                )
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_buffering_started",
                    {
                        "session": session.id,
                        "slides_total": len(state.slides),
                        "segments_total": segments_total,
                        "generation_steps_total": segments_total * 2,
                    },
                )

            buffered_slides: List[_BufferedSlide] = []
            segments_done = 0
            generation_steps_done = 0
            duration = 0.0
            for index, ((title, body), clean_body) in enumerate(zip(state.slides, clean_bodies), 1):
                if state.stop_event.is_set():
                    self._finish_buffering(session, state, "stopped")
                    return
                with self._lock:
                    state.current_index = index
                    state.current_title = title
                if session.metadata is not None:
                    session.metadata.update({"current_slide": index, "current_title": title})
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_buffering_slide_started",
                        {
                            "session": session.id,
                            "slide": index,
                            "slides_total": len(state.slides),
                            "segments_total": segments_total,
                            "segments_done": segments_done,
                            "progress": (segments_done / segments_total) if segments_total else (index - 1) / len(state.slides),
                            "title": title,
                        },
                    )

                def publish_buffering_progress(title_text: str, segment_index: int, stage: str) -> None:
                    progress = (
                        generation_steps_done / (segments_total * 2)
                        if segments_total
                        else (index / len(state.slides))
                    )
                    if self.event_bus:
                        self.event_bus.publish(
                            "presentation_buffering_progress",
                            {
                                "session": session.id,
                                "slide": index,
                                "slides_total": len(state.slides),
                                "slides_done": index - 1,
                                "segments_total": segments_total,
                                "segments_done": segments_done,
                                "generation_steps_total": segments_total * 2,
                                "generation_steps_done": generation_steps_done,
                                "current_segment": segment_index,
                                "stage": stage,
                                "progress": progress,
                                "script_path": state.script_path,
                                "title": title,
                                "text": title_text,
                            },
                        )

                def on_progress(segment_index: int, stage: str, text: str) -> None:
                    nonlocal generation_steps_done
                    if stage in {"tts_completed", "sadtalker_completed"}:
                        generation_steps_done += 1
                    absolute_segment = segments_done + 1
                    with self._lock:
                        state.current_segment = absolute_segment
                        state.current_segment_text = text
                        state.generation_stage = stage
                        state.generation_steps_done = generation_steps_done
                    if session.metadata is not None:
                        session.metadata.update(
                            {
                                "current_segment": absolute_segment,
                                "current_segment_text": text,
                                "generation_stage": stage,
                                "generation_steps_done": generation_steps_done,
                            }
                        )
                    publish_buffering_progress(text, absolute_segment, stage)

                def on_segment(segment: BufferedSpeechSegment) -> None:
                    nonlocal segments_done, duration
                    segments_done += 1
                    duration += segment.duration
                    with self._lock:
                        state.segments_done = segments_done
                        state.buffer_duration = duration
                    if session.metadata is not None:
                        session.metadata.update(
                            {
                                "segments_done": segments_done,
                                "buffer_duration": duration,
                            }
                        )

                segments = (
                    self.engine.prepare_text_segments(
                        clean_body,
                        style_id=state.style_id,
                        request_id=f"{session.id}-buffer-{index}",
                        on_segment=on_segment,
                        on_progress=on_progress,
                    )
                    if clean_body
                    else []
                )
                buffered_slides.append(_BufferedSlide(title=title, body=body, clean_body=clean_body, segments=segments))
                with self._lock:
                    state.buffered_slides = buffered_slides
                    state.buffer_done = index
                    state.segments_done = segments_done
                    state.generation_steps_done = generation_steps_done
                    state.buffer_duration = duration
                if session.metadata is not None:
                    session.metadata.update(
                        {
                            "buffer_done": index,
                            "segments_done": segments_done,
                            "generation_steps_done": generation_steps_done,
                            "buffer_duration": duration,
                        }
                    )
                progress = (
                    generation_steps_done / (segments_total * 2)
                    if segments_total
                    else (index / len(state.slides))
                )
                if self.event_bus:
                    payload = {
                        "session": session.id,
                        "slide": index,
                        "slides_total": len(state.slides),
                        "slides_done": index,
                        "segments_total": segments_total,
                        "segments_done": segments_done,
                        "generation_steps_total": segments_total * 2,
                        "generation_steps_done": generation_steps_done,
                        "progress": progress,
                        "title": title,
                    }
                    self.event_bus.publish("presentation_buffering_slide_completed", payload)
                    self.event_bus.publish("presentation_buffering_progress", payload)

            with self._lock:
                state.current_index = 0
                state.current_title = None
                state.buffer_status = "ready"
            if session.metadata is not None:
                session.metadata.update({"buffer_status": "ready", "current_slide": 0, "current_title": None})
            if self.sessions:
                self.sessions.update_status(session.id, "ready")
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_buffering_completed",
                    {
                        "session": session.id,
                        "slides_total": len(state.slides),
                        "slides_done": len(state.slides),
                        "segments_total": segments_total,
                        "segments_done": segments_done,
                        "generation_steps_total": segments_total * 2,
                        "generation_steps_done": generation_steps_done,
                        "progress": 1.0,
                        "duration": duration,
                    },
                )
        except Exception as exc:
            with self._lock:
                state.buffered_slides.clear()
                state.buffer_status = "error"
            if session.metadata is not None:
                session.metadata["buffer_status"] = "error"
            if self.sessions:
                self.sessions.update_status(session.id, "error")
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_buffering_error",
                    {"session": session.id, "error": str(exc)},
                )

    def _finish_buffering(self, session: Session, state: _PresentationState, status: str) -> None:
        with self._lock:
            state.buffered_slides.clear()
            state.buffer_status = status
        if session.metadata is not None:
            session.metadata["buffer_status"] = status
        if self.sessions:
            self.sessions.update_status(session.id, status)

    def _run_presentation(
        self,
        session: Session,
        state: _PresentationState,
        buffered_slides: Optional[List[_BufferedSlide]],
    ) -> None:
        ppt = None
        image_commands: Optional["Queue[Optional[str]]"] = None
        image_ready: Optional["Queue[Dict[str, Any]]"] = None
        image_process: Optional[Process] = None
        if state.image_viewer:
            image_commands = Queue()
            image_ready = Queue()
            initial_image = _first_markdown_image_source(state.slides, state.script_dir)
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_image_viewer_starting",
                    {"session": session.id, "image": initial_image},
                )
            image_process = Process(
                target=_image_viewer_worker,
                args=(image_commands, initial_image, image_ready),
                daemon=True,
            )
            image_process.start()
            viewer_ready = False
            viewer_status: Dict[str, Any] = {}
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline and not state.stop_event.is_set():
                if image_process is not None and not image_process.is_alive():
                    break
                try:
                    viewer_status = image_ready.get(timeout=0.1)
                    viewer_ready = viewer_status.get("status") == "ready"
                    break
                except Empty:
                    continue
            if self.event_bus:
                event_name = "presentation_image_viewer_ready" if viewer_ready else "presentation_image_viewer_error"
                self.event_bus.publish(
                    event_name,
                    {
                        "session": session.id,
                        "image": initial_image,
                        **viewer_status,
                    },
                )
            if not viewer_ready:
                stopped = state.stop_event.is_set()
                if image_commands is not None:
                    image_commands.put(None)
                if image_process is not None:
                    image_process.join(timeout=1)
                    if image_process.is_alive():
                        image_process.terminate()
                with self._lock:
                    if state.all_buffered:
                        state.buffer_status = "stopped" if stopped else "error"
                    else:
                        self._states.pop(session.id, None)
                if session.metadata is not None:
                    session.metadata["status"] = "stopped" if stopped else "error"
                    if state.all_buffered:
                        session.metadata["buffer_status"] = "stopped" if stopped else "error"
                if self.sessions:
                    self.sessions.update_status(session.id, "stopped" if stopped else "error")
                if self.event_bus and not stopped:
                    self.event_bus.publish(
                        "presentation_error",
                        {
                            "session": session.id,
                            "stage": "image_viewer_startup",
                            "error": viewer_status.get("error", "Image viewer did not become ready before timeout"),
                        },
                    )
                return

        if state.ppt_path and PowerPointController is not None:
            try:
                ppt = PowerPointController()
                ppt.open_file(state.ppt_path)
                ppt.start_slideshow()
                time.sleep(state.wait_seconds)
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
                    state.loop_count = loop_count
                slide_entries = buffered_slides if buffered_slides is not None else state.slides
                for index, slide_entry in enumerate(slide_entries, 1):
                    if buffered_slides is not None:
                        title = slide_entry.title
                        body = slide_entry.body
                        clean_body = slide_entry.clean_body
                        segments = slide_entry.segments
                    else:
                        title, body = slide_entry
                        clean_body = IMAGE_PATTERN.sub("", body or "").strip()
                        segments = []
                    if state.stop_event.is_set():
                        final_status = "stopped"
                        break
                    with self._lock:
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
                            image_src = _resolve_image_source(match.group(1), state.script_dir)
                            image_commands.put(image_src)
                            if self.event_bus:
                                self.event_bus.publish(
                                    "presentation_image_shown",
                                    {"session": session.id, "slide": index, "image": image_src, "loop": loop_count},
                                )
                    status = "skipped"
                    if clean_body:
                        request_id = f"{session.id}-{loop_count}-{index}"
                        if buffered_slides is not None:
                            request_id = self.engine.play_segments(segments, request_id=request_id)
                        else:
                            request_id = self.engine.submit_text(clean_body, style_id=state.style_id, request_id=request_id)
                        status = self.engine.wait_for(request_id, timeout=state.timeout_sec)
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
                    if state.stop_event.is_set():
                        final_status = "stopped"
                        break
                    if state.stop_event.wait(max(state.wait_seconds, 0.0)):
                        final_status = "stopped"
                        break
                if final_status == "stopped" or not state.loop:
                    break
                if self.event_bus:
                    self.event_bus.publish(
                        "presentation_loop_waiting",
                        {"session": session.id, "loop": loop_count, "delay": state.loop_delay},
                    )
                if state.stop_event.wait(state.loop_delay):
                    final_status = "stopped"
                    break
        finally:
            with self._lock:
                if state.all_buffered:
                    state.buffer_status = "ready" if final_status == "completed" else final_status
                    state.current_index = 0 if final_status == "completed" else state.current_index
                    state.current_title = None if final_status == "completed" else state.current_title
                else:
                    self._states.pop(session.id, None)
            if session.metadata is not None and state.all_buffered:
                session.metadata["buffer_status"] = state.buffer_status
            if self.sessions:
                self.sessions.update_status(session.id, "ready" if state.all_buffered and final_status == "completed" else final_status)
            if self.event_bus:
                self.event_bus.publish(
                    "presentation_finished",
                    {
                        "session": session.id,
                        "status": final_status,
                        "slides": len(state.slides),
                    },
                )
            if session.metadata is not None:
                session.metadata["status"] = "ready" if state.all_buffered and final_status == "completed" else final_status
            if image_commands is not None:
                image_commands.put(None)
            if image_process is not None:
                image_process.join(timeout=3)
                if image_process.is_alive():
                    image_process.terminate()


__all__ = ["PresentMode"]
