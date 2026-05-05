"""Incremental JSONL watcher for Codex rollout files."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CodexMessageEvent:
    session_id: Optional[str]
    session_file: Path
    timestamp: Optional[str]
    source_type: str
    role: Optional[str]
    text: str
    cwd: Optional[str]
    raw: Dict[str, Any]


@dataclass(frozen=True)
class CodexWatcherActivity:
    session_file: Path
    bytes_read: int
    lines_read: int
    events_emitted: int
    ignored_lines: int
    offset: int
    partial_buffer_bytes: int


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return _extract_text(text)
    return ""


def normalize_codex_event(raw: Dict[str, Any], path: Path, session_meta: Dict[str, Any]) -> Optional[CodexMessageEvent]:
    source_type = str(raw.get("type") or raw.get("source_type") or "")
    payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else {}
    item = raw.get("item") if isinstance(raw.get("item"), dict) else {}
    timestamp = raw.get("timestamp") or raw.get("created_at") or payload.get("timestamp")
    session_id = session_meta.get("id") or raw.get("session_id") or payload.get("session_id")
    cwd = session_meta.get("cwd") or payload.get("cwd")

    if source_type == "session_meta":
        session_meta.update(payload)
        return None

    if source_type == "event_msg":
        msg_type = payload.get("type") or raw.get("event_type")
        if msg_type == "agent_message":
            text = _extract_text(payload.get("message") or payload.get("text") or payload.get("content"))
            role = "assistant"
        elif msg_type == "user_message":
            text = _extract_text(payload.get("message") or payload.get("text") or payload.get("content"))
            role = "user"
        else:
            return None
        if not text:
            return None
        return CodexMessageEvent(
            session_id=str(session_id) if session_id else None,
            session_file=path,
            timestamp=str(timestamp) if timestamp else None,
            source_type=f"event_msg.{msg_type}",
            role=role,
            text=text,
            cwd=str(cwd) if cwd else None,
            raw=raw,
        )

    if source_type == "response_item":
        message = item or payload
        if message.get("type") != "message" and "role" not in message:
            return None
        role_value = message.get("role")
        text = _extract_text(message.get("content") or message.get("text"))
        if not text:
            return None
        return CodexMessageEvent(
            session_id=str(session_id) if session_id else None,
            session_file=path,
            timestamp=str(timestamp) if timestamp else None,
            source_type="response_item.message",
            role=str(role_value) if role_value else None,
            text=text,
            cwd=str(cwd) if cwd else None,
            raw=raw,
        )
    return None


class CodexJsonlWatcher:
    """Watches one Codex rollout JSONL file and emits normalized messages."""

    def __init__(self, path: Path, *, start_at_end: bool = True, poll_interval: float = 0.5) -> None:
        self.path = Path(path).expanduser().resolve()
        self.poll_interval = poll_interval
        self.offset = self.path.stat().st_size if start_at_end and self.path.exists() else 0
        self.partial_buffer = ""
        self.session_meta: Dict[str, Any] = self._read_initial_meta()
        self.last_activity = CodexWatcherActivity(self.path, 0, 0, 0, 0, self.offset, 0)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_initial_meta(self) -> Dict[str, Any]:
        try:
            from core.codex_sessions import read_rollout_meta

            return dict(read_rollout_meta(self.path))
        except Exception:
            return {}

    def read_available(self) -> List[CodexMessageEvent]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.partial_buffer = ""
        with self.path.open("r", encoding="utf-8") as fh:
            fh.seek(self.offset)
            chunk = fh.read()
            self.offset = fh.tell()
        if not chunk:
            self.last_activity = CodexWatcherActivity(self.path, 0, 0, 0, 0, self.offset, len(self.partial_buffer.encode("utf-8")))
            return []
        text = self.partial_buffer + chunk
        if text.endswith("\n"):
            lines = text.splitlines()
            self.partial_buffer = ""
        else:
            lines = text.splitlines()
            self.partial_buffer = lines.pop() if lines else text
        events, ignored = self._parse_lines(lines)
        self.last_activity = CodexWatcherActivity(
            self.path,
            len(chunk.encode("utf-8")),
            len(lines),
            len(events),
            ignored,
            self.offset,
            len(self.partial_buffer.encode("utf-8")),
        )
        return events

    def _parse_lines(self, lines: Iterable[str]) -> tuple[List[CodexMessageEvent], int]:
        events: List[CodexMessageEvent] = []
        ignored = 0
        for line in lines:
            line = line.strip()
            if not line:
                ignored += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                ignored += 1
                continue
            if not isinstance(raw, dict):
                ignored += 1
                continue
            event = normalize_codex_event(raw, self.path, self.session_meta)
            if event is not None:
                events.append(event)
            else:
                ignored += 1
        return events, ignored

    def start(
        self,
        callback: Callable[[CodexMessageEvent], None],
        error_callback: Optional[Callable[[Exception], None]] = None,
        activity_callback: Optional[Callable[[CodexWatcherActivity], None]] = None,
    ) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(callback, error_callback, activity_callback),
            name=f"CodexJsonlWatcher-{self.path.name}",
            daemon=True,
        )
        self._thread.start()

    def _read_and_emit(
        self,
        callback: Callable[[CodexMessageEvent], None],
        activity_callback: Optional[Callable[[CodexWatcherActivity], None]],
    ) -> None:
        events = self.read_available()
        if self.last_activity.bytes_read or self.last_activity.lines_read or self.last_activity.partial_buffer_bytes:
            if activity_callback:
                activity_callback(self.last_activity)
        for event in events:
            callback(event)

    def _run(
        self,
        callback: Callable[[CodexMessageEvent], None],
        error_callback: Optional[Callable[[Exception], None]],
        activity_callback: Optional[Callable[[CodexWatcherActivity], None]],
    ) -> None:
        # A Codex rollout is a single append-only JSONL file. Polling the file
        # size/offset is more reliable than depending on platform file events,
        # especially when another process keeps the file open and appends often.
        self._poll(callback, error_callback, activity_callback)

    def _watch_filter(self, change: object, path: str) -> bool:
        return Path(path).resolve() == self.path

    def _poll(
        self,
        callback: Callable[[CodexMessageEvent], None],
        error_callback: Optional[Callable[[Exception], None]],
        activity_callback: Optional[Callable[[CodexWatcherActivity], None]],
    ) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._read_and_emit(callback, activity_callback)
            except Exception as exc:
                if error_callback:
                    error_callback(exc)
                time.sleep(self.poll_interval)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)


__all__ = ["CodexJsonlWatcher", "CodexMessageEvent", "CodexWatcherActivity", "normalize_codex_event"]
