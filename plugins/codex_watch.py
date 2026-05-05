"""Codex session watcher mode that speaks agent updates through TalkEngine."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Set

from controller.events import EventBus
from controller.session import Session, SessionManager
from core.codex_sessions import CodexSessionInfo, list_codex_sessions, read_rollout_meta
from core.codex_watcher import CodexJsonlWatcher, CodexMessageEvent, CodexWatcherActivity
from plugins.base import Mode

if TYPE_CHECKING:
    from core.talk_engine import TalkEngine


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_PROFILE = "\n".join(
    [
        "性格: 落ち着いていて実務的",
        "口調: 短く自然な日本語",
        "一人称: 私",
        "ユーザーへの呼び方: あなた",
        "禁止事項: 内部ログやツール出力をそのまま読まない",
    ]
)

Rewriter = Callable[[List[CodexMessageEvent], List[str], List[str], str, str], Dict[str, Any]]


@dataclass
class _CodexWatchState:
    stop_event: threading.Event
    message_queue: "queue.Queue[Optional[CodexMessageEvent]]"
    watcher: CodexJsonlWatcher
    worker_thread: threading.Thread
    codex_session_id: Optional[str]
    thread_name: str
    cwd: Optional[str]
    rollout_path: Path
    character_profile: str
    style_id: Optional[int]
    model: str
    event_types: Set[str]
    batch_window_sec: float = 0.75
    max_batch_events: int = 8
    catch_up_drain_limit: int = 32
    max_utterance_chars: Optional[int] = None
    speech_chars_per_second: float = 8.0
    speech_backlog_until: float = 0.0
    last_delivery_policy: str = "full"
    last_estimated_wait_sec: float = 0.0
    watch_status: str = "running"
    last_codex_message: Optional[str] = None
    last_utterance: Optional[str] = None
    last_error: Optional[str] = None
    last_watcher_activity: Optional[str] = None
    bytes_read: int = 0
    lines_read: int = 0
    ignored_lines: int = 0
    messages_seen: int = 0
    utterances_generated: int = 0
    recent_codex_messages: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    recent_character_utterances: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    last_seen_message_hashes: Deque[str] = field(default_factory=lambda: deque(maxlen=80))


def _as_int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _message_hash(event: CodexMessageEvent) -> str:
    key = f"{event.role}\0{event.text.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _resolve_session_by_id(session_id: str) -> Optional[CodexSessionInfo]:
    for info in list_codex_sessions():
        if info.id == session_id:
            return info
    return None


class CodexWatchMode(Mode):
    name = "codex_watch"
    description = "Watch a Codex rollout JSONL and speak assistant updates."

    def __init__(
        self,
        engine: "TalkEngine",
        event_bus: Optional[EventBus] = None,
        sessions: Optional[SessionManager] = None,
        rewriter: Optional[Rewriter] = None,
    ) -> None:
        super().__init__(engine=engine, event_bus=event_bus, sessions=sessions, rewriter=rewriter)
        self.engine = engine
        self.event_bus = event_bus
        self.sessions = sessions
        self.rewriter = rewriter
        self._states: Dict[str, _CodexWatchState] = {}
        self._lock = threading.Lock()

    def start(self, session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
        info = self._resolve_target(payload)
        if info is None:
            raise ValueError("'session_id' or 'rollout_path' must identify a Codex session")

        style_id = _as_int_or_none(payload.get("style_id"))
        if style_id is None:
            style_id = self.engine.config.default_style_id
        model = str(payload.get("model") or DEFAULT_MODEL)
        event_types = set(payload.get("event_types") or ["assistant_message"])
        character_profile = str(payload.get("character_profile") or DEFAULT_PROFILE)
        batch_window_sec = max(0.0, float(payload.get("batch_window_sec", 0.75)))
        max_batch_events = max(1, int(payload.get("max_batch_events", 8)))
        catch_up_drain_limit = max(max_batch_events, int(payload.get("catch_up_drain_limit", 32)))
        max_utterance_chars = _as_int_or_none(payload.get("max_utterance_chars"))
        if max_utterance_chars is not None:
            max_utterance_chars = max(80, max_utterance_chars)
        speech_chars_per_second = max(2.0, float(payload.get("speech_chars_per_second", 8.0)))

        watcher = CodexJsonlWatcher(info.rollout_path, start_at_end=True)
        message_queue: "queue.Queue[Optional[CodexMessageEvent]]" = queue.Queue()
        stop_event = threading.Event()
        state = _CodexWatchState(
            stop_event=stop_event,
            message_queue=message_queue,
            watcher=watcher,
            worker_thread=threading.Thread(
                target=self._rewrite_worker,
                args=(session.id,),
                name=f"CodexWatchRewrite-{session.id}",
                daemon=True,
            ),
            codex_session_id=info.id,
            thread_name=info.thread_name,
            cwd=info.cwd,
            rollout_path=info.rollout_path,
            character_profile=character_profile,
            style_id=style_id,
            model=model,
            event_types=event_types,
            batch_window_sec=batch_window_sec,
            max_batch_events=max_batch_events,
            catch_up_drain_limit=catch_up_drain_limit,
            max_utterance_chars=max_utterance_chars,
            speech_chars_per_second=speech_chars_per_second,
        )
        with self._lock:
            self._states[session.id] = state
        session.metadata.update(
            {
                "mode": self.name,
                "codex_session_id": info.id,
                "thread_name": info.thread_name,
                "cwd": info.cwd or "",
                "rollout_path": str(info.rollout_path),
                "model": model,
                "style_id": str(style_id),
                "batch_window_sec": str(batch_window_sec),
                "max_batch_events": str(max_batch_events),
                "catch_up_drain_limit": str(catch_up_drain_limit),
                "max_utterance_chars": str(max_utterance_chars or ""),
                "speech_chars_per_second": str(speech_chars_per_second),
            }
        )
        state.worker_thread.start()
        watcher.start(
            lambda event: self._enqueue_message(session.id, event),
            lambda exc: self._record_error(session.id, exc),
            lambda activity: self._record_activity(session.id, activity),
        )
        if self.sessions:
            self.sessions.update_status(session.id, "running")
        self._publish(
            "codex_watch.started",
            {
                "session": session.id,
                "codex_session_id": info.id,
                "thread_name": info.thread_name,
                "cwd": info.cwd,
                "rollout_path": str(info.rollout_path),
                "model": model,
            },
        )
        self._publish(
            "codex_watch.session_selected",
            {
                "session": session.id,
                "codex_session_id": info.id,
                "thread_name": info.thread_name,
                "cwd": info.cwd,
                "rollout_path": str(info.rollout_path),
            },
        )
        return {
            "status": "running",
            "session": session.id,
            "codex_session_id": info.id,
            "rollout_path": str(info.rollout_path),
        }

    def _resolve_target(self, payload: Dict[str, Any]) -> Optional[CodexSessionInfo]:
        session_id = payload.get("session_id")
        if session_id:
            found = _resolve_session_by_id(str(session_id))
            if found:
                return found
        rollout_path = payload.get("rollout_path")
        if not rollout_path:
            return None
        path = Path(str(rollout_path)).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Codex rollout not found: {path}")
        meta = read_rollout_meta(path)
        session_id = str(meta.get("id") or path.stem.removeprefix("rollout-"))
        indexed = _resolve_session_by_id(session_id)
        if indexed:
            return indexed
        stat = path.stat()
        return CodexSessionInfo(
            id=session_id,
            thread_name=session_id,
            updated_at=str(meta.get("updated_at") or ""),
            cwd=str(meta.get("cwd")) if meta.get("cwd") else None,
            rollout_path=path,
            mtime=stat.st_mtime,
        )

    def _enqueue_message(self, session_id: str, event: CodexMessageEvent) -> None:
        state = self._state(session_id)
        if state is None or state.stop_event.is_set():
            return
        state.messages_seen += 1
        state.last_codex_message = event.text
        state.recent_codex_messages.append(event.text)
        self._publish(
            "codex_watch.codex_message",
            {
                "session": session_id,
                "codex_session_id": event.session_id,
                "source_type": event.source_type,
                "role": event.role,
                "text": event.text,
                "timestamp": event.timestamp,
                "cwd": event.cwd,
            },
        )
        if self._should_rewrite(state, event):
            state.message_queue.put(event)

    def _record_activity(self, session_id: str, activity: CodexWatcherActivity) -> None:
        state = self._state(session_id)
        if state is None or state.stop_event.is_set():
            return
        state.bytes_read += activity.bytes_read
        state.lines_read += activity.lines_read
        state.ignored_lines += activity.ignored_lines
        state.last_watcher_activity = (
            f"read {activity.bytes_read} bytes, {activity.lines_read} lines, "
            f"{activity.events_emitted} message events, {activity.ignored_lines} ignored"
        )
        self._publish(
            "codex_watch.file_changed",
            {
                "session": session_id,
                "codex_session_id": state.codex_session_id,
                "rollout_path": str(activity.session_file),
                "bytes_read": activity.bytes_read,
                "lines_read": activity.lines_read,
                "events_emitted": activity.events_emitted,
                "ignored_lines": activity.ignored_lines,
                "offset": activity.offset,
                "partial_buffer_bytes": activity.partial_buffer_bytes,
            },
        )

    def _should_rewrite(self, state: _CodexWatchState, event: CodexMessageEvent) -> bool:
        if not event.text.strip():
            return False
        digest = _message_hash(event)
        if digest in state.last_seen_message_hashes:
            return False
        state.last_seen_message_hashes.append(digest)
        if event.source_type == "event_msg.agent_message":
            return "assistant_message" in state.event_types or "event_msg.agent_message" in state.event_types
        if event.source_type == "response_item.message" and event.role == "assistant":
            return "assistant_message" in state.event_types or "response_item.message" in state.event_types
        return False

    def _rewrite_worker(self, session_id: str) -> None:
        while True:
            state = self._state(session_id)
            if state is None:
                return
            try:
                event = state.message_queue.get(timeout=0.2)
            except queue.Empty:
                if state.stop_event.is_set():
                    return
                continue
            if event is None:
                return
            events = [event]
            while len(events) < state.max_batch_events:
                try:
                    next_event = state.message_queue.get(timeout=state.batch_window_sec)
                except queue.Empty:
                    break
                if next_event is None:
                    state.message_queue.put(None)
                    break
                events.append(next_event)
            try:
                delivery_context = self._delivery_context(state)
                if delivery_context["policy"] == "catch_up":
                    events, merge_info = self._drain_catch_up_events(events, state)
                    delivery_context.update(merge_info)
                self._publish(
                    "codex_watch.rewrite_started",
                    {
                        "session": session_id,
                        "codex_session_id": state.codex_session_id,
                        "texts": [item.text for item in events],
                        "event_count": len(events),
                        "delivery_policy": delivery_context["policy"],
                        "estimated_wait_sec": delivery_context["estimated_wait_sec"],
                        "merged_event_count": delivery_context.get("merged_event_count", len(events)),
                        "drained_event_count": delivery_context.get("drained_event_count", 0),
                    },
                )
                result = self._rewrite(events, state, delivery_context)
                should_speak = bool(result.get("should_speak"))
                utterance = str(result.get("utterance") or "").strip()
                self._publish(
                    "codex_watch.rewrite_completed",
                    {
                        "session": session_id,
                        "codex_session_id": state.codex_session_id,
                        "should_speak": should_speak,
                        "utterance": utterance,
                        "event_count": len(events),
                        "delivery_policy": delivery_context["policy"],
                        "estimated_wait_sec": delivery_context["estimated_wait_sec"],
                        "merged_event_count": delivery_context.get("merged_event_count", len(events)),
                        "drained_event_count": delivery_context.get("drained_event_count", 0),
                    },
                )
                if not should_speak or not utterance:
                    continue
                if state.max_utterance_chars is not None:
                    utterance = utterance[: state.max_utterance_chars]
                estimated_duration = self._reserve_speech_slot(state, utterance)
                request_id = self.engine.submit_text(utterance, style_id=state.style_id, request_id=f"{session_id}-{state.utterances_generated + 1}")
                state.utterances_generated += 1
                state.last_utterance = utterance
                state.recent_character_utterances.append(utterance)
                self._publish(
                    "codex_watch.speech_queued",
                    {
                        "session": session_id,
                        "codex_session_id": state.codex_session_id,
                        "utterance": utterance,
                        "request": request_id,
                        "style_id": state.style_id,
                        "estimated_duration_sec": estimated_duration,
                        "speech_backlog_until": state.speech_backlog_until,
                    },
                )
            except Exception as exc:
                self._record_error(session_id, exc)

    def _drain_catch_up_events(
        self,
        events: List[CodexMessageEvent],
        state: _CodexWatchState,
    ) -> tuple[List[CodexMessageEvent], Dict[str, Any]]:
        merged = list(events)
        drained = 0
        while len(merged) < state.catch_up_drain_limit:
            try:
                next_event = state.message_queue.get_nowait()
            except queue.Empty:
                break
            if next_event is None:
                state.message_queue.put(None)
                break
            merged.append(next_event)
            drained += 1
        return merged, {
            "merged_event_count": len(merged),
            "drained_event_count": drained,
        }

    def _delivery_context(self, state: _CodexWatchState) -> Dict[str, Any]:
        now = time.monotonic()
        estimated_wait_sec = max(0.0, state.speech_backlog_until - now)
        try:
            engine_status = self.engine.status()
        except Exception:
            engine_status = {}
        queued = engine_status.get("queued")
        engine_available_in = engine_status.get("speech_available_in_sec")
        if isinstance(engine_available_in, (int, float)):
            estimated_wait_sec = max(estimated_wait_sec, float(engine_available_in))
        else:
            if isinstance(queued, int) and queued > 0:
                estimated_wait_sec = max(estimated_wait_sec, queued * 2.0)
            if engine_status.get("session"):
                estimated_wait_sec = max(estimated_wait_sec, 2.0)
        pending_batches = state.message_queue.qsize()
        if estimated_wait_sec >= 18 or pending_batches >= 6:
            policy = "catch_up"
            target_retention = 0.35
            instruction = (
                "Codexの進行に発話がかなり遅れている。入力された全イベントを読んだうえで、出力のセリフだけを強く圧縮する。"
                "全体として何が起きたか、今ユーザーが知るべき結論・重要な変更・注意点・次の行動を優先する。"
                "耳で聞いても追えない細部、反復、途中の迷いはセリフに出さず、要点を短く統合する。"
            )
        elif estimated_wait_sec >= 8 or pending_batches >= 3:
            policy = "balanced"
            target_retention = 0.55
            instruction = (
                "発話が少し遅れている。入力された全イベントを読んだうえで、出力のセリフを適度に圧縮する。"
                "全体の要点、判断理由、ユーザーが次に理解すべきことを残し、耳で追いにくい細部は省く。"
            )
        else:
            policy = "full"
            target_retention = 0.70
            instruction = (
                "発話は追いついている。入力された全イベントを読んだうえで、全体の要点に沿って十分に追随する。"
                "短文を過剰に膨らませず、長文は省略しすぎず、聞いて意味がつながる量で話す。"
            )
        state.last_delivery_policy = policy
        state.last_estimated_wait_sec = estimated_wait_sec
        return {
            "policy": policy,
            "estimated_wait_sec": round(estimated_wait_sec, 2),
            "pending_batches": pending_batches,
            "engine_status": engine_status,
            "target_retention": target_retention,
            "instruction": instruction,
        }

    def _estimate_speech_duration(self, state: _CodexWatchState, utterance: str) -> float:
        char_count = max(len(utterance.strip()), 1)
        return max(1.0, char_count / state.speech_chars_per_second)

    def _reserve_speech_slot(self, state: _CodexWatchState, utterance: str) -> float:
        duration = self._estimate_speech_duration(state, utterance)
        now = time.monotonic()
        state.speech_backlog_until = max(now, state.speech_backlog_until) + duration
        return round(duration, 2)

    def _rewrite(
        self,
        events: List[CodexMessageEvent],
        state: _CodexWatchState,
        delivery_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.rewriter:
            return self.rewriter(
                events,
                list(state.recent_codex_messages),
                list(state.recent_character_utterances),
                state.character_profile,
                state.model,
            )
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed") from exc

        client = OpenAI()
        user_payload = {
            "character_profile": state.character_profile,
            "recent_codex_messages": list(state.recent_codex_messages)[-8:],
            "recent_character_utterances": list(state.recent_character_utterances)[-8:],
            "delivery_context": delivery_context or self._delivery_context(state),
            "new_codex_updates": [
                {
                    "source_type": event.source_type,
                    "role": event.role,
                    "text": event.text,
                    "timestamp": event.timestamp,
                }
                for event in events
            ],
        }
        completion = client.chat.completions.create(
            model=state.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたはVTuberキャラクターのセリフ変換器。"
                        "入力された完結済みCodex更新を、キャラクターが今読み上げる自然な日本語セリフに変換する。"
                        "説明文ではなく、発話としてそのまま使えるJSONだけを出す。"
                        '形式は {"should_speak": true, "utterance": "セリフ"}。'
                        "調査方針、作業方針、実装方針、途中経過、発見事項、注意点、次の行動、完了報告は必ずshould_speak=true。"
                        "ユーザーに見せているassistantの進捗説明は、最終回答でなくても喋る。"
                        "should_speak=falseにしてよいのは、空、重複、純粋なtool output、内部ログ、メタデータだけの更新に限る。"
                        "入力は間引かず、new_codex_updates全体を材料として読む。"
                        "セリフでは、耳で聞いても識別しにくい詳細を列挙するより、全体として何が重要かを判断して伝える。"
                        "ファイル名、数値、原因、制約、次の行動などは、ユーザーが判断するために必要な場合だけ残す。"
                        "元の文章が短ければ短く話す。元の文章が長ければ、全体の要点に追随できるだけの長さで話す。"
                        "ただしdelivery_contextのinstructionを優先し、遅れている場合は出力セリフの情報量を落として追いつく。"
                        "複数イベントがある場合は、入力を捨てずに全体を統合し、出力側で反復や重要度の低い途中経過を省く。"
                        "複数文や少し長い発話になってよい。必要なら接続詞を補い、聞き取りやすい話し言葉にする。"
                        "ただし内部ログやツール出力の全文コピーではなく、キャラクターの口調に自然に変換する。"
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI response was not a JSON object")
        return parsed

    def _record_error(self, session_id: str, exc: Exception) -> None:
        state = self._state(session_id)
        error = str(exc)
        if state:
            state.last_error = error
            state.watch_status = "error" if state.stop_event.is_set() else state.watch_status
        if self.sessions and state and state.stop_event.is_set():
            self.sessions.update_status(session_id, "error")
        self._publish("codex_watch.error", {"session": session_id, "error": error})

    def stop(self, session: Session) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._states.pop(session.id, None)
        if state is None:
            return None
        state.watch_status = "stopped"
        state.stop_event.set()
        state.watcher.stop()
        state.message_queue.put(None)
        state.worker_thread.join(timeout=2.0)
        if self.sessions:
            self.sessions.update_status(session.id, "stopped")
        self._publish("codex_watch.stopped", {"session": session.id, "codex_session_id": state.codex_session_id})
        return {"status": "stopped", "session": session.id}

    def status(self, session: Session) -> Dict[str, Any]:
        state = self._state(session.id)
        if state is None:
            payload = session.as_dict()
            payload.update({"watch_status": "stopped"})
            return payload
        return {
            "session": session.id,
            "watch_status": state.watch_status,
            "codex_session_id": state.codex_session_id,
            "thread_name": state.thread_name,
            "cwd": state.cwd,
            "rollout_path": str(state.rollout_path),
            "last_codex_message": state.last_codex_message,
            "last_utterance": state.last_utterance,
            "messages_seen": state.messages_seen,
            "utterances_generated": state.utterances_generated,
            "recent_utterance_history": list(state.recent_character_utterances),
            "recent_codex_messages": list(state.recent_codex_messages),
            "last_error": state.last_error,
            "last_watcher_activity": state.last_watcher_activity,
            "bytes_read": state.bytes_read,
            "lines_read": state.lines_read,
            "ignored_lines": state.ignored_lines,
            "delivery_policy": state.last_delivery_policy,
            "estimated_wait_sec": round(max(0.0, state.speech_backlog_until - time.monotonic()), 2),
            "last_estimated_wait_sec": round(state.last_estimated_wait_sec, 2),
            "speech_backlog_until": state.speech_backlog_until,
            "pending_speech_batches": state.message_queue.qsize(),
            "model": state.model,
            "style_id": state.style_id,
            "batch_window_sec": state.batch_window_sec,
            "max_batch_events": state.max_batch_events,
            "max_utterance_chars": state.max_utterance_chars,
        }

    def _state(self, session_id: str) -> Optional[_CodexWatchState]:
        with self._lock:
            return self._states.get(session_id)

    def _publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_bus:
            self.event_bus.publish(event_type, payload)


CodexAvatarMode = CodexWatchMode


__all__ = ["CodexAvatarMode", "CodexWatchMode"]
