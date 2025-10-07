"""Event bus utilities for the controller service."""

from __future__ import annotations

import queue
import threading
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, DefaultDict, Dict, List, Tuple

EventCallback = Callable[[str, Dict[str, Any]], None]


class EventBus:
    """Publish/subscribe hub used by controller and plugins."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, List[EventCallback]] = defaultdict(list)
        self._lock = threading.Lock()

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(event_type, []))
            wildcard = list(self._subscribers.get("*", []))
        merged = subscribers + wildcard
        for callback in merged:
            try:
                callback(event_type, deepcopy(payload))
            except Exception:  # pragma: no cover - defensive hook isolation
                continue

    def subscribe(self, event_type: str, callback: EventCallback) -> Callable[[], None]:
        with self._lock:
            self._subscribers[event_type].append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                if not self._subscribers[event_type]:
                    self._subscribers.pop(event_type, None)

        return unsubscribe

    def subscribe_queue(self, event_type: str = "*", maxsize: int = 100) -> Tuple["queue.Queue[Tuple[str, Dict[str, Any]]]", Callable[[], None]]:
        q: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue(maxsize=maxsize)

        def _push(evt_type: str, payload: Dict[str, Any]) -> None:
            try:
                q.put_nowait((evt_type, payload))
            except queue.Full:
                # drop oldest item to make room, then enqueue
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait((evt_type, payload))
                except queue.Full:  # pragma: no cover - extremely unlikely
                    pass

        unsubscribe = self.subscribe(event_type, _push)
        return q, unsubscribe

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()


__all__ = ["EventBus", "EventCallback"]
