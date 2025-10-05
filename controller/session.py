"""Session tracking utilities for controller-managed modes."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Session:
    id: str
    mode: str
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


class SessionManager:
    """In-memory session registry."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def create(self, mode: str, *, session_id: Optional[str] = None, metadata: Optional[Dict[str, str]] = None) -> Session:
        session_id = session_id or uuid.uuid4().hex
        session = Session(id=session_id, mode=mode, metadata=metadata or {})
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def update_status(self, session_id: str, status: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.status = status
        session.updated_at = time.time()
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list(self) -> List[Session]:
        return list(self._sessions.values())


__all__ = ["Session", "SessionManager"]
