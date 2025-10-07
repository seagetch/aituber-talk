"""Base mode interface for controller plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from controller.session import Session


class Mode(ABC):
    name: str = "mode"
    description: str = ""

    def __init__(self, **dependencies: Any) -> None:
        self.dependencies = dependencies

    @abstractmethod
    def start(self, session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Begin a session with the provided payload."""

    def pause(self, session: Session) -> Optional[Dict[str, Any]]:  # pragma: no cover - default no-op
        return None

    def resume(self, session: Session) -> Optional[Dict[str, Any]]:  # pragma: no cover - default no-op
        return None

    def stop(self, session: Session) -> Optional[Dict[str, Any]]:  # pragma: no cover - default no-op
        return None

    def status(self, session: Session) -> Dict[str, Any]:
        return session.as_dict()


__all__ = ["Mode"]
