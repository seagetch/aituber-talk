"""Controller application bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI

from controller import events
from controller.api import attach_routes
from controller.registry import ModeRegistry
from controller.session import SessionManager
from core.talk_engine import TalkEngine, TalkEngineConfig
from plugins.base import Mode
from plugins.present import PresentMode
from plugins.talk import TalkMode


class ControllerApp:
    """High-level orchestrator exposing the controller FastAPI app."""

    def __init__(
        self,
        *,
        engine: Optional[TalkEngine] = None,
        config: Optional[TalkEngineConfig] = None,
        registry: Optional[ModeRegistry] = None,
        sessions: Optional[SessionManager] = None,
        event_bus: Optional[events.EventBus] = None,
        upload_dir: Optional[Path] = None,
    ) -> None:
        self.engine = engine or TalkEngine(config)
        self.registry = registry or ModeRegistry()
        self.sessions = sessions or SessionManager()
        self.event_bus = event_bus or events.EventBus()
        self.upload_dir = Path(upload_dir or Path("uploads")).resolve()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.fastapi = FastAPI(title="AITuber Controller", version="0.1.0")
        self._mode_cache: Dict[str, Mode] = {}
        self._register_builtin_modes()
        attach_routes(self.fastapi, self)

    # ------------------------------------------------------------------
    def _register_builtin_modes(self) -> None:
        self.registry.register(
            "talk",
            lambda **deps: TalkMode(
                engine=self.engine,
                event_bus=self.event_bus,
                sessions=self.sessions,
            ),
            description="Send text to the shared speech engine",
            metadata={
                "style_id": self.engine.config.default_style_id,
                "form": [
                    {
                        "type": "textarea",
                        "name": "text",
                        "label": "Text",
                        "lines": 4,
                        "required": True,
                    },
                    {
                        "type": "checkbox",
                        "name": "sync",
                        "label": "Wait for completion",
                        "default": False,
                    },
                ],
            },
        )
        self.registry.register(
            "present",
            lambda **deps: PresentMode(
                engine=self.engine,
                event_bus=self.event_bus,
                sessions=self.sessions,
            ),
            description="Drive a PowerPoint-backed presentation",
            metadata={
                "requires": ["script_path"],
                "optional": ["ppt_path", "style_id"],
                "form": [
                    {
                        "type": "file",
                        "name": "script_path",
                        "label": "Script file",
                        "accept": [".md", ".txt"],
                        "required": True,
                    },
                    {
                        "type": "file",
                        "name": "ppt_path",
                        "label": "PowerPoint",
                        "accept": [".ppt", ".pptx"],
                        "required": False,
                    },
                    {
                        "type": "slider",
                        "name": "wait",
                        "label": "Slide delay",
                        "minimum": 0.0,
                        "maximum": 5.0,
                        "step": 0.1,
                        "default": 1.0,
                    },
                    {
                        "type": "slider",
                        "name": "timeout",
                        "label": "Slide timeout",
                        "minimum": 30.0,
                        "maximum": 600.0,
                        "step": 10.0,
                        "default": 300.0,
                    },
                ],
            },
        )

    def get_mode(self, name: str) -> Mode:
        if name not in self._mode_cache:
            self._mode_cache[name] = self.registry.create(
                name,
                engine=self.engine,
                event_bus=self.event_bus,
                sessions=self.sessions,
            )
        return self._mode_cache[name]

    def start_engine(self) -> None:
        self.engine.start()


__all__ = ["ControllerApp"]
