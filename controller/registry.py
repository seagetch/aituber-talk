"""Plugin registry for controller modes."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Optional, Protocol


class ModeFactory(Protocol):
    def __call__(self, **kwargs):  # pragma: no cover - typing helper
        ...


class ModeRegistry:
    """Registers and resolves available controller modes."""

    def __init__(self) -> None:
        self._factories: Dict[str, ModeFactory] = {}
        self._descriptions: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        factory: ModeFactory,
        *,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._factories[name] = factory
        if description:
            self._descriptions[name] = description
        if metadata is not None:
            self._metadata[name] = dict(metadata)

    def load_entry_points(self) -> None:
        try:
            import importlib.metadata as metadata
        except ImportError:  # pragma: no cover - Python <3.8 fallback
            import importlib_metadata as metadata  # type: ignore

        for entry_point in metadata.entry_points().select(group="aituber_talk.modes"):
            module_name, _, attr = entry_point.value.partition(":")
            module = import_module(module_name)
            factory = getattr(module, attr)
            self.register(entry_point.name, factory)

    def create(self, name: str, **kwargs):
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(f"Mode not registered: {name}")
        return factory(**kwargs)

    def list_modes(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "description": self._descriptions.get(name, ""),
                "name": name,
                "metadata": self._metadata.get(name, {}),
            }
            for name in sorted(self._factories)
        }


__all__ = ["ModeRegistry", "ModeFactory"]
