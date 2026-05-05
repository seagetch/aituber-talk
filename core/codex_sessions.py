"""Read-only Codex session discovery helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CodexSessionInfo:
    id: str
    thread_name: str
    updated_at: str
    cwd: Optional[str]
    rollout_path: Path
    mtime: float

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["rollout_path"] = str(self.rollout_path)
        return payload


def _codex_home(codex_home: Optional[Path] = None) -> Path:
    return Path(codex_home or Path.home() / ".codex").expanduser()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def read_session_index(codex_home: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    index_path = _codex_home(codex_home) / "session_index.jsonl"
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(index_path):
        session_id = row.get("id")
        if isinstance(session_id, str) and session_id:
            indexed[session_id] = row
    return indexed


def read_rollout_meta(path: Path, *, max_lines: int = 80) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if idx >= max_lines:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                payload = data.get("payload")
                if data.get("type") == "session_meta" and isinstance(payload, dict):
                    meta.update(payload)
                    break
                if data.get("source_type") == "session_meta" and isinstance(payload, dict):
                    meta.update(payload)
                    break
                if "session_meta" in data and isinstance(data["session_meta"], dict):
                    nested = data["session_meta"].get("payload", data["session_meta"])
                    if isinstance(nested, dict):
                        meta.update(nested)
                        break
    except OSError:
        return {}
    return meta


def list_codex_sessions(codex_home: Optional[Path] = None) -> List[CodexSessionInfo]:
    root = _codex_home(codex_home)
    indexed = read_session_index(root)
    sessions_dir = root / "sessions"
    found: Dict[str, CodexSessionInfo] = {}
    for rollout_path in sessions_dir.glob("**/rollout-*.jsonl"):
        if not rollout_path.is_file():
            continue
        meta = read_rollout_meta(rollout_path)
        session_id = meta.get("id")
        if not isinstance(session_id, str) or not session_id:
            session_id = rollout_path.stem.removeprefix("rollout-")
        index_row = indexed.get(session_id, {})
        stat = rollout_path.stat()
        thread_name = index_row.get("thread_name") or meta.get("thread_name") or session_id
        updated_at = index_row.get("updated_at") or meta.get("updated_at") or ""
        cwd = meta.get("cwd")
        info = CodexSessionInfo(
            id=session_id,
            thread_name=str(thread_name),
            updated_at=str(updated_at),
            cwd=str(cwd) if cwd is not None else None,
            rollout_path=rollout_path.resolve(),
            mtime=stat.st_mtime,
        )
        previous = found.get(session_id)
        if previous is None or info.mtime >= previous.mtime:
            found[session_id] = info
    return sorted(found.values(), key=lambda item: (item.updated_at, item.mtime), reverse=True)


__all__ = ["CodexSessionInfo", "list_codex_sessions", "read_rollout_meta", "read_session_index"]
