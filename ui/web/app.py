"""Gradio UI targeting the controller service with mode-specific controls and live updates."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
from collections import deque
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import gradio as gr
import requests
from requests import exceptions as requests_exceptions

DEFAULT_CONTROLLER_URL = os.getenv("AITUBER_CONTROLLER_URL", "http://127.0.0.1:8001")
DEFAULT_SPEAKER_URL = os.getenv("AIVIS_SPEAKER_URL", "http://127.0.0.1:10101")

STYLE = """
<style>
:root { color-scheme: dark; }
body { background: #0f172a; }
.session-grid { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.session-card { border: 1px solid #374151; background: #1f2937; border-radius: 12px; padding: 12px; transition: border-color 0.2s, box-shadow 0.2s; }
.session-card.selected { border-color: #38bdf8; box-shadow: 0 0 0 1px #38bdf8; }
.session-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-weight: 600; }
.session-id { font-family: "JetBrains Mono", "Consolas", monospace; color: #9ca3af; }
.session-status { padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; background: #374151; color: #f9fafb; }
.session-status.running { background: #1d4ed833; color: #bfdbfe; }
.session-status.completed { background: #14532d33; color: #bbf7d0; }
.session-status.stopped { background: #4c1d9533; color: #e0e7ff; }
.session-status.error { background: #7f1d1d33; color: #fecaca; }
.session-meta { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.85rem; color: #d1d5db; margin-bottom: 10px; }
.session-meta span strong { color: #f9fafb; }
.progress-container { background: #111827; border-radius: 6px; height: 12px; overflow: hidden; margin-bottom: 6px; }
.progress-bar { height: 100%; background: linear-gradient(90deg, #a855f7, #38bdf8); transition: width 0.3s ease; }
.progress-text { font-size: 0.75rem; color: #9ca3af; margin-bottom: 4px; }
.session-flags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.session-flag { font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; background: #374151; color: #e5e7eb; }
.session-empty { padding: 20px; border: 1px dashed #374151; border-radius: 10px; text-align: center; color: #9ca3af; margin-top: 12px; }
</style>
"""

def _fetch_json(method: str, url: str, *, timeout: float = 10, **kwargs) -> Dict[str, Any]:
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return {"raw": resp.text}
    except Exception as exc:  # pragma: no cover - robustness guard
        return {"error": str(exc)}

def fetch_speakers(speaker_url: str) -> List[Dict[str, Any]]:
    data = _fetch_json("GET", f"{speaker_url}/speakers", timeout=5)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "error" not in data:
        return data.get("speakers", [])
    return []

def list_sessions(controller_url: str) -> List[Dict[str, Any]]:
    data = _fetch_json("GET", f"{controller_url}/sessions", timeout=5)
    if isinstance(data, dict) and "error" not in data:
        return data.get("sessions", [])
    return []

def list_present_sessions(controller_url: str) -> List[Dict[str, Any]]:
    return [s for s in list_sessions(controller_url) if s.get("mode") == "present"]

def get_session_status(controller_url: str, session_id: str) -> Dict[str, Any]:
    return _fetch_json("GET", f"{controller_url}/sessions/{session_id}", timeout=5)

def send_command(controller_url: str, session_id: str, command: str) -> str:
    if not session_id:
        return "Session ID is required"
    payload = {"command": command}
    data = _fetch_json(
        "POST",
        f"{controller_url}/sessions/{session_id}/commands",
        json=payload,
        timeout=10,
    )
    if isinstance(data, dict) and "error" in data and len(data) == 1:
        return data["error"]
    return json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data)

def create_session(controller_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _fetch_json("POST", f"{controller_url}/sessions", json=payload, timeout=30)

def upload_asset(controller_url: str, file_path: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not file_path:
        return None, None
    try:
        filename = os.path.basename(file_path)
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(file_path, "rb") as fh:
            resp = requests.post(
                f"{controller_url}/files",
                files={"file": (filename, fh, mime)},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("path"), None
    except Exception as exc:  # pragma: no cover - upload robustness
        return None, f"Upload failed for {file_path}: {exc}"
    return None, "Unexpected upload response"

def list_uploaded_files(controller_url: str) -> List[Dict[str, Any]]:
    data = _fetch_json("GET", f"{controller_url}/files", timeout=5)
    if isinstance(data, dict) and "files" in data:
        return data.get("files", [])
    return []


def delete_uploaded_file(controller_url: str, filename: str) -> Optional[str]:
    response = _fetch_json("DELETE", f"{controller_url}/files/{filename}", timeout=5)
    if isinstance(response, dict) and "error" in response:
        return response["error"]
    return None


def _human_size(num_bytes: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < step:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= step
    return f"{value:.1f} PB"


PRESENT_EVENT_TYPES = {
    "presentation_started",
    "presentation_slide_started",
    "presentation_slide_completed",
    "presentation_finished",
    "presentation_error",
    "presentation_stopped",
    "speech_queued",
    "speech_completed",
    "speech_stopped",
    "speech_error",
}


def _iter_controller_events(base_url: str, stop_event: threading.Event) -> Iterator[Tuple[str, Dict[str, Any]]]:
    endpoint = f"{base_url.rstrip('/')}/events"
    headers = {"Accept": "text/event-stream"}
    backoff = 1.0
    while not stop_event.is_set():
        try:
            with requests.get(endpoint, stream=True, timeout=(5, 65), headers=headers) as resp:
                resp.raise_for_status()
                yield "__connected__", {}
                event_name: Optional[str] = None
                data_lines: List[str] = []
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        return
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        if event_name:
                            data_text = "\n".join(data_lines).strip()
                            if data_text:
                                try:
                                    payload = json.loads(data_text)
                                except json.JSONDecodeError:
                                    payload = {"raw": data_text}
                            else:
                                payload = {}
                            yield event_name, payload
                        event_name = None
                        data_lines.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
            backoff = 1.0
        except GeneratorExit:
            return
        except requests_exceptions.ReadTimeout:
            if stop_event.is_set():
                return
            yield "__timeout__", {}
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
        except requests.RequestException as exc:
            if stop_event.is_set():
                return
            yield "__error__", {"error": str(exc)}
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
        except Exception as exc:
            if stop_event.is_set():
                return
            yield "__error__", {"error": str(exc)}
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

def render_present_cards(details: Dict[str, Dict[str, Any]], selected_id: Optional[str]) -> str:
    if not details:
        return "<div class='session-empty'>No active presentation sessions.</div>"
    items = sorted(
        details.items(),
        key=lambda kv: kv[1].get("session", {}).get("created_at", 0) or 0,
        reverse=True,
    )
    parts: List[str] = ["<div class='session-grid'>"]
    for sid, detail in items:
        session_info = detail.get("session", {}) or {}
        status_info = detail.get("status", {}) or {}
        metadata = session_info.get("metadata", {}) or {}
        slides_total = status_info.get("slides_total") or metadata.get("slides_total") or 0
        current_slide = status_info.get("current_slide") or 0
        progress_pct = 0.0
        if slides_total:
            progress_pct = max(0.0, min(100.0, (current_slide / slides_total) * 100.0))
        status_label = (session_info.get("status") or "running").lower()
        script_name = Path(metadata.get("script_path", "")).name if metadata.get("script_path") else "?"
        ppt_name = Path(metadata.get("ppt_path", "")).name if metadata.get("ppt_path") else "?"
        current_title = status_info.get("current_title")
        flags: List[str] = []
        if current_title:
            flags.append(f"<span class='session-flag'>Slide: {current_title}</span>")
        if status_info.get("paused"):
            flags.append("<span class='session-flag'>Paused</span>")
        queued = status_info.get("queued")
        if queued:
            flags.append(f"<span class='session-flag'>Queue: {queued}</span>")
        selected_class = " selected" if sid == selected_id else ""
        parts.append(
            f"""
            <div class=\"session-card status-{status_label}{selected_class}\">
                <div class=\"session-header\">
                    <span class=\"session-id\">#{sid[:8]}</span>
                    <span class=\"session-status {status_label}\">{status_label.capitalize()}</span>
                </div>
                <div class=\"session-meta\">
                    <span><strong>Slides:</strong> {current_slide}/{slides_total or '?'} </span>
                    <span><strong>Script:</strong> {script_name}</span>
                    <span><strong>PPT:</strong> {ppt_name}</span>
                </div>
                <div class=\"progress-container\">
                    <div class=\"progress-bar\" style=\"width: {progress_pct:.1f}%\"></div>
                </div>
                <div class=\"progress-text\">{progress_pct:.1f}% complete</div>
                {('<div class=\"session-flags\">' + ''.join(flags) + '</div>') if flags else ''}
            </div>
            """
        )
    parts.append("</div>")
    return "\n".join(parts)

def format_present_status(detail: Dict[str, Any]) -> str:
    if not isinstance(detail, dict):
        return "No active presentation sessions."
    if "error" in detail:
        return f"Error: {detail['error']}"
    session = detail.get("session", {}) or {}
    status_info = detail.get("status", {}) or {}
    metadata = session.get("metadata", {}) or {}
    lines = [
        f"**Session:** {session.get('id', '?')} ({session.get('status', 'unknown')})",
    ]
    slides_total = status_info.get("slides_total") or metadata.get("slides_total")
    current_slide = status_info.get("current_slide") or 0
    lines.append(f"**Slides:** {current_slide}/{slides_total or '?'}")
    title = status_info.get("current_title")
    if title:
        lines.append(f"**Slide title:** {title}")
    script_name = Path(metadata.get("script_path", "")).name if metadata.get("script_path") else "?"
    lines.append(f"**Script:** {script_name}")
    ppt_name = Path(metadata.get("ppt_path", "")).name if metadata.get("ppt_path") else "?"
    lines.append(f"**PowerPoint:** {ppt_name}")
    paused = status_info.get("paused")
    if paused is not None:
        lines.append(f"**Paused:** {'Yes' if paused else 'No'}")
    queued = status_info.get("queued")
    if queued is not None:
        lines.append(f"**Queued motions:** {queued}")
    engine_session = status_info.get("session")
    if engine_session:
        lines.append(f"**Engine session:** {engine_session}")
    return "\n".join(lines)

def build_ui(
    controller_url: str = DEFAULT_CONTROLLER_URL,
    speaker_url: str = DEFAULT_SPEAKER_URL):
    result = gr.Blocks()
    with result:
        speakers = fetch_speakers(speaker_url)
        if not speakers:
            speakers = [{"name": "Default", "styles": [{"id": 888753760, "name": "Default"}]}]
        speaker_names = [sp.get("name", f"Speaker {idx}") for idx, sp in enumerate(speakers)]
        default_speaker = speaker_names[0]

        def style_choices(name: str) -> List[str]:
            for sp in speakers:
                if sp.get("name") == name:
                    return [f"{style.get('name', 'Style')} ({style.get('id')})" for style in sp.get('styles', [])]
            return []

        initial_styles = style_choices(default_speaker)
        default_style_choice = initial_styles[0] if initial_styles else f"Default ({speakers[0].get('styles', [{}])[0].get('id', 888753760)})"

        present_selection: Dict[str, Optional[str]] = {"id": None}
        selection_lock = threading.Lock()
        event_log: deque[str] = deque(maxlen=100)
        slide_event_tracker: Dict[str, int] = {}
        status_event_tracker: Dict[str, str] = {}
        snapshot_cache: Dict[str, Dict[str, Any]] = {}

        def parse_style(choice: Optional[str]) -> int:
            if not choice:
                return speakers[0].get("styles", [{}])[0].get("id", 888753760)
            try:
                return int(choice.split("(")[-1].strip(")"))
            except Exception:
                return speakers[0].get("styles", [{}])[0].get("id", 888753760)

        def submit_talk(text: str, style_choice: Optional[str], sync: bool) -> str:
            if not text:
                return "Please enter text"
            style_id = parse_style(style_choice)
            payload = {
                "mode": "talk",
                "payload": {
                    "text": text,
                    "style_id": style_id,
                    "sync": sync,
                },
            }
            resp = create_session(controller_url, payload)
            if isinstance(resp, dict) and "error" in resp and len(resp) == 1:
                return resp["error"]
            return json.dumps(resp, ensure_ascii=False, indent=2) if isinstance(resp, dict) else str(resp)


        def submit_present(
            style_choice: Optional[str],
            script_upload: Optional[str],
            script_choice: Optional[str],
            ppt_upload: Optional[str],
            ppt_choice: Optional[str],
            wait_seconds: float,
            timeout_seconds: float,
            uploads_cache: Optional[List[Dict[str, Any]]],
        ) -> Tuple[str, Optional[str]]:
            uploads = uploads_cache or []
            files_by_name = {entry.get("name"): entry for entry in uploads if isinstance(entry, dict)}
        
            def resolve_existing(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                entry = files_by_name.get(name)
                if entry:
                    return entry.get("path")
                return None
        
            style_id = parse_style(style_choice)
        
            script_server_path = resolve_existing(script_choice)
            if not script_server_path and script_upload:
                script_server_path, err = upload_asset(controller_url, script_upload)
                if err:
                    return err, None
                script_choice = Path(script_server_path or "").name if script_server_path else script_choice
            if not script_server_path:
                return "Please provide or select a script (.md or .txt)", None
        
            ppt_server_path = resolve_existing(ppt_choice)
            if not ppt_server_path and ppt_upload:
                ppt_server_path, err = upload_asset(controller_url, ppt_upload)
                if err:
                    return err, None
                ppt_choice = Path(ppt_server_path or "").name if ppt_server_path else ppt_choice
        
            payload: Dict[str, Any] = {
                "mode": "present",
                "payload": {
                    "script_path": script_server_path,
                    "wait": float(wait_seconds),
                    "timeout": float(timeout_seconds),
                    "style_id": style_id,
                },
            }
            if ppt_server_path:
                payload["payload"]["ppt_path"] = ppt_server_path
        
            resp = create_session(controller_url, payload)
            if isinstance(resp, dict) and "error" in resp and len(resp) == 1:
                return resp["error"], None
            session_id = resp.get("session", {}).get("id") if isinstance(resp, dict) else None
            message = json.dumps(resp, ensure_ascii=False, indent=2) if isinstance(resp, dict) else str(resp)
            return message, session_id
        
        def get_event_log_text() -> str:
            return "\n".join(event_log) if event_log else "No events yet."
        
        def load_upload_choices() -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], List[Tuple[str, str]]]:
            files = list_uploaded_files(controller_url)
            script_entries = [entry for entry in files if Path(entry.get("name", "")).suffix.lower() in (".md", ".txt")]
            ppt_entries = [entry for entry in files if Path(entry.get("name", "")).suffix.lower() in (".ppt", ".pptx")]
        
            def build_choices(items: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
                choices: List[Tuple[str, str]] = []
                for item in items:
                    name = item.get("name")
                    if not name:
                        continue
                    size = item.get("size", 0)
                    label = f"{name} ({_human_size(int(size))})"
                    choices.append((label, name))
                return choices
        
            return files, build_choices(script_entries), build_choices(ppt_entries)
        
        def refresh_upload_components(current_script: Optional[str], current_ppt: Optional[str]) -> Tuple[gr.Dropdown, gr.Dropdown, List[Dict[str, Any]], str]:
            files, script_choices, ppt_choices = load_upload_choices()
            script_values = {value for _, value in script_choices}
            ppt_values = {value for _, value in ppt_choices}
            script_value = current_script if current_script in script_values else None
            ppt_value = current_ppt if current_ppt in ppt_values else None
            return (
                gr.update(choices=script_choices, value=script_value),
                gr.update(choices=ppt_choices, value=ppt_value),
                files,
                "",
            )
        
        def delete_upload_entry(target: str, script_selected: Optional[str], ppt_selected: Optional[str]) -> Tuple[gr.Dropdown, gr.Dropdown, List[Dict[str, Any]], str]:
            filename = script_selected if target == "script" else ppt_selected
            if not filename:
                files, script_choices, ppt_choices = load_upload_choices()
                script_values = {value for _, value in script_choices}
                ppt_values = {value for _, value in ppt_choices}
                script_value = script_selected if script_selected in script_values else None
                ppt_value = ppt_selected if ppt_selected in ppt_values else None
                return (
                    gr.update(choices=script_choices, value=script_value),
                    gr.update(choices=ppt_choices, value=ppt_value),
                    files,
                    "Select a file to delete first.",
                )
            error = delete_uploaded_file(controller_url, filename)
            if error:
                files, script_choices, ppt_choices = load_upload_choices()
                script_values = {value for _, value in script_choices}
                ppt_values = {value for _, value in ppt_choices}
                script_value = script_selected if script_selected in script_values else None
                ppt_value = ppt_selected if ppt_selected in ppt_values else None
                return (
                    gr.update(choices=script_choices, value=script_value),
                    gr.update(choices=ppt_choices, value=ppt_value),
                    files,
                    error,
                )
            files, script_choices, ppt_choices = load_upload_choices()
            script_values = {value for _, value in script_choices}
            ppt_values = {value for _, value in ppt_choices}
            script_value = script_selected if (target != "script" and script_selected in script_values) else None
            ppt_value = ppt_selected if (target != "ppt" and ppt_selected in ppt_values) else None
            message = f"Deleted {filename}."
            return (
                gr.update(choices=script_choices, value=script_value),
                gr.update(choices=ppt_choices, value=ppt_value),
                files,
                message,
            )
        
        def append_log(event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
            timestamp = time.strftime("%H:%M:%S")
            summary: Optional[str] = None
            session_ref: Optional[str] = None
            if isinstance(payload, dict):
                session_ref = payload.get('session')
                if event_name == "presentation_started":
                    summary = f"session={session_ref} slides={payload.get('slides')}"
                elif event_name == "presentation_slide_started":
                    summary = f"slide={payload.get('slide')} title={payload.get('title') or '?'}"
                elif event_name == "presentation_slide_completed":
                    summary = f"slide={payload.get('slide')} status={payload.get('status')}"
                elif event_name == "presentation_finished":
                    summary = f"status={payload.get('status')} slides={payload.get('slides')}"
                elif event_name == "presentation_error":
                    summary = f"stage={payload.get('stage')} error={payload.get('error')}"
                elif event_name == "presentation_stopped":
                    summary = f"session={session_ref}"
                elif event_name == "speech_queued":
                    summary = f"session={session_ref} request={payload.get('request')}"
                elif event_name == "speech_completed":
                    summary = f"session={session_ref} status={payload.get('status')}"
                elif event_name == "speech_stopped":
                    summary = f"session={session_ref}"
                elif event_name == "speech_error":
                    summary = f"session={session_ref} error={payload.get('error')}"
                elif event_name == "presentation_slide_progress":
                    summary = f"session={session_ref} slide={payload.get('slide')} title={payload.get('title') or '?'}"
                elif event_name == "presentation_status_change":
                    summary = f"session={session_ref} status={payload.get('status')}"
            if summary is None and payload is not None:
                try:
                    payload_text = json.dumps(payload, ensure_ascii=False)
                except TypeError:
                    payload_text = str(payload)
                summary = payload_text
            message = f"[{timestamp}] {event_name}: {summary}" if summary else f"[{timestamp}] {event_name}"
            event_log.append(message)
            if session_ref:
                if event_name in {"presentation_slide_started", "presentation_slide_completed", "presentation_slide_progress"} and isinstance(payload, dict):
                    slide = payload.get('slide')
                    if slide is not None:
                        slide_event_tracker[session_ref] = slide
                if event_name in {"presentation_finished", "presentation_error", "presentation_stopped", "presentation_started", "presentation_status_change"} and isinstance(payload, dict):
                    status_value = payload.get('status')
                    if status_value is None:
                        if event_name == "presentation_error":
                            status_value = 'error'
                        elif event_name == "presentation_stopped":
                            status_value = 'stopped'
                        elif event_name == "presentation_started":
                            status_value = 'running'
                    if status_value is not None:
                        status_event_tracker[session_ref] = status_value
                if event_name in {"presentation_slide_progress"} and isinstance(payload, dict):
                    slide_event_tracker[session_ref] = payload.get('slide') or slide_event_tracker.get(session_ref, 0)



        def track_progress(session_id: Optional[str], detail: Dict[str, Any]) -> None:
            if not session_id:
                return
            status_info = detail.get('status', {}) or {}
            session_info = detail.get('session', {}) or {}
            metadata = session_info.get('metadata', {}) or {}
            current_slide = status_info.get('current_slide') or 0
            current_title = status_info.get('current_title')
            slides_total = status_info.get('slides_total') or metadata.get('slides_total')
            status_label = session_info.get('status')
            prev = snapshot_cache.get(session_id)
            if prev is None:
                snapshot_cache[session_id] = {
                    'slide': current_slide,
                    'status': status_label,
                }
                return
            recorded_slide = slide_event_tracker.get(session_id)
            if current_slide and current_slide != prev.get('slide') and current_slide != recorded_slide:
                append_log(
                    'presentation_slide_progress',
                    {
                        'session': session_id,
                        'slide': current_slide,
                        'slides_total': slides_total,
                        'title': current_title,
                    },
                )
                slide_event_tracker[session_id] = current_slide
            recorded_status = status_event_tracker.get(session_id)
            if status_label and status_label != prev.get('status') and status_label != recorded_status:
                append_log(
                    'presentation_status_change',
                    {
                        'session': session_id,
                        'status': status_label,
                    },
                )
                status_event_tracker[session_id] = status_label
            snapshot_cache[session_id] = {
                'slide': current_slide,
                'status': status_label,
            }

        def present_snapshot(selected_session: Optional[str]) -> Tuple[str, Any, str, Optional[str]]:
            sessions = list_present_sessions(controller_url)
            detail_map: Dict[str, Dict[str, Any]] = {}
            for session in sessions:
                sid = session.get("id")
                if not sid:
                    continue
                detail = get_session_status(controller_url, sid)
                if isinstance(detail, dict):
                    detail.setdefault("session", session)
                else:
                    detail = {"session": session, "status": {}, "error": detail}
                if "session" not in detail or not detail["session"]:
                    detail["session"] = session
                detail_map[sid] = detail
            if not detail_map:
                with selection_lock:
                    present_selection["id"] = None
                return (
                    "<div class='session-empty'>No active presentation sessions.</div>",
                    gr.update(choices=[], value=None),
                    "No active presentation sessions.",
                    None,
                )
            ordered_ids = sorted(
                detail_map.keys(),
                key=lambda sid: detail_map[sid].get("session", {}).get("created_at", 0) or 0,
                reverse=True,
            )
            active_ids = [
                sid
                for sid in ordered_ids
                if detail_map[sid].get("session", {}).get("status") not in {"completed", "stopped"}
            ]
            if selected_session not in detail_map:
                selected_session = active_ids[0] if active_ids else ordered_ids[0]
            cards_html = render_present_cards(detail_map, selected_session)
            status_text = (
                format_present_status(detail_map[selected_session]) if selected_session else "No active presentation sessions."
            )
            dropdown = gr.update(choices=ordered_ids, value=selected_session)
            with selection_lock:
                present_selection["id"] = selected_session
            if selected_session:
                track_progress(selected_session, detail_map[selected_session])
            return cards_html, dropdown, status_text, selected_session

        def refresh_present(selected_session: Optional[str]):
            cards_html, dropdown, status_text, selected_session = present_snapshot(selected_session)
            return cards_html, dropdown, status_text, get_event_log_text(), selected_session

        def refresh_sessions_json() -> str:
            sessions = list_sessions(controller_url)
            return json.dumps(sessions, ensure_ascii=False, indent=2) or "[]"

        def watch_present_sessions(initial_selection: Optional[str]):
            selected = initial_selection
            cards_html, dropdown, status_text, selected = present_snapshot(selected)
            yield cards_html, dropdown, status_text, get_event_log_text(), selected

            events_q: Queue[Tuple[str, Dict[str, Any]]] = Queue()
            stop_event = threading.Event()

            def pump_events() -> None:
                try:
                    for event in _iter_controller_events(controller_url, stop_event):
                        events_q.put(event)
                except Exception as exc:
                    events_q.put(("__error__", {"error": str(exc)}))
                finally:
                    events_q.put(("__closed__", {}))

            listener = threading.Thread(
                target=pump_events,
                name="present-event-listener",
                daemon=True,
            )
            listener.start()

            last_emit = time.monotonic()
            try:
                while True:
                    try:
                        event_name, payload = events_q.get(timeout=2.0)
                    except Empty:
                        event_name, payload = "__tick__", {}
                    if event_name == "__closed__":
                        append_log("stream_closed")
                        break
                    log_name: Optional[str] = None
                    log_payload: Optional[Dict[str, Any]] = None
                    if event_name == "__tick__":
                        # yield even on tick to keep snapshot cache up to date
                        pass
                    elif event_name == "__timeout__":
                        log_name, log_payload = "stream_timeout", None
                    elif event_name == "__connected__":
                        log_name, log_payload = "stream_connected", None
                    elif event_name == "__error__":
                        log_name, log_payload = "stream_error", payload or {}
                    elif event_name in PRESENT_EVENT_TYPES:
                        log_name, log_payload = event_name, payload or {}
                    else:
                        continue
                    if event_name not in {"__tick__", "__timeout__"} and log_name:
                        append_log(log_name, log_payload)
                    with selection_lock:
                        selected = present_selection["id"]
                    cards_html, dropdown, status_text, selected = present_snapshot(selected)
                    yield cards_html, dropdown, status_text, get_event_log_text(), selected
                    last_emit = time.monotonic()
            finally:
                stop_event.set()
                listener.join(timeout=1.0)
        
        def test_event_stream() -> str:
            try:
                with requests.get(
                    f"{controller_url.rstrip('/')}/events",
                    stream=True,
                    timeout=(3, 5),
                    headers={"Accept": "text/event-stream"},
                ) as resp:
                    resp.raise_for_status()
                    start = time.monotonic()
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if raw_line:
                            return f"Received: {raw_line}"
                        if time.monotonic() - start > 3:
                            break
                return "Connected to /events but no data within 3 seconds."
            except requests_exceptions.ReadTimeout:
                return "Connected, but no events arrived before the timeout."
            except Exception as exc:
                return f"Error: {exc}"
        
        demo = gr.Blocks(title="AITuber Controller UI")
        with demo:
            gr.HTML(STYLE)
            gr.Markdown("## AITuber Controller")
            gr.Markdown(
                "Upload presentation assets and send requests to the controller service. Talk and Present modes expose their own controls below."
            )

            with gr.Row():
                speaker_dropdown = gr.Dropdown(choices=speaker_names, value=default_speaker, label="Speaker")
                style_dropdown = gr.Dropdown(choices=initial_styles, value=default_style_choice, label="Style")
                speaker_dropdown.change(
                    lambda name: gr.update(
                        choices=(choices := style_choices(name)),
                        value=choices[0] if choices else None,
                    ),
                    inputs=[speaker_dropdown],
                    outputs=[style_dropdown],
                )

            with gr.Tabs():
                with gr.TabItem("Talk"):
                    talk_text = gr.Textbox(label="Text", lines=4)
                    talk_sync = gr.Checkbox(label="Wait for completion", value=False)
                    talk_button = gr.Button("Send Talk Request")
                    talk_result = gr.Textbox(label="Result", interactive=False)
                    talk_button.click(
                        submit_talk,
                        inputs=[talk_text, style_dropdown, talk_sync],
                        outputs=talk_result,
                    )

                with gr.TabItem("Present"):
                    present_selected_state = gr.State(None)
                    uploads_state = gr.State([])

                    with gr.Row():
                        script_file = gr.File(label="Upload Script (.md, .txt)", file_types=[".md", ".txt"], type="filepath")
                        ppt_file = gr.File(label="Upload PowerPoint (.pptx, .ppt)", file_types=[".pptx", ".ppt"], type="filepath")

                    with gr.Row():
                        script_existing_dropdown = gr.Dropdown(label="Existing scripts", choices=[], value=None, interactive=True)
                        ppt_existing_dropdown = gr.Dropdown(label="Existing PowerPoints", choices=[], value=None, interactive=True)

                    with gr.Row():
                        refresh_uploads_button = gr.Button("Refresh uploads", variant="secondary")
                        delete_script_button = gr.Button("Delete selected script", variant="stop")
                        delete_ppt_button = gr.Button("Delete selected PowerPoint", variant="stop")

                    upload_feedback = gr.Markdown("")

                    wait_slider = gr.Slider(label="Slide delay (seconds)", minimum=0.0, maximum=5.0, value=1.0, step=0.1)
                    timeout_slider = gr.Slider(label="Slide timeout (seconds)", minimum=30.0, maximum=600.0, value=300.0, step=10.0)
                    present_button = gr.Button("Start Presentation")
                    present_result = gr.Textbox(label="Result", interactive=False)

                    present_cards = gr.HTML("<div class='session-empty'>No active presentation sessions.</div>")
                    present_session_dropdown = gr.Dropdown(label="Session", choices=[], value=None)
                    present_status = gr.Markdown("No active presentation sessions.")
                    present_event_log = gr.Textbox(label="Event Stream", lines=8, interactive=False)

                    present_button.click(
                        submit_present,
                        inputs=[
                            style_dropdown,
                            script_file,
                            script_existing_dropdown,
                            ppt_file,
                            ppt_existing_dropdown,
                            wait_slider,
                            timeout_slider,
                            uploads_state,
                        ],
                        outputs=[present_result, present_selected_state],
                    ).then(
                        refresh_present,
                        inputs=[present_selected_state],
                        outputs=[present_cards, present_session_dropdown, present_status, present_event_log, present_selected_state],
                    ).then(
                        refresh_upload_components,
                        inputs=[script_existing_dropdown, ppt_existing_dropdown],
                        outputs=[script_existing_dropdown, ppt_existing_dropdown, uploads_state, upload_feedback],
                    )

                    present_session_dropdown.change(
                        refresh_present,
                        inputs=[present_session_dropdown],
                        outputs=[present_cards, present_session_dropdown, present_status, present_event_log, present_selected_state],
                    )

                    refresh_timer = gr.Timer(value=2.0)
                    refresh_timer.tick(
                        refresh_present,
                        inputs=[present_selected_state],
                        outputs=[present_cards, present_session_dropdown, present_status, present_event_log, present_selected_state],
                        queue=False,
                        show_progress='hidden',
                        trigger_mode='always_last',
                    )

                    refresh_uploads_button.click(
                        refresh_upload_components,
                        inputs=[script_existing_dropdown, ppt_existing_dropdown],
                        outputs=[script_existing_dropdown, ppt_existing_dropdown, uploads_state, upload_feedback],
                    )

                    delete_script_button.click(
                        delete_upload_entry,
                        inputs=[gr.State("script"), script_existing_dropdown, ppt_existing_dropdown],
                        outputs=[script_existing_dropdown, ppt_existing_dropdown, uploads_state, upload_feedback],
                    )

                    delete_ppt_button.click(
                        delete_upload_entry,
                        inputs=[gr.State("ppt"), script_existing_dropdown, ppt_existing_dropdown],
                        outputs=[script_existing_dropdown, ppt_existing_dropdown, uploads_state, upload_feedback],
                    )

                    check_stream_button = gr.Button("Check Event Stream", variant="secondary")
                    check_stream_result = gr.Textbox(label="Stream Check Result", lines=2, interactive=False)
                    check_stream_button.click(
                        test_event_stream,
                        outputs=check_stream_result,
                    )

                    with gr.Row():
                        pause_btn = gr.Button("Pause")
                        resume_btn = gr.Button("Resume")
                        stop_btn = gr.Button("Stop")
                    command_feedback = gr.Textbox(label="Command Result", interactive=False)
                    pause_btn.click(
                        lambda session_id: send_command(controller_url, session_id, "pause"),
                        inputs=[present_session_dropdown],
                        outputs=command_feedback,
                    )
                    resume_btn.click(
                        lambda session_id: send_command(controller_url, session_id, "resume"),
                        inputs=[present_session_dropdown],
                        outputs=command_feedback,
                    )
                    stop_btn.click(
                        lambda session_id: send_command(controller_url, session_id, "stop"),
                        inputs=[present_session_dropdown],
                        outputs=command_feedback,
                    )

                with gr.TabItem("FaceTracker"):
                    facetracker_session_id = gr.State(None)

                    with gr.Row():
                        ft_camera_index = gr.Number(label="Camera Index", value=0, precision=0)
                        ft_osc_host = gr.Textbox(label="VMC Host", value="127.0.0.1")
                        ft_osc_port = gr.Number(label="VMC Port", value=39540, precision=0)
                    
                    ft_flip_camera = gr.Checkbox(label="Flip Camera Horizontally", value=False)

                    with gr.Row():
                        ft_start_button = gr.Button("Start Tracking", variant="primary")
                        ft_stop_button = gr.Button("Stop Tracking")

                    ft_status = gr.Textbox(label="Status", interactive=False)

                    def start_facetracker_web(cam_idx, host, port, flip_cam):
                        payload = {
                            "mode": "facetracker",
                            "payload": {
                                "camera_index": cam_idx,
                                "osc_host": host,
                                "osc_port": port,
                                "flip_camera": flip_cam,
                            },
                        }
                        resp = create_session(controller_url, payload)
                        if "error" in resp:
                            return resp["error"], None
                        
                        session_id = resp.get("session", {}).get("id")
                        status_msg = f"Started session: {session_id}"
                        return status_msg, session_id

                    def stop_facetracker_web(session_id):
                        if not session_id:
                            return "No active session to stop.", None
                        
                        result = send_command(controller_url, session_id, "stop")
                        return f"Stopped session {session_id}. Result: {result}", None

                    ft_start_button.click(
                        start_facetracker_web,
                        inputs=[ft_camera_index, ft_osc_host, ft_osc_port, ft_flip_camera],
                        outputs=[ft_status, facetracker_session_id],
                    )

                    ft_stop_button.click(
                        stop_facetracker_web,
                        inputs=[facetracker_session_id],
                        outputs=[ft_status, facetracker_session_id],
                    )

                with gr.TabItem("Sessions"):
                    sessions_output = gr.Textbox(label="Sessions", lines=10, interactive=False)
                    refresh_button = gr.Button("Refresh")
                    refresh_button.click(
                        refresh_sessions_json,
                        outputs=sessions_output,
                    )

        demo.load(
            watch_present_sessions,
            inputs=[present_selected_state],
            outputs=[present_cards, present_session_dropdown, present_status, present_event_log, present_selected_state],
            queue=False,
        )

        demo.load(
            refresh_upload_components,
            inputs=[gr.State(None), gr.State(None)],
            outputs=[script_existing_dropdown, ppt_existing_dropdown, uploads_state, upload_feedback],
        )

    return result
    
    
    
    if __name__ == "__main__":
        ui_app = build_ui()
        ui_app.launch(server_name="0.0.0.0", server_port=8000)

