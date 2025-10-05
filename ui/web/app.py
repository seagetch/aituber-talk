"""Gradio UI targeting the controller service with mode-specific controls and live updates."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests

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
        script_name = Path(metadata.get("script_path", "")).name if metadata.get("script_path") else "—"
        ppt_name = Path(metadata.get("ppt_path", "")).name if metadata.get("ppt_path") else "—"
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
        f"**Session:** {session.get('id', '—')} ({session.get('status', 'unknown')})",
    ]
    slides_total = status_info.get("slides_total") or metadata.get("slides_total")
    current_slide = status_info.get("current_slide") or 0
    lines.append(f"**Slides:** {current_slide}/{slides_total or '?'}")
    title = status_info.get("current_title")
    if title:
        lines.append(f"**Slide title:** {title}")
    script_name = Path(metadata.get("script_path", "")).name if metadata.get("script_path") else "—"
    lines.append(f"**Script:** {script_name}")
    ppt_name = Path(metadata.get("ppt_path", "")).name if metadata.get("ppt_path") else "—"
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
    speaker_url: str = DEFAULT_SPEAKER_URL,
) -> gr.Blocks:
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
        script_path: Optional[str],
        ppt_path: Optional[str],
        wait_seconds: float,
        timeout_seconds: float,
    ) -> Tuple[str, Optional[str]]:
        style_id = parse_style(style_choice)
        script_uploaded, err = upload_asset(controller_url, script_path)
        if err:
            return err, None
        if not script_uploaded:
            return "Please upload a script file (.md or .txt)", None
        payload: Dict[str, Any] = {
            "mode": "present",
            "payload": {
                "script_path": script_uploaded,
                "wait": float(wait_seconds),
                "timeout": float(timeout_seconds),
                "style_id": style_id,
            },
        }
        if ppt_path:
            ppt_uploaded, err = upload_asset(controller_url, ppt_path)
            if err:
                return err, None
            if ppt_uploaded:
                payload["payload"]["ppt_path"] = ppt_uploaded
        resp = create_session(controller_url, payload)
        if isinstance(resp, dict) and "error" in resp and len(resp) == 1:
            return resp["error"], None
        session_id = resp.get("session", {}).get("id") if isinstance(resp, dict) else None
        message = json.dumps(resp, ensure_ascii=False, indent=2) if isinstance(resp, dict) else str(resp)
        return message, session_id

    def refresh_present(selected_session: Optional[str]):
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
        active_ids = [sid for sid in ordered_ids if detail_map[sid].get("session", {}).get("status") not in {"completed", "stopped"}]
        if selected_session not in detail_map:
            selected_session = active_ids[0] if active_ids else ordered_ids[0]
        cards_html = render_present_cards(detail_map, selected_session)
        status_text = format_present_status(detail_map[selected_session]) if selected_session else "No active presentation sessions."
        dropdown = gr.update(choices=ordered_ids, value=selected_session)
        return cards_html, dropdown, status_text, selected_session

    def refresh_sessions_json() -> str:
        sessions = list_sessions(controller_url)
        return json.dumps(sessions, ensure_ascii=False, indent=2) or "[]"

    with gr.Blocks(title="AITuber Controller UI") as demo:
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
                script_file = gr.File(label="Script file (Markdown)", file_types=[".md", ".txt"], type="filepath")
                ppt_file = gr.File(label="PowerPoint file (optional)", file_types=[".pptx", ".ppt"], type="filepath")
                wait_slider = gr.Slider(label="Slide delay (seconds)", minimum=0.0, maximum=5.0, value=1.0, step=0.1)
                timeout_slider = gr.Slider(label="Slide timeout (seconds)", minimum=30.0, maximum=600.0, value=300.0, step=10.0)
                present_button = gr.Button("Start Presentation")
                present_result = gr.Textbox(label="Result", interactive=False)

                present_cards = gr.HTML("<div class='session-empty'>No active presentation sessions.</div>")
                present_session_dropdown = gr.Dropdown(label="Session", choices=[], value=None)
                present_status = gr.Markdown("No active presentation sessions.")

                present_button.click(
                    submit_present,
                    inputs=[style_dropdown, script_file, ppt_file, wait_slider, timeout_slider],
                    outputs=[present_result, present_selected_state],
                ).then(
                    refresh_present,
                    inputs=[present_selected_state],
                    outputs=[present_cards, present_session_dropdown, present_status, present_selected_state],
                )

                present_session_dropdown.change(
                    refresh_present,
                    inputs=[present_session_dropdown],
                    outputs=[present_cards, present_session_dropdown, present_status, present_selected_state],
                )

                #gr.Poll(
                #    refresh_present,
                #    every=1.0,
                #    inputs=[present_selected_state],
                #    outputs=[present_cards, present_session_dropdown, present_status, present_selected_state],
                #)

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

            with gr.TabItem("Sessions"):
                sessions_output = gr.Textbox(label="Sessions", lines=10, interactive=False)
                refresh_button = gr.Button("Refresh")
                refresh_button.click(
                    refresh_sessions_json,
                    outputs=sessions_output,
                )

        demo.load(
            refresh_present,
            inputs=[present_selected_state],
            outputs=[present_cards, present_session_dropdown, present_status, present_selected_state],
        )

    return demo


if __name__ == "__main__":
    ui_app = build_ui()
    ui_app.launch(server_name="0.0.0.0", server_port=8000)
