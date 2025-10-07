"""
PyQt UI targeting the controller service with mode-specific controls and live updates.
This application is a PyQt-based equivalent of the Gradio web UI.
"""

import json
import mimetypes
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

import requests
from PyQt6.QtCore import (
    QObject, QRunnable, QThreadPool, pyqtSignal,
    pyqtSlot, QThread, QTimer)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget)
from requests import exceptions as requests_exceptions

# --- Constants and Configuration ---
DEFAULT_CONTROLLER_URL = os.getenv("AITUBER_CONTROLLER_URL", "http://127.0.0.1:8001")
DEFAULT_SPEAKER_URL = os.getenv("AIVIS_SPEAKER_URL", "http://127.0.0.1:10101")

# --- Backend Communication (copied and adapted from ui/web/app.py) ---

def _fetch_json(method: str, url: str, *, timeout: float = 10, **kwargs) -> Dict[str, Any]:
    """Makes an HTTP request and returns the JSON response."""
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return {"raw": resp.text}
    except Exception as exc:
        return {"error": str(exc)}

def fetch_speakers(speaker_url: str) -> List[Dict[str, Any]]:
    """Fetches available speakers and their styles."""
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
    """Creates a new session (talk or present)."""
    return _fetch_json("POST", f"{controller_url}/sessions", json=payload, timeout=30)

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
                            payload = json.loads(data_text) if data_text else {}
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
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if stop_event.is_set():
                return
            yield "__error__", {"error": str(exc)}
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)

# --- Event Stream Listener Thread ---

class EventStreamListener(QThread):
    """Listens to the controller's server-sent events in a background thread."""
    newEvent = pyqtSignal(str, dict)

    def __init__(self, controller_url: str):
        super().__init__()
        self.controller_url = controller_url
        self._stop_event = threading.Event()

    def run(self):
        self.newEvent.emit("__log__", {"message": "Starting event stream listener..."})
        for event_name, payload in _iter_controller_events(self.controller_url, self._stop_event):
            if self._stop_event.is_set():
                break
            self.newEvent.emit(event_name, payload)
        # Final log message is omitted to prevent errors during shutdown.

    def stop(self):
        """Sets an event to gracefully stop the thread."""
        self.newEvent.emit("__log__", {"message": "Stopping event stream listener..."})
        self._stop_event.set()

# --- Worker Thread for non-blocking requests ---

class WorkerSignals(QObject):
    """Defines signals available from a running worker thread."""
    result = pyqtSignal(object) # Can emit dict or list
    error = pyqtSignal(str)
    finished = pyqtSignal()

class Worker(QRunnable):
    """Worker thread for executing long-running tasks."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

# --- Main Application Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AITuber Controller")
        self.setGeometry(100, 100, 1200, 800)
        self.threadpool = QThreadPool()
        self.event_log_deque = deque(maxlen=200)
        self.speakers = []

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.speaker_style_group = QGroupBox("Speaker Configuration")
        self.speaker_style_layout = QFormLayout()
        self.speaker_dropdown = QComboBox()
        self.style_dropdown = QComboBox()
        self.speaker_style_layout.addRow("Speaker:", self.speaker_dropdown)
        self.speaker_style_layout.addRow("Style:", self.style_dropdown)
        self.speaker_style_group.setLayout(self.speaker_style_layout)
        self.layout.addWidget(self.speaker_style_group)

        self.tabs = QTabWidget()
        self.talk_tab = QWidget()
        self.present_tab = QWidget()
        self.sessions_tab = QWidget()
        self.tabs.addTab(self.talk_tab, "Talk")
        self.tabs.addTab(self.present_tab, "Present")
        self.tabs.addTab(self.sessions_tab, "Sessions")
        self.layout.addWidget(self.tabs)

        self.init_talk_tab()
        self.init_present_tab()
        self.init_sessions_tab()
        
        self.load_speakers()

        self.speaker_dropdown.currentTextChanged.connect(self.update_style_choices)

        self.start_event_listener()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_present_sessions)
        self.refresh_timer.start(5000)

    def closeEvent(self, event):
        """Signal background threads to stop and trigger app exit."""
        self.refresh_timer.stop()
        self.event_listener.stop()  # Set the stop flag
        QApplication.instance().quit()  # Exit the event loop immediately
        super().closeEvent(event)

    def init_talk_tab(self):
        layout = QVBoxLayout(self.talk_tab)
        form_layout = QFormLayout()
        self.talk_text = QTextEdit()
        self.talk_text.setPlaceholderText("Enter text to speak...")
        form_layout.addRow("Text:", self.talk_text)
        self.talk_sync_checkbox = QCheckBox("Wait for completion")
        form_layout.addRow(self.talk_sync_checkbox)
        layout.addLayout(form_layout)
        self.talk_button = QPushButton("Send Talk Request")
        self.talk_result = QTextEdit()
        self.talk_result.setReadOnly(True)
        layout.addWidget(self.talk_button)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.talk_result)
        self.talk_button.clicked.connect(self.submit_talk)

    def init_present_tab(self):
        layout = QGridLayout(self.present_tab)
        controls_group = QGroupBox("Presentation Setup")
        controls_layout = QFormLayout()

        self.script_path_edit = QLineEdit()
        self.script_path_edit.setReadOnly(True)
        script_button = QPushButton("Select Script")
        script_button.clicked.connect(lambda: self.select_file(self.script_path_edit, "Scripts (*.md *.txt)"))
        script_layout = QHBoxLayout()
        script_layout.addWidget(self.script_path_edit)
        script_layout.addWidget(script_button)
        controls_layout.addRow("Script File:", script_layout)

        self.ppt_path_edit = QLineEdit()
        self.ppt_path_edit.setReadOnly(True)
        ppt_button = QPushButton("Select PowerPoint")
        ppt_button.clicked.connect(lambda: self.select_file(self.ppt_path_edit, "PowerPoint Files (*.pptx *.ppt)"))
        ppt_layout = QHBoxLayout()
        ppt_layout.addWidget(self.ppt_path_edit)
        ppt_layout.addWidget(ppt_button)
        controls_layout.addRow("PowerPoint File:", ppt_layout)

        self.wait_slider = QDoubleSpinBox()
        self.wait_slider.setRange(0.0, 5.0)
        self.wait_slider.setValue(1.0)
        self.wait_slider.setSingleStep(0.1)
        controls_layout.addRow("Slide Delay (s):", self.wait_slider)

        self.timeout_slider = QDoubleSpinBox()
        self.timeout_slider.setRange(30.0, 600.0)
        self.timeout_slider.setValue(300.0)
        self.timeout_slider.setSingleStep(10.0)
        controls_layout.addRow("Slide Timeout (s):", self.timeout_slider)
        
        controls_group.setLayout(controls_layout)
        
        self.start_present_button = QPushButton("Start Presentation")
        self.present_result = QTextEdit()
        self.present_result.setReadOnly(True)

        left_vbox = QVBoxLayout()
        left_vbox.addWidget(controls_group)
        left_vbox.addWidget(self.start_present_button)
        left_vbox.addWidget(QLabel("Result:"))
        left_vbox.addWidget(self.present_result)
        left_vbox.addStretch()
        layout.addLayout(left_vbox, 0, 0)

        status_group = QGroupBox("Live Status")
        status_layout = QVBoxLayout()
        self.present_session_dropdown = QComboBox()
        self.present_session_dropdown.currentTextChanged.connect(self.update_present_status_display)

        session_controls_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.pause_btn.clicked.connect(lambda: self.send_present_command("pause"))
        self.resume_btn.clicked.connect(lambda: self.send_present_command("resume"))
        self.stop_btn.clicked.connect(lambda: self.send_present_command("stop"))
        session_controls_layout.addWidget(self.pause_btn)
        session_controls_layout.addWidget(self.resume_btn)
        session_controls_layout.addWidget(self.stop_btn)

        self.command_feedback = QLineEdit()
        self.command_feedback.setReadOnly(True)
        self.present_status = QTextEdit()
        self.present_status.setReadOnly(True)
        self.present_event_log = QTextEdit()
        self.present_event_log.setReadOnly(True)

        status_layout.addWidget(QLabel("Active Session:"))
        status_layout.addWidget(self.present_session_dropdown)
        status_layout.addLayout(session_controls_layout)
        status_layout.addWidget(self.command_feedback)
        status_layout.addWidget(QLabel("Status Details:"))
        status_layout.addWidget(self.present_status)
        status_layout.addWidget(QLabel("Event Log:"))
        status_layout.addWidget(self.present_event_log)
        status_group.setLayout(status_layout)
        
        layout.addWidget(status_group, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.start_present_button.clicked.connect(self.submit_present)

    def init_sessions_tab(self):
        layout = QVBoxLayout(self.sessions_tab)
        self.sessions_output = QTextEdit()
        self.sessions_output.setReadOnly(True)
        self.refresh_sessions_button = QPushButton("Refresh")
        layout.addWidget(self.refresh_sessions_button)
        layout.addWidget(self.sessions_output)
        self.refresh_sessions_button.clicked.connect(self.refresh_sessions)

    def start_event_listener(self):
        self.event_listener = EventStreamListener(DEFAULT_CONTROLLER_URL)
        self.event_listener.newEvent.connect(self.handle_event)
        self.event_listener.start()

    def load_speakers(self):
        worker = Worker(fetch_speakers, DEFAULT_SPEAKER_URL)
        worker.signals.result.connect(self.populate_speaker_dropdown)
        self.threadpool.start(worker)

    def populate_speaker_dropdown(self, speakers: List[Dict[str, Any]]):
        self.speakers = speakers
        if not self.speakers:
            self.speakers = [{"name": "Default", "styles": [{"id": 888753760, "name": "Default"}]}]
        self.speaker_dropdown.clear()
        self.speaker_dropdown.addItems([sp.get("name", f"Speaker {i}") for i, sp in enumerate(self.speakers)])

    def update_style_choices(self, speaker_name: str):
        self.style_dropdown.clear()
        for sp in self.speakers:
            if sp.get("name") == speaker_name:
                styles = [f"{style.get('name', 'Style')} ({style.get('id')})" for style in sp.get('styles', [])]
                self.style_dropdown.addItems(styles)
                break

    def get_selected_style_id(self) -> int:
        choice = self.style_dropdown.currentText()
        if not choice or not self.speakers:
            return 888753760
        try:
            return int(choice.split(" ")[-1].strip("()") )
        except (ValueError, IndexError):
            speaker_name = self.speaker_dropdown.currentText()
            for sp in self.speakers:
                if sp.get("name") == speaker_name:
                    return sp.get("styles", [{}])[0].get("id", 888753760)
            return 888753760

    def select_file(self, line_edit: QLineEdit, filter: str):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter)
        if file_path:
            line_edit.setText(file_path)

    @pyqtSlot(str, dict)
    def handle_event(self, event_name: str, payload: dict):
        timestamp = time.strftime("%H:%M:%S")
        if event_name == "__log__":
            log_message = f"[{timestamp}] [SYSTEM] {payload.get('message')}"
        else:
            log_message = f"[{timestamp}] [{event_name}] {json.dumps(payload)}"
        self.event_log_deque.append(log_message)
        self.present_event_log.setText("\n".join(self.event_log_deque))
        self.present_event_log.verticalScrollBar().setValue(self.present_event_log.verticalScrollBar().maximum())
        if event_name.startswith("presentation_"):
            self.refresh_present_sessions()

    @pyqtSlot()
    def refresh_present_sessions(self):
        worker = Worker(list_present_sessions, DEFAULT_CONTROLLER_URL)
        worker.signals.result.connect(self.update_present_sessions_display)
        self.threadpool.start(worker)

    def update_present_sessions_display(self, sessions: List[Dict[str, Any]]):
        current_selection = self.present_session_dropdown.currentText()
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        session_ids = [s.get("id", "") for s in sessions]
        self.present_session_dropdown.blockSignals(True)
        self.present_session_dropdown.clear()
        self.present_session_dropdown.addItems(session_ids)
        self.present_session_dropdown.blockSignals(False)
        if current_selection in session_ids:
            self.present_session_dropdown.setCurrentText(current_selection)
        elif session_ids:
            self.present_session_dropdown.setCurrentIndex(0)
        self.update_present_status_display()

    def update_present_status_display(self):
        session_id = self.present_session_dropdown.currentText()
        if not session_id:
            self.present_status.clear()
            return
        worker = Worker(get_session_status, DEFAULT_CONTROLLER_URL, session_id)
        worker.signals.result.connect(self.format_and_display_status)
        self.threadpool.start(worker)

    def format_and_display_status(self, detail: Dict[str, Any]):
        if not isinstance(detail, dict) or "error" in detail:
            self.present_status.setText(json.dumps(detail, indent=2))
            return
        session = detail.get("session", {}) or {}
        status_info = detail.get("status", {}) or {}
        metadata = session.get("metadata", {}) or {}
        lines = [f"**Session:** {session.get('id', '?')} ({session.get('status', 'unknown')})"]
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
        if status_info.get("paused") is not None:
            lines.append(f"**Paused:** {'Yes' if status_info.get("paused") else 'No'}")
        if status_info.get("queued") is not None:
            lines.append(f"**Queued motions:** {status_info.get("queued")}")
        self.present_status.setText("\n".join(lines))

    @pyqtSlot()
    def submit_talk(self):
        text = self.talk_text.toPlainText()
        if not text:
            self.talk_result.setText("Please enter text.")
            return
        payload = {
            "mode": "talk",
            "payload": {
                "text": text,
                "style_id": self.get_selected_style_id(),
                "sync": self.talk_sync_checkbox.isChecked(),
            },
        }
        worker = Worker(create_session, DEFAULT_CONTROLLER_URL, payload)
        worker.signals.result.connect(lambda res: self.display_general_result(res, self.talk_result))
        self.threadpool.start(worker)

    @pyqtSlot()
    def submit_present(self):
        script_local_path = self.script_path_edit.text()
        if not script_local_path:
            self.present_result.setText("Error: Script file is required.")
            return
        worker = Worker(self._upload_and_start_present, script_local_path, self.ppt_path_edit.text())
        worker.signals.result.connect(lambda res: self.display_general_result(res, self.present_result))
        worker.signals.error.connect(lambda err: self.present_result.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def _upload_and_start_present(self, script_local_path: str, ppt_local_path: Optional[str]) -> Dict[str, Any]:
        def upload_asset(file_path: str) -> Tuple[Optional[str], Optional[str]]:
            if not file_path:
                return None, None
            try:
                filename = os.path.basename(file_path)
                mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                with open(file_path, "rb") as fh:
                    resp = requests.post(
                        f"{DEFAULT_CONTROLLER_URL}/files",
                        files={"file": (filename, fh, mime)},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get("path"), None
            except Exception as exc:
                return None, f"Upload failed for {file_path}: {exc}"

        script_server_path, err = upload_asset(script_local_path)
        if err:
            raise RuntimeError(err)
        if not script_server_path:
            raise RuntimeError("Failed to get server path for script.")

        ppt_server_path = None
        if ppt_local_path:
            ppt_server_path, err = upload_asset(ppt_local_path)
            if err:
                print(f"Warning: PowerPoint upload failed: {err}")

        payload: Dict[str, Any] = {
            "mode": "present",
            "payload": {
                "script_path": script_server_path,
                "wait": self.wait_slider.value(),
                "timeout": self.timeout_slider.value(),
                "style_id": self.get_selected_style_id(),
            },
        }
        if ppt_server_path:
            payload["payload"]["ppt_path"] = ppt_server_path
        return create_session(DEFAULT_CONTROLLER_URL, payload)

    def display_general_result(self, result: dict, target_widget: QTextEdit):
        if "error" in result and len(result) == 1:
            target_widget.setText(result["error"])
        else:
            target_widget.setText(json.dumps(result, ensure_ascii=False, indent=2))
        self.refresh_present_sessions()

    @pyqtSlot()
    def refresh_sessions(self):
        worker = Worker(lambda: _fetch_json("GET", f"{DEFAULT_CONTROLLER_URL}/sessions"))
        worker.signals.result.connect(lambda res: self.sessions_output.setText(json.dumps(res, indent=2)))
        worker.signals.error.connect(lambda err: self.sessions_output.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def send_present_command(self, command: str):
        session_id = self.present_session_dropdown.currentText()
        if not session_id:
            self.command_feedback.setText("No session selected.")
            return
        worker = Worker(send_command, DEFAULT_CONTROLLER_URL, session_id, command)
        worker.signals.result.connect(lambda res: self.command_feedback.setText(str(res)))
        worker.signals.error.connect(lambda err: self.command_feedback.setText(f"Error: {err}"))
        self.threadpool.start(worker)

if __name__ == "__main__":
    try:
        import PyQt6
    except ImportError:
        print("PyQt6 not found. Please install it: pip install PyQt6")
        sys.exit(1)
    try:
        import requests
    except ImportError:
        print("requests not found. Please install it: pip install requests")
        sys.exit(1)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())