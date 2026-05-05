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
    QEvent, QObject, QRunnable, QThreadPool, pyqtSignal,
    pyqtSlot, QThread, QTimer)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QProgressBar, QPushButton, QTabWidget, QTextEdit,
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

def list_codex_sessions(controller_url: str) -> List[Dict[str, Any]]:
    data = _fetch_json("GET", f"{controller_url}/codex/sessions", timeout=10)
    if isinstance(data, dict) and "error" not in data:
        return data.get("sessions", [])
    return []

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

def update_settings(
    controller_url: str,
    jaw_open_scale: float,
    jaw_open_source_index: int,
    jaw_open_zero_value: float,
    jaw_open_one_value: float,
) -> str:
    data = _fetch_json(
        "PATCH",
        f"{controller_url}/settings",
        json={
            "jaw_open_scale": float(jaw_open_scale),
            "jaw_open_source_index": int(jaw_open_source_index),
            "jaw_open_zero_value": float(jaw_open_zero_value),
            "jaw_open_one_value": float(jaw_open_one_value),
        },
        timeout=10,
    )
    if isinstance(data, dict) and "error" in data and len(data) == 1:
        return data["error"]
    return json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data)

def get_settings(controller_url: str) -> Dict[str, Any]:
    data = _fetch_json("GET", f"{controller_url}/settings", timeout=5)
    return data if isinstance(data, dict) and "error" not in data else {}

def create_session(controller_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a new session (talk or present)."""
    return _fetch_json("POST", f"{controller_url}/sessions", json=payload, timeout=30)

def _iter_controller_events(base_url: str, stop_event: threading.Event) -> Iterator[Tuple[str, Dict[str, Any]]]:
    endpoint = f"{base_url.rstrip('/')}/events"
    headers = {"Accept": "text/event-stream"}
    backoff = 1.0
    while not stop_event.is_set():
        try:
            with requests.get(endpoint, stream=True, timeout=(5, 10), headers=headers) as resp:
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
        except requests_exceptions.ReadTimeout:
            if stop_event.is_set():
                return
            yield "__timeout__", {}
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
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
        self.facetracker_session_id: Optional[str] = None
        self._motion_settings_loaded = False
        self._settings_inflight = False
        self._present_refresh_inflight = False
        self._present_refresh_pending = False
        self._present_status_inflight = False
        self._present_status_pending = False
        self._last_progress_log_at = 0.0
        self._ui_updates_paused = False
        self._event_listener_connected = False

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.speaker_style_group = QGroupBox("Speaker Configuration")
        self.speaker_style_layout = QFormLayout()
        self.speaker_dropdown = QComboBox()
        self.style_dropdown = QComboBox()
        self.speaker_style_layout.addRow("Speaker:", self.speaker_dropdown)
        self.speaker_style_layout.addRow("Style:", self.style_dropdown)
        self.jaw_open_scale_spinbox = QDoubleSpinBox()
        self.jaw_open_scale_spinbox.setRange(0.0, 5.0)
        self.jaw_open_scale_spinbox.setValue(1.0)
        self.jaw_open_scale_spinbox.setSingleStep(0.1)
        self.speaker_style_layout.addRow("jawOpen scale:", self.jaw_open_scale_spinbox)
        self.jaw_open_source_index_spinbox = QSpinBox()
        self.jaw_open_source_index_spinbox.setRange(0, 63)
        self.jaw_open_source_index_spinbox.setValue(35)
        self.speaker_style_layout.addRow("jawOpen source index:", self.jaw_open_source_index_spinbox)
        self.jaw_open_zero_value_spinbox = QDoubleSpinBox()
        self.jaw_open_zero_value_spinbox.setRange(-10.0, 10.0)
        self.jaw_open_zero_value_spinbox.setValue(0.0)
        self.jaw_open_zero_value_spinbox.setSingleStep(0.01)
        self.speaker_style_layout.addRow("source <= this gives jawOpen 0:", self.jaw_open_zero_value_spinbox)
        self.jaw_open_one_value_spinbox = QDoubleSpinBox()
        self.jaw_open_one_value_spinbox.setRange(-10.0, 10.0)
        self.jaw_open_one_value_spinbox.setValue(0.15)
        self.jaw_open_one_value_spinbox.setSingleStep(0.01)
        self.speaker_style_layout.addRow("source >= this gives jawOpen 1:", self.jaw_open_one_value_spinbox)
        self.jaw_open_current_label = QLabel("jawOpen current: source ? -> value ?; applied range: <= ? => 0, >= ? => 1")
        self.speaker_style_layout.addRow(self.jaw_open_current_label)
        self.apply_motion_settings_button = QPushButton("Apply Motion Settings")
        self.speaker_style_layout.addRow(self.apply_motion_settings_button)
        self.speaker_style_group.setLayout(self.speaker_style_layout)
        self.layout.addWidget(self.speaker_style_group)

        self.tabs = QTabWidget()
        self.talk_tab = QWidget()
        self.codex_watch_tab = QWidget()
        self.present_tab = QWidget()
        self.facetracker_tab = QWidget()
        self.sessions_tab = QWidget()
        self.tabs.addTab(self.talk_tab, "Talk")
        self.tabs.addTab(self.codex_watch_tab, "Codex Watch")
        self.tabs.addTab(self.present_tab, "Present")
        self.tabs.addTab(self.facetracker_tab, "FaceTracker")
        self.tabs.addTab(self.sessions_tab, "Sessions")
        self.layout.addWidget(self.tabs)

        self.init_talk_tab()
        self.init_codex_watch_tab()
        self.init_present_tab()
        self.init_facetracker_tab()
        self.init_sessions_tab()
        
        self.load_speakers()
        self.load_motion_settings()

        self.speaker_dropdown.currentTextChanged.connect(self.update_style_choices)
        self.apply_motion_settings_button.clicked.connect(self.apply_motion_settings)

        self.start_event_listener()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_present_sessions)
        self.refresh_timer.start(5000)
        self.motion_settings_timer = QTimer(self)
        self.motion_settings_timer.timeout.connect(self.load_motion_settings)
        self.motion_settings_timer.start(1000)

    def closeEvent(self, event):
        """Signal background threads to stop and trigger app exit."""
        self.refresh_timer.stop()
        self.motion_settings_timer.stop()
        self.codex_sessions_timer.stop()
        self.event_listener.stop()
        self.event_listener.wait(1000)
        if self.event_listener.isRunning():
            self.event_listener.terminate()
            self.event_listener.wait(1000)
        self.threadpool.clear()
        QApplication.instance().quit()  # Exit the event loop immediately
        super().closeEvent(event)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            minimized = self.isMinimized()
            if minimized:
                self.pause_ui_updates()
            else:
                self.resume_ui_updates()
        super().changeEvent(event)

    def pause_ui_updates(self):
        if self._ui_updates_paused:
            return
        self._ui_updates_paused = True
        self.refresh_timer.stop()
        self.motion_settings_timer.stop()
        self.codex_sessions_timer.stop()
        if self._event_listener_connected:
            try:
                self.event_listener.newEvent.disconnect(self.handle_event)
            except TypeError:
                pass
            self._event_listener_connected = False

    def resume_ui_updates(self):
        if not self._ui_updates_paused:
            return
        if not self._event_listener_connected:
            self.event_listener.newEvent.connect(self.handle_event)
            self._event_listener_connected = True
        self._ui_updates_paused = False
        self.refresh_timer.start(5000)
        self.motion_settings_timer.start(1000)
        self.codex_sessions_timer.start(3000)
        self.refresh_present_sessions()
        self.refresh_codex_sessions()
        self.load_motion_settings()

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

    def init_codex_watch_tab(self):
        layout = QVBoxLayout(self.codex_watch_tab)
        form_layout = QFormLayout()
        self.codex_session_dropdown = QComboBox()
        form_layout.addRow("Codex session:", self.codex_session_dropdown)
        self.codex_session_details = QTextEdit()
        self.codex_session_details.setReadOnly(True)
        form_layout.addRow("Session details:", self.codex_session_details)
        self.codex_profile = QTextEdit()
        self.codex_profile.setPlainText(
            "性格: 落ち着いていて実務的\n"
            "口調: 短く自然な日本語\n"
            "一人称: 私\n"
            "ユーザーへの呼び方: あなた\n"
            "禁止事項: 内部ログやツール出力をそのまま読まない"
        )
        form_layout.addRow("Character profile:", self.codex_profile)
        self.codex_model_edit = QLineEdit("gpt-5.4-nano")
        form_layout.addRow("OpenAI model:", self.codex_model_edit)
        self.codex_assistant_messages_checkbox = QCheckBox("Assistant-visible messages only")
        self.codex_assistant_messages_checkbox.setChecked(True)
        form_layout.addRow("Watch event types:", self.codex_assistant_messages_checkbox)
        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.refresh_codex_sessions_button = QPushButton("Refresh Codex Sessions")
        self.auto_refresh_codex_sessions_checkbox = QCheckBox("Auto refresh session list")
        self.auto_refresh_codex_sessions_checkbox.setChecked(True)
        self.start_codex_watch_button = QPushButton("Start Watching")
        self.stop_codex_watch_button = QPushButton("Stop Watching")
        buttons_layout.addWidget(self.refresh_codex_sessions_button)
        buttons_layout.addWidget(self.auto_refresh_codex_sessions_checkbox)
        buttons_layout.addWidget(self.start_codex_watch_button)
        buttons_layout.addWidget(self.stop_codex_watch_button)
        layout.addLayout(buttons_layout)
        self.codex_session_list_status = QLabel("Session list: not loaded")
        layout.addWidget(self.codex_session_list_status)

        watch_group = QGroupBox("Codex Watch Status")
        watch_layout = QFormLayout()
        self.codex_watch_state_label = QLabel("stopped")
        self.codex_watch_state_label.setMinimumHeight(34)
        self.codex_watch_state_label.setStyleSheet(
            "QLabel { padding: 8px 12px; border: 1px solid #475569; border-radius: 6px; "
            "background: #111827; color: #cbd5e1; font-weight: 700; }"
        )
        self.set_codex_watch_state("stopped")
        watch_layout.addRow("Watch status:", self.codex_watch_state_label)
        self.codex_last_message = QTextEdit()
        self.codex_last_message.setReadOnly(True)
        self.codex_last_message.setMaximumHeight(100)
        watch_layout.addRow("Last Codex message:", self.codex_last_message)
        self.codex_last_utterance = QTextEdit()
        self.codex_last_utterance.setReadOnly(True)
        self.codex_last_utterance.setMaximumHeight(80)
        watch_layout.addRow("Last generated utterance:", self.codex_last_utterance)
        self.codex_utterance_history = QTextEdit()
        self.codex_utterance_history.setReadOnly(True)
        self.codex_utterance_history.setMaximumHeight(120)
        watch_layout.addRow("Recent utterance history:", self.codex_utterance_history)
        self.codex_error_log = QTextEdit()
        self.codex_error_log.setReadOnly(True)
        self.codex_error_log.setMaximumHeight(80)
        watch_layout.addRow("Error log:", self.codex_error_log)
        watch_group.setLayout(watch_layout)
        layout.addWidget(watch_group)
        self.codex_watch_status = QTextEdit()
        self.codex_watch_status.setReadOnly(True)
        layout.addWidget(QLabel("Event log:"))
        layout.addWidget(self.codex_watch_status)

        self.codex_sessions: Dict[str, Dict[str, Any]] = {}
        self.codex_watch_session_id: Optional[str] = None
        self._codex_refresh_inflight = False
        self.refresh_codex_sessions_button.clicked.connect(self.refresh_codex_sessions)
        self.codex_session_dropdown.currentTextChanged.connect(self.update_codex_session_details)
        self.start_codex_watch_button.clicked.connect(self.submit_codex_watch)
        self.stop_codex_watch_button.clicked.connect(self.stop_codex_watch)
        self.codex_sessions_timer = QTimer(self)
        self.codex_sessions_timer.timeout.connect(self.auto_refresh_codex_sessions)
        self.codex_sessions_timer.start(3000)
        self.refresh_codex_sessions()

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

        self.loop_checkbox = QCheckBox("Loop presentation")
        self.loop_checkbox.setChecked(False)
        controls_layout.addRow("Loop:", self.loop_checkbox)

        self.loop_delay_slider = QDoubleSpinBox()
        self.loop_delay_slider.setRange(0.0, 300.0)
        self.loop_delay_slider.setValue(0.0)
        self.loop_delay_slider.setSingleStep(1.0)
        controls_layout.addRow("Loop Delay (s):", self.loop_delay_slider)

        self.all_buffered_checkbox = QCheckBox("All Buffered Mode")
        self.all_buffered_checkbox.setChecked(False)
        self.all_buffered_checkbox.toggled.connect(
            lambda enabled: self.start_present_button.setText("Prepare Presentation" if enabled else "Start Presentation")
        )
        controls_layout.addRow("All Buffered:", self.all_buffered_checkbox)

        self.timeout_slider = QDoubleSpinBox()
        self.timeout_slider.setRange(30.0, 600.0)
        self.timeout_slider.setValue(300.0)
        self.timeout_slider.setSingleStep(10.0)
        controls_layout.addRow("Slide Timeout (s):", self.timeout_slider)

        self.image_viewer_mode = QComboBox()
        self.image_viewer_mode.addItems(["auto", "on", "off"])
        controls_layout.addRow("Image Viewer:", self.image_viewer_mode)
        
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
        self.play_buffered_btn = QPushButton("Play Buffered Presentation")
        self.play_buffered_btn.setEnabled(False)
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        self.play_buffered_btn.clicked.connect(lambda: self.send_present_command("play"))
        self.pause_btn.clicked.connect(lambda: self.send_present_command("pause"))
        self.resume_btn.clicked.connect(lambda: self.send_present_command("resume"))
        self.stop_btn.clicked.connect(lambda: self.send_present_command("stop"))
        session_controls_layout.addWidget(self.play_buffered_btn)
        session_controls_layout.addWidget(self.pause_btn)
        session_controls_layout.addWidget(self.resume_btn)
        session_controls_layout.addWidget(self.stop_btn)

        self.command_feedback = QLineEdit()
        self.command_feedback.setReadOnly(True)
        self.present_status = QTextEdit()
        self.present_status.setReadOnly(True)
        self.buffer_progress_bar = QProgressBar()
        self.buffer_progress_bar.setRange(0, 100)
        self.buffer_progress_bar.setValue(0)
        self.present_event_log = QTextEdit()
        self.present_event_log.setReadOnly(True)

        status_layout.addWidget(QLabel("Active Session:"))
        status_layout.addWidget(self.present_session_dropdown)
        status_layout.addLayout(session_controls_layout)
        status_layout.addWidget(self.command_feedback)
        status_layout.addWidget(QLabel("All Buffered Progress:"))
        status_layout.addWidget(self.buffer_progress_bar)
        status_layout.addWidget(QLabel("Status Details:"))
        status_layout.addWidget(self.present_status)
        status_layout.addWidget(QLabel("Event Log:"))
        status_layout.addWidget(self.present_event_log)
        status_group.setLayout(status_layout)
        
        layout.addWidget(status_group, 0, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self.start_present_button.clicked.connect(self.submit_present)

    def init_facetracker_tab(self):
        """Set up the UI for the 'FaceTracker' tab."""
        layout = QVBoxLayout(self.facetracker_tab)
        form_layout = QFormLayout()

        self.camera_index_spinbox = QSpinBox()
        self.camera_index_spinbox.setRange(0, 10)
        self.camera_index_spinbox.setValue(0)
        form_layout.addRow("Camera Index:", self.camera_index_spinbox)

        self.osc_host_edit = QLineEdit("127.0.0.1")
        form_layout.addRow("VMC Host:", self.osc_host_edit)

        self.osc_port_spinbox = QSpinBox()
        self.osc_port_spinbox.setRange(1, 65535)
        self.osc_port_spinbox.setValue(39540)
        form_layout.addRow("VMC Port:", self.osc_port_spinbox)

        self.flip_camera_checkbox = QCheckBox("Flip Camera Horizontally")
        form_layout.addRow(self.flip_camera_checkbox)

        layout.addLayout(form_layout)

        buttons_layout = QHBoxLayout()
        self.start_facetracker_button = QPushButton("Start Tracking")
        self.stop_facetracker_button = QPushButton("Stop Tracking")
        buttons_layout.addWidget(self.start_facetracker_button)
        buttons_layout.addWidget(self.stop_facetracker_button)
        layout.addLayout(buttons_layout)

        self.facetracker_status_label = QLabel("Status: Not running")
        layout.addWidget(self.facetracker_status_label)
        layout.addStretch()

        self.start_facetracker_button.clicked.connect(self.start_facetracker)
        self.stop_facetracker_button.clicked.connect(self.stop_facetracker)

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
        self._event_listener_connected = True
        self.event_listener.start()

    def load_speakers(self):
        worker = Worker(fetch_speakers, DEFAULT_SPEAKER_URL)
        worker.signals.result.connect(self.populate_speaker_dropdown)
        self.threadpool.start(worker)

    def load_motion_settings(self):
        if self._ui_updates_paused:
            return
        if self._settings_inflight:
            return
        self._settings_inflight = True
        worker = Worker(get_settings, DEFAULT_CONTROLLER_URL)
        worker.signals.result.connect(self.populate_motion_settings)
        worker.signals.finished.connect(lambda: setattr(self, "_settings_inflight", False))
        self.threadpool.start(worker)

    def populate_motion_settings(self, settings: Dict[str, Any]):
        if self._ui_updates_paused:
            return
        if not isinstance(settings, dict):
            return
        if not self._motion_settings_loaded and "jaw_open_scale" in settings:
            self.jaw_open_scale_spinbox.setValue(float(settings["jaw_open_scale"]))
        if not self._motion_settings_loaded and "jaw_open_source_index" in settings:
            self.jaw_open_source_index_spinbox.setValue(int(settings["jaw_open_source_index"]))
        if not self._motion_settings_loaded and "jaw_open_zero_value" in settings:
            self.jaw_open_zero_value_spinbox.setValue(float(settings["jaw_open_zero_value"]))
        if not self._motion_settings_loaded and "jaw_open_one_value" in settings:
            self.jaw_open_one_value_spinbox.setValue(float(settings["jaw_open_one_value"]))
        self._motion_settings_loaded = True
        source = settings.get("jaw_open_current_source")
        value = settings.get("jaw_open_current_value")
        source_text = f"{float(source):.4f}" if source is not None else "?"
        value_text = f"{float(value):.4f}" if value is not None else "?"
        zero = settings.get("jaw_open_zero_value")
        one = settings.get("jaw_open_one_value")
        zero_text = f"{float(zero):.4f}" if zero is not None else "?"
        one_text = f"{float(one):.4f}" if one is not None else "?"
        self.jaw_open_current_label.setText(
            f"jawOpen current: source {source_text} -> value {value_text}; "
            f"applied range: <= {zero_text} => 0, >= {one_text} => 1"
        )

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

    def apply_motion_settings(self):
        worker = Worker(
            update_settings,
            DEFAULT_CONTROLLER_URL,
            self.jaw_open_scale_spinbox.value(),
            self.jaw_open_source_index_spinbox.value(),
            self.jaw_open_zero_value_spinbox.value(),
            self.jaw_open_one_value_spinbox.value(),
        )
        worker.signals.result.connect(self.motion_settings_applied)
        worker.signals.error.connect(lambda err: self.talk_result.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def motion_settings_applied(self, result: str):
        self.talk_result.setPlainText(str(result))
        self._motion_settings_loaded = False
        self.load_motion_settings()

    def set_codex_watch_state(self, state: str) -> None:
        state = (state or "unknown").lower()
        labels = {
            "running": ("監視中", "#052e16", "#22c55e", "#bbf7d0"),
            "starting": ("開始中", "#082f49", "#38bdf8", "#bae6fd"),
            "stopped": ("停止中", "#111827", "#475569", "#cbd5e1"),
            "error": ("エラー", "#450a0a", "#ef4444", "#fecaca"),
        }
        text, bg, border, color = labels.get(state, (state, "#111827", "#475569", "#cbd5e1"))
        self.codex_watch_state_label.setText(f"Codex Watch: {text}")
        self.codex_watch_state_label.setStyleSheet(
            f"QLabel {{ padding: 8px 12px; border: 1px solid {border}; border-radius: 6px; "
            f"background: {bg}; color: {color}; font-weight: 700; }}"
        )

    @pyqtSlot(str, dict)
    def handle_event(self, event_name: str, payload: dict):
        if self._ui_updates_paused:
            return
        timestamp = time.strftime("%H:%M:%S")
        if event_name == "__timeout__":
            return
        if event_name == "presentation_buffering_progress":
            self.apply_buffering_progress_event(payload)
            now = time.monotonic()
            if now - self._last_progress_log_at < 0.5:
                return
            self._last_progress_log_at = now
        elif event_name == "presentation_buffering_completed":
            self.apply_buffering_completed_event(payload)
        if event_name == "__log__":
            log_message = f"[{timestamp}] [SYSTEM] {payload.get('message')}"
        else:
            log_message = f"[{timestamp}] [{event_name}] {json.dumps(payload)}"
        self.event_log_deque.append(log_message)
        self.present_event_log.setPlainText("\n".join(self.event_log_deque))
        self.present_event_log.verticalScrollBar().setValue(self.present_event_log.verticalScrollBar().maximum())
        if event_name.startswith("codex_watch."):
            if event_name == "codex_watch.started":
                self.set_codex_watch_state("running")
            elif event_name == "codex_watch.stopped":
                self.set_codex_watch_state("stopped")
            elif event_name == "codex_watch.error":
                self.set_codex_watch_state("error")
                self.codex_error_log.setPlainText(str(payload.get("error") or ""))
            elif event_name == "codex_watch.codex_message":
                self.codex_last_message.setPlainText(str(payload.get("text") or ""))
            elif event_name == "codex_watch.file_changed":
                self.codex_last_message.setPlaceholderText(
                    "Watcher saw file changes: "
                    f"{payload.get('lines_read', 0)} lines, "
                    f"{payload.get('events_emitted', 0)} message events"
                )
            elif event_name in {"codex_watch.rewrite_completed", "codex_watch.speech_queued"}:
                utterance = str(payload.get("utterance") or "")
                if utterance:
                    self.codex_last_utterance.setPlainText(utterance)
                    current_history = self.codex_utterance_history.toPlainText().splitlines()
                    current_history.append(utterance)
                    self.codex_utterance_history.setPlainText("\n".join(current_history[-10:]))
            current = self.codex_watch_status.toPlainText()
            lines = ([current] if current else []) + [log_message]
            self.codex_watch_status.setPlainText("\n".join("\n".join(lines).splitlines()[-120:]))
            self.codex_watch_status.verticalScrollBar().setValue(self.codex_watch_status.verticalScrollBar().maximum())
        if event_name.startswith("presentation_") and event_name != "presentation_buffering_progress":
            self.refresh_present_sessions()

    def apply_buffering_completed_event(self, payload: Dict[str, Any]) -> None:
        if self._ui_updates_paused:
            return
        session_id = payload.get("session", "?")
        slides_done = payload.get("slides_done", payload.get("slides_total", "?"))
        slides_total = payload.get("slides_total", "?")
        segments_done = payload.get("segments_done", "?")
        segments_total = payload.get("segments_total", "?")
        steps_done = payload.get("generation_steps_done", "?")
        steps_total = payload.get("generation_steps_total", "?")
        duration = payload.get("duration")
        self.buffer_progress_bar.setValue(100)
        self.play_buffered_btn.setEnabled(True)
        lines = [
            f"**Session:** {session_id} (ready)",
            f"**Slides:** {slides_done}/{slides_total}",
            (
                f"**All Buffered:** ready "
                f"(segments {segments_done}/{segments_total}, steps {steps_done}/{steps_total}, 100.0%)"
            ),
            "**Ready:** Press Play Buffered Presentation to start playback.",
        ]
        if duration is not None:
            lines.append(f"**Buffered audio duration:** {float(duration):.1f}s")
        self.present_status.setPlainText("\n".join(lines))

    def apply_buffering_progress_event(self, payload: Dict[str, Any]) -> None:
        if self._ui_updates_paused:
            return
        progress = float(payload.get("progress") or 0.0) * 100.0
        self.buffer_progress_bar.setValue(int(max(0.0, min(100.0, progress))))
        title = payload.get("title") or "?"
        stage = payload.get("stage") or "?"
        text = str(payload.get("text") or "").replace("\n", " ")
        lines = [
            f"**Session:** {payload.get('session', '?')} (buffering)",
            f"**Slides:** {payload.get('slide', '?')}/{payload.get('slides_total', '?')}",
            (
                f"**All Buffered:** buffering "
                f"(segments {payload.get('segments_done', 0)}/{payload.get('segments_total', '?')}, "
                f"steps {payload.get('generation_steps_done', 0)}/{payload.get('generation_steps_total', '?')}, "
                f"{progress:.1f}%)"
            ),
            f"**Slide title:** {title}",
            f"**Buffering stage:** segment {payload.get('current_segment', '?')} / {payload.get('segments_total', '?')} {stage}",
        ]
        if text:
            lines.append(f"**Buffering text:** {text[:160]}")
        self.present_status.setPlainText("\n".join(lines))

    @pyqtSlot()
    def refresh_present_sessions(self):
        if self._ui_updates_paused:
            return
        if self._present_refresh_inflight:
            self._present_refresh_pending = True
            return
        self._present_refresh_inflight = True
        worker = Worker(list_present_sessions, DEFAULT_CONTROLLER_URL)
        worker.signals.result.connect(self.update_present_sessions_display)
        worker.signals.finished.connect(self._finish_present_refresh)
        self.threadpool.start(worker)

    def _finish_present_refresh(self):
        self._present_refresh_inflight = False
        if self._present_refresh_pending and not self._ui_updates_paused:
            self._present_refresh_pending = False
            self.refresh_present_sessions()

    def update_present_sessions_display(self, sessions: List[Dict[str, Any]]):
        if self._ui_updates_paused:
            return
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
        self.play_buffered_btn.setEnabled(False)
        self.update_present_status_display()

    def update_present_status_display(self):
        if self._ui_updates_paused:
            return
        session_id = self.present_session_dropdown.currentText()
        if not session_id:
            self.present_status.clear()
            return
        if self._present_status_inflight:
            self._present_status_pending = True
            return
        self._present_status_inflight = True
        worker = Worker(get_session_status, DEFAULT_CONTROLLER_URL, session_id)
        worker.signals.result.connect(self.format_and_display_status)
        worker.signals.finished.connect(self._finish_present_status)
        self.threadpool.start(worker)

    def _finish_present_status(self):
        self._present_status_inflight = False
        if self._present_status_pending and not self._ui_updates_paused:
            self._present_status_pending = False
            self.update_present_status_display()

    def format_and_display_status(self, detail: Dict[str, Any]):
        if self._ui_updates_paused:
            return
        if not isinstance(detail, dict) or "error" in detail:
            self.present_status.setPlainText(json.dumps(detail, indent=2))
            return
        session = detail.get("session", {}) or {}
        status_info = detail.get("status", {}) or {}
        metadata = session.get("metadata", {}) or {}
        lines = [f"**Session:** {session.get('id', '?')} ({session.get('status', 'unknown')})"]
        slides_total = status_info.get("slides_total") or metadata.get("slides_total")
        current_slide = status_info.get("current_slide") or 0
        lines.append(f"**Slides:** {current_slide}/{slides_total or '?'}")
        if status_info.get("loop") or metadata.get("loop"):
            loop_count = status_info.get("loop_count") or 1
            loop_delay = status_info.get("loop_delay", metadata.get("loop_delay", 0))
            lines.append(f"**Loop:** On (cycle {loop_count}, delay {loop_delay}s)")
        else:
            lines.append("**Loop:** Off")
        buffering = status_info.get("buffering", {}) or {}
        if buffering.get("enabled") or status_info.get("all_buffered") or metadata.get("all_buffered"):
            buffer_status = buffering.get("status") or status_info.get("buffer_status") or metadata.get("buffer_status") or "unknown"
            buffer_done = buffering.get("slides_done", status_info.get("buffer_done", metadata.get("buffer_done", 0)))
            buffer_total = buffering.get("slides_total", status_info.get("buffer_total", metadata.get("buffer_total", slides_total or 0)))
            segments_done = buffering.get("segments_done", status_info.get("segments_done", metadata.get("segments_done", 0)))
            segments_total = buffering.get("segments_total", status_info.get("segments_total", metadata.get("segments_total", 0)))
            steps_done = buffering.get("generation_steps_done", status_info.get("generation_steps_done", metadata.get("generation_steps_done", 0)))
            steps_total = buffering.get("generation_steps_total", status_info.get("generation_steps_total", metadata.get("generation_steps_total", 0)))
            stage = buffering.get("stage") or status_info.get("generation_stage") or metadata.get("generation_stage")
            current_segment = buffering.get("current_segment") or status_info.get("current_segment") or metadata.get("current_segment")
            current_text = buffering.get("current_segment_text") or status_info.get("current_segment_text") or metadata.get("current_segment_text")
            buffer_progress = float(buffering.get("progress", status_info.get("buffer_progress") or 0.0)) * 100.0
            self.buffer_progress_bar.setValue(int(max(0.0, min(100.0, buffer_progress))))
            self.play_buffered_btn.setEnabled(buffer_status in {"ready", "completed"})
            lines.append(
                f"**All Buffered:** {buffer_status} "
                f"(slides {buffer_done}/{buffer_total}, segments {segments_done}/{segments_total}, "
                f"steps {steps_done}/{steps_total}, {buffer_progress:.1f}%)"
            )
            if buffer_status in {"ready", "completed"}:
                lines.append("**Ready:** Press Play Buffered Presentation to start playback.")
            if stage:
                lines.append(f"**Buffering stage:** segment {current_segment or '?'} / {segments_total or '?'} {stage}")
            if current_text:
                preview = str(current_text).replace("\n", " ")
                lines.append(f"**Buffering text:** {preview[:160]}")
        else:
            self.buffer_progress_bar.setValue(0)
            self.play_buffered_btn.setEnabled(False)
        title = status_info.get("current_title")
        if title:
            lines.append(f"**Slide title:** {title}")
        script_name = Path(metadata.get("script_path", "")).name if metadata.get("script_path") else "?"
        lines.append(f"**Script:** {script_name}")
        ppt_name = Path(metadata.get("ppt_path", "")).name if metadata.get("ppt_path") else "?"
        lines.append(f"**PowerPoint:** {ppt_name}")
        current_source = status_info.get("jaw_open_current_source")
        current_value = status_info.get("jaw_open_current_value")
        if current_source is not None or current_value is not None:
            source_text = f"{float(current_source):.4f}" if current_source is not None else "?"
            value_text = f"{float(current_value):.4f}" if current_value is not None else "?"
            lines.append(f"**jawOpen current:** source {source_text} -> value {value_text}")
        if status_info.get("paused") is not None:
            lines.append(f"**Paused:** {'Yes' if status_info.get('paused') else 'No'}")
        if status_info.get("queued") is not None:
            lines.append(f"**Queued motions:** {status_info.get('queued')}")
        self.present_status.setPlainText("\n".join(lines))

    def refresh_codex_sessions(self):
        if self._codex_refresh_inflight:
            return
        self._codex_refresh_inflight = True
        self.codex_session_list_status.setText("Session list: refreshing...")
        worker = Worker(list_codex_sessions, DEFAULT_CONTROLLER_URL)
        worker.signals.result.connect(self.populate_codex_sessions)
        worker.signals.error.connect(lambda err: self.codex_session_list_status.setText(f"Session list: error: {err}"))
        worker.signals.finished.connect(lambda: setattr(self, "_codex_refresh_inflight", False))
        self.threadpool.start(worker)

    def auto_refresh_codex_sessions(self):
        if self._ui_updates_paused:
            return
        if not self.auto_refresh_codex_sessions_checkbox.isChecked():
            return
        self.refresh_codex_sessions()

    def populate_codex_sessions(self, sessions: List[Dict[str, Any]]):
        self.codex_sessions = {str(item.get("id")): item for item in sessions if item.get("id")}
        current = self.codex_session_dropdown.currentData()
        self.codex_session_dropdown.blockSignals(True)
        self.codex_session_dropdown.clear()
        for item in sessions:
            session_id = item.get("id")
            if not session_id:
                continue
            cwd = Path(item.get("cwd") or "").name if item.get("cwd") else "?"
            label = f"{item.get('thread_name') or session_id} | {cwd} | {item.get('updated_at') or ''}"
            self.codex_session_dropdown.addItem(label, str(session_id))
        self.codex_session_dropdown.blockSignals(False)
        if current and current in self.codex_sessions:
            idx = self.codex_session_dropdown.findData(current)
            if idx >= 0:
                self.codex_session_dropdown.setCurrentIndex(idx)
        session_ids = list(self.codex_sessions)
        self.codex_session_list_status.setText(
            f"Session list: {len(session_ids)} sessions loaded at {time.strftime('%H:%M:%S')}"
        )
        self.update_codex_session_details()

    def update_codex_session_details(self):
        session_id = self.codex_session_dropdown.currentData()
        item = self.codex_sessions.get(str(session_id), {}) if session_id else {}
        if not item:
            self.codex_session_details.setPlainText("No Codex session selected.")
            return
        lines = [
            f"session id: {item.get('id', session_id)}",
            f"thread name: {item.get('thread_name', '?')}",
            f"cwd: {item.get('cwd') or '?'}",
            f"updated_at: {item.get('updated_at') or '?'}",
            f"rollout path: {item.get('rollout_path') or '?'}",
        ]
        self.codex_session_details.setPlainText("\n".join(lines))

    def submit_codex_watch(self):
        session_id = self.codex_session_dropdown.currentData()
        if not session_id:
            self.codex_watch_status.setPlainText("Select a Codex session first.")
            return
        item = self.codex_sessions.get(str(session_id), {})
        event_types = ["assistant_message"] if self.codex_assistant_messages_checkbox.isChecked() else []
        payload = {
            "mode": "codex_watch",
            "payload": {
                "session_id": str(session_id),
                "rollout_path": item.get("rollout_path"),
                "character_profile": self.codex_profile.toPlainText(),
                "style_id": self.get_selected_style_id(),
                "model": self.codex_model_edit.text() or "gpt-5.4-nano",
                "event_types": event_types or ["assistant_message"],
            },
        }
        worker = Worker(create_session, DEFAULT_CONTROLLER_URL, payload)
        worker.signals.result.connect(self.codex_watch_started)
        worker.signals.error.connect(lambda err: (self.set_codex_watch_state("error"), self.codex_watch_status.setPlainText(f"Error: {err}")))
        self.set_codex_watch_state("starting")
        self.threadpool.start(worker)

    def codex_watch_started(self, result: Dict[str, Any]):
        if isinstance(result, dict):
            self.codex_watch_session_id = result.get("session", {}).get("id")
            session_status = result.get("session", {}).get("status") or result.get("result", {}).get("status") or "unknown"
            self.set_codex_watch_state(str(session_status))
            self.codex_watch_status.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            self.set_codex_watch_state("unknown")
            self.codex_watch_status.setPlainText(str(result))

    def stop_codex_watch(self):
        if not self.codex_watch_session_id:
            self.codex_watch_status.setPlainText("No Codex Watch session is running.")
            return
        worker = Worker(send_command, DEFAULT_CONTROLLER_URL, self.codex_watch_session_id, "stop")
        worker.signals.result.connect(self.codex_watch_stopped)
        worker.signals.error.connect(lambda err: (self.set_codex_watch_state("error"), self.codex_watch_status.setPlainText(f"Error: {err}")))
        self.threadpool.start(worker)

    def codex_watch_stopped(self, result: str):
        self.set_codex_watch_state("stopped")
        self.codex_watch_status.setPlainText(str(result))

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
                "image_viewer": self.image_viewer_mode.currentText(),
                "loop": self.loop_checkbox.isChecked(),
                "loop_delay": self.loop_delay_slider.value(),
                "all_buffered": self.all_buffered_checkbox.isChecked(),
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
        worker.signals.result.connect(self.present_command_applied)
        worker.signals.error.connect(lambda err: self.command_feedback.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def present_command_applied(self, result: str):
        self.command_feedback.setText(str(result))
        self.refresh_present_sessions()

    @pyqtSlot()
    def start_facetracker(self):
        """Starts a new facetracker session."""
        if self.facetracker_session_id:
            self.facetracker_status_label.setText("Status: A session is already active. Stop it first.")
            return

        payload = {
            "mode": "facetracker",
            "payload": {
                "camera_index": self.camera_index_spinbox.value(),
                "osc_host": self.osc_host_edit.text(),
                "osc_port": self.osc_port_spinbox.value(),
                "flip_camera": self.flip_camera_checkbox.isChecked(),
            },
        }
        
        worker = Worker(create_session, DEFAULT_CONTROLLER_URL, payload)
        worker.signals.result.connect(self.handle_facetracker_start_response)
        worker.signals.error.connect(lambda err: self.facetracker_status_label.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def handle_facetracker_start_response(self, result: dict):
        if "error" in result:
            self.facetracker_status_label.setText(f"Error: {result['error']}")
            return
        
        session_id = result.get("session", {}).get("id")
        if session_id:
            self.facetracker_session_id = session_id
            self.facetracker_status_label.setText(f"Status: Running (Session: {session_id[:8]})")
        else:
            self.facetracker_status_label.setText("Error: Could not get session ID from response.")

    @pyqtSlot()
    def stop_facetracker(self):
        """Stops the current facetracker session."""
        if not self.facetracker_session_id:
            self.facetracker_status_label.setText("Status: No active session to stop.")
            return

        worker = Worker(send_command, DEFAULT_CONTROLLER_URL, self.facetracker_session_id, "stop")
        worker.signals.result.connect(self.handle_facetracker_stop_response)
        worker.signals.error.connect(lambda err: self.facetracker_status_label.setText(f"Error: {err}"))
        self.threadpool.start(worker)

    def handle_facetracker_stop_response(self, result: str):
        self.facetracker_status_label.setText("Status: Not running")
        self.facetracker_session_id = None

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
