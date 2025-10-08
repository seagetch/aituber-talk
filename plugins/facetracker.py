"""FaceTracker mode using nijitrack."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial.transform import Rotation as R

from controller.session import Session
from plugins.base import Mode

# Helper functions adapted from nijitrack.py
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def compute_fixed_rotation_matrix(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]

    z_axis = normalize(np.cross(right_eye - left_eye, chin - left_eye))
    x_axis = normalize(right_eye - left_eye)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R_mat, landmarks[1] # Nose tip

def rotation_matrix_to_quaternion(R_mat):
    quat = R.from_matrix(R_mat).as_quat()
    return quat[0], quat[1], quat[2], quat[3]


class FaceTrackerMode(Mode):
    """Runs face tracking using nijitrack and sends data via VMC."""

    name = "facetracker"
    description = "Real-time face tracking with nijitrack."

    def __init__(self, engine: "TalkEngine", **dependencies: Any) -> None:
        super().__init__(**dependencies)
        try:
            import nijitrack
        except ImportError as e:
            raise ImportError("nijitrack is not installed. Please run setup_nijitrack script.") from e
        
        self.engine = engine
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, session: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Starts the face tracking thread."""
        if self._thread is not None and self._thread.is_alive():
            return {"status": "already_running", "session_id": session.id}

        # Disable TalkEngine idle motion
        self.engine.set_idle(False)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._tracker_loop,
            args=(session.id, payload),
            name=f"FaceTracker-{session.id[:8]}",
            daemon=True
        )
        self._thread.start()
        
        return {"status": "started", "session_id": session.id}

    def stop(self, session: Session) -> Optional[Dict[str, Any]]:
        """Stops the face tracking thread."""
        if self._thread is None or not self._thread.is_alive():
            # Re-enable idle motion just in case
            self.engine.set_idle(True)
            return {"status": "not_running"}

        self._stop_event.set()
        self._thread.join(timeout=5)
        
        # Always re-enable idle motion on stop
        self.engine.set_idle(True)

        if self._thread.is_alive():
            return {"status": "error", "detail": "Thread did not terminate in time."}
        
        self._thread = None
        return {"status": "stopped"}

    def status(self, session: Session) -> Dict[str, Any]:
        is_alive = self._thread is not None and self._thread.is_alive()
        return {
            "session_id": session.id,
            "is_running": is_alive,
        }

    def _tracker_loop(self, session_id: str, payload: Dict[str, Any]) -> None:
        """The main loop for face tracking."""
        camera_index = payload.get("camera_index", 0)
        osc_host = payload.get("osc_host", "127.0.0.1")
        osc_port = payload.get("osc_port", 39540)
        flip_camera = payload.get("flip_camera", False)

        landmarker = None
        cap = None
        try:
            osc_client = SimpleUDPClient(osc_host, osc_port)
            
            # Use a relative path from the project root
            import os
            model_path = "nijitrack/face_landmarker_v2_with_blendshapes.task"

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Could not find landmark model at {model_path}")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                running_mode=vision.RunningMode.VIDEO
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)

            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera device {camera_index}.")

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                if flip_camera:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp)

                if result and result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    R_mat, nose = compute_fixed_rotation_matrix(pts)
                    quat = rotation_matrix_to_quaternion(R_mat)
                    
                    osc_client.send_message("/VMC/Ext/Bone/Pos", [
                        "Head",
                        nose[0], nose[1], nose[2],
                        quat[0], quat[1], quat[2], quat[3]
                    ])

                    if result.face_blendshapes:
                        blendshapes = result.face_blendshapes[0]
                        for bs in blendshapes:
                            osc_client.send_message("/VMC/Ext/Blend/Val", [bs.category_name, bs.score])
                        osc_client.send_message("/VMC/Ext/Blend/Apply", [])
                
                # Small sleep to prevent busy-looping and high CPU usage
                time.sleep(0.001)

        except Exception as e:
            # In a real implementation, we would use the event bus to report this error.
            print(f"[FaceTracker-{session_id[:8]}] Error: {e}")
        finally:
            if cap:
                cap.release()
            if landmarker:
                landmarker.close()
            print(f"[FaceTracker-{session_id[:8]}] Thread finished.")