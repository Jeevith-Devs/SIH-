"""
Hardware-agnostic camera abstractions.
Enables seamless replacement of the virtual optical sensor with physical hardware (GigE, USB, RTSP).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, List
import threading
import time
import cv2
import numpy as np


class CameraInterface(ABC):
    """Universal optical camera sensor interface."""

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Captures next frame. Returns None if stream ended."""
        pass

    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        pass

    @abstractmethod
    def release(self):
        """Releases sensor resources."""
        pass


class VirtualCameraAdapter(CameraInterface):
    """Adapter wrapping virtual simulation camera."""

    def __init__(self, scenario_engine):
        self.engine = scenario_engine
        self.last_frame = None

    def get_frame(self) -> Optional[np.ndarray]:
        # Stepped via pipeline
        return self.last_frame

    def set_current_frame(self, frame: np.ndarray):
        self.last_frame = frame

    def get_resolution(self) -> Tuple[int, int]:
        return self.engine.camera.width, self.engine.camera.height

    def release(self):
        pass


class RealCameraAdapter(CameraInterface):
    """Driver connecting to physical optical cameras via OpenCV VideoCapture (USB/GigE/RTSP)."""

    def __init__(self, camera_index_or_url: Any = 0, width: int = 640, height: int = 480, threaded: bool = True):
        self.device_id = camera_index_or_url
        self.width = width
        self.height = height
        self.threaded = threaded
        self.cap = None
        self.is_connected = False
        
        self._last_raw_frame = None
        self._last_gray_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
        self.connect()

    def connect(self) -> bool:
        """Connects to the video capture source."""
        try:
            # On Windows, cv2.CAP_DSHOW provides fast startup for USB webcams
            if isinstance(self.device_id, int):
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.device_id)
            else:
                self.cap = cv2.VideoCapture(self.device_id)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                # Attempt to read a test frame
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.is_connected = True
                    self._last_raw_frame = frame
                    self._last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if self.threaded:
                        self._running = True
                        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
                        self._thread.start()
                    return True
        except Exception as e:
            print(f"[RealCameraAdapter Error]: Connection failed: {e}")

        self.is_connected = False
        return False

    def _capture_worker(self):
        """Background worker thread continuously grabbing newest frames."""
        while self._running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    self._last_raw_frame = frame
                    self._last_gray_frame = gray
            else:
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns the newest grayscale frame for perception."""
        if not self.is_connected:
            return None
        if self.threaded:
            with self._lock:
                return self._last_gray_frame.copy() if self._last_gray_frame is not None else None
        else:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._last_raw_frame = frame
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None

    def get_raw_bgr_frame(self) -> Optional[np.ndarray]:
        """Returns the newest raw color BGR frame for visual display."""
        if not self.is_connected:
            return None
        if self.threaded:
            with self._lock:
                return self._last_raw_frame.copy() if self._last_raw_frame is not None else None
        return self._last_raw_frame

    def get_resolution(self) -> Tuple[int, int]:
        if self.cap and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return self.width, self.height

    @staticmethod
    def list_available_cameras(max_devices: int = 4) -> List[int]:
        """Probes system for available video capture devices."""
        available = []
        for i in range(max_devices):
            try:
                temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        available.append(i)
                    temp_cap.release()
            except Exception:
                pass
        return available

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False

