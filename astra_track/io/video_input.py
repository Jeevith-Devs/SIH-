"""
Mode B: Video file input reader for benchmark evaluation (.mp4, .avi, .mov).
Supports frame stepping, scrubbing, playback speed control, and mathematical PTU tracking.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class VideoInputReader:
    """Processes recorded video files frame-by-frame with full transport controls."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.file_path: Optional[str] = None
        self.is_opened = False
        
        self.total_frames = 0
        self.fps = 30.0
        self.width = 640
        self.height = 480
        self.duration_sec = 0.0

        self.current_frame_idx = 0
        self.playback_speed = 1.0
        self.is_paused = True

        # Simulated PTU state for video mode
        self.sim_pan = 0.0
        self.sim_tilt = 0.0

    def open(self, file_path: str) -> bool:
        self.close()
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.is_opened = False
            return False

        self.file_path = file_path
        self.is_opened = True
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = max(1.0, float(self.cap.get(cv2.CAP_PROP_FPS)))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0

        self.current_frame_idx = 0
        self.is_paused = True
        self.sim_pan = 0.0
        self.sim_tilt = 0.0
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Reads next frame. Returns (success, frame_gray, frame_index)."""
        if not self.is_opened or not self.cap:
            return False, None, self.current_frame_idx

        ret, frame = self.cap.read()
        if not ret:
            return False, None, self.current_frame_idx

        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        return True, gray, self.current_frame_idx

    def seek(self, frame_index: int) -> bool:
        if not self.is_opened or not self.cap:
            return False
        idx = max(0, min(self.total_frames - 1, frame_index))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.current_frame_idx = idx
        return True

    def step_forward(self) -> Tuple[bool, Optional[np.ndarray], int]:
        return self.read_frame()

    def step_backward(self) -> Tuple[bool, Optional[np.ndarray], int]:
        target = max(0, self.current_frame_idx - 2)
        self.seek(target)
        return self.read_frame()

    def restart(self):
        self.seek(0)
        self.sim_pan = 0.0
        self.sim_tilt = 0.0

    def update_simulated_ptu(self, target_x: float, target_y: float, dt: float) -> Tuple[float, float]:
        """Simulates mathematical PTU response when tracking real video input."""
        err_x = target_x - (self.width / 2.0)
        err_y = target_y - (self.height / 2.0)
        
        # 4 deg HFOV, 3 deg VFOV
        deg_per_px_x = 4.0 / self.width
        deg_per_px_y = 3.0 / self.height

        d_pan = err_x * deg_per_px_x
        d_tilt = err_y * deg_per_px_y

        # Rate command clamped to 5 deg/s
        max_rate = 5.0
        pan_rate = max(-max_rate, min(max_rate, 1.2 * d_pan))
        tilt_rate = max(-max_rate, min(max_rate, 1.2 * d_tilt))

        self.sim_pan += pan_rate * dt
        self.sim_tilt += tilt_rate * dt

        return self.sim_pan, self.sim_tilt

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.is_opened = False
