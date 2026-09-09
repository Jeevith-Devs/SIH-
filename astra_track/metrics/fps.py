"""
Real-time FPS and latency profiler.
"""

from typing import List, Dict, Any
import numpy as np


class FPSProfiler:
    """Tracks frame processing throughput and latency."""

    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.fps_history: List[float] = []
        self.latency_ms_history: List[float] = []

    def reset(self):
        self.fps_history.clear()
        self.latency_ms_history.clear()

    def record(self, fps: float, latency_ms: float):
        self.fps_history.append(fps)
        self.latency_ms_history.append(latency_ms)

    def get_summary(self) -> Dict[str, float]:
        if not self.fps_history:
            return {
                "avg_fps": self.target_fps,
                "min_fps": self.target_fps,
                "max_fps": self.target_fps,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0
            }

        fps_arr = np.array(self.fps_history, dtype=np.float64)
        lat_arr = np.array(self.latency_ms_history, dtype=np.float64)

        return {
            "avg_fps": float(np.mean(fps_arr)),
            "min_fps": float(np.min(fps_arr)),
            "max_fps": float(np.max(fps_arr)),
            "avg_latency_ms": float(np.mean(lat_arr)),
            "max_latency_ms": float(np.max(lat_arr))
        }
