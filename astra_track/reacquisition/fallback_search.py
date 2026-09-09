"""
Fallback search patterns: expanding Archimedean spiral and raster sweep for wide optical reacquisition.
"""

import math
from typing import Tuple


class FallbackSearchScanner:
    """Generates search offset commands to steer camera when target escapes predicted ROI."""

    def __init__(self, max_radius_deg: float = 3.5, expansion_rate: float = 0.5):
        self.max_radius_deg = max_radius_deg
        self.expansion_rate = expansion_rate
        self.step_count = 0
        self.spiral_theta = 0.0

    def reset(self):
        self.step_count = 0
        self.spiral_theta = 0.0

    def get_spiral_offset(self, dt: float) -> Tuple[float, float]:
        """
        Generates expanding Archimedean spiral offsets (d_pan_deg, d_tilt_deg).
        r = a * theta
        """
        self.step_count += 1
        # Angular advance per step
        self.spiral_theta += 18.0 * dt
        r = min(self.max_radius_deg, self.expansion_rate * self.spiral_theta)

        d_pan = r * math.cos(self.spiral_theta)
        d_tilt = r * math.sin(self.spiral_theta)

        return float(d_pan), float(d_tilt)

    def get_raster_offset(self, t: float) -> Tuple[float, float]:
        """
        Wide horizontal raster sweep.
        """
        sweep_freq = 0.8 # Hz
        sweep_amp = 3.0 # degrees
        d_pan = sweep_amp * math.sin(2.0 * math.pi * sweep_freq * t)
        d_tilt = (math.sin(0.2 * t)) * 1.5
        return float(d_pan), float(d_tilt)
