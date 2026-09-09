"""
Dynamic platform disturbances: camera mounting jitter and vehicle platform motion.
"""

import math
from typing import Tuple
import numpy as np
from astra_track.app.constants import DisturbancePattern
from astra_track.app.config import DisturbanceConfig


class DisturbanceSimulator:
    """Simulates platform vibrations, mechanical jitter, and vehicle drift."""

    def __init__(self, config: DisturbanceConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.platform_x = 0.0
        self.platform_y = 0.0

    def reset(self):
        self.platform_x = 0.0
        self.platform_y = 0.0

    def get_jitter(self, t: float) -> Tuple[float, float]:
        """
        High-frequency camera mounting jitter (up to +/-20 px).
        """
        amp = max(0.0, min(20.0, self.config.camera_jitter_amp))
        if amp <= 0.001:
            return 0.0, 0.0

        pattern = self.config.camera_jitter_pattern

        if pattern == DisturbancePattern.RANDOM.value:
            jx = float(self.rng.normal(0.0, amp * 0.5))
            jy = float(self.rng.normal(0.0, amp * 0.5))
        elif pattern == DisturbancePattern.SINUSOIDAL.value:
            # 12 Hz mechanical resonance
            omega = 2.0 * math.pi * 12.0
            jx = amp * math.sin(omega * t)
            jy = amp * math.cos(omega * t * 0.8)
        elif pattern == DisturbancePattern.CIRCULAR.value:
            omega = 2.0 * math.pi * 8.0
            jx = amp * math.cos(omega * t)
            jy = amp * math.sin(omega * t)
        else: # LINEAR
            jx = amp * math.sin(2.0 * math.pi * 10.0 * t)
            jy = 0.0

        return max(-20.0, min(20.0, jx)), max(-20.0, min(20.0, jy))

    def step_platform(self, dt: float, t: float) -> Tuple[float, float]:
        """
        Low-frequency terminal platform motion/drift (up to +/-20 px/frame).
        """
        amp = max(0.0, min(20.0, self.config.platform_motion_amp))
        if amp <= 0.001:
            return 0.0, 0.0

        pattern = self.config.platform_motion_pattern
        speed = self.config.platform_speed

        if pattern == DisturbancePattern.LINEAR.value:
            # Drift along direction
            self.platform_x += speed * dt * 0.5
            self.platform_y += speed * dt * 0.3
            # Bound within [-amp*5, amp*5]
            bound = amp * 8.0
            if abs(self.platform_x) > bound:
                self.platform_x = math.copysign(bound, self.platform_x)
            if abs(self.platform_y) > bound:
                self.platform_y = math.copysign(bound, self.platform_y)
            dx = math.sin(t * 0.8) * amp
            dy = math.cos(t * 0.5) * amp
        elif pattern == DisturbancePattern.CIRCULAR.value:
            omega = 0.5 # 0.5 rad/s slow roll
            dx = amp * math.cos(omega * t)
            dy = amp * math.sin(omega * t)
        elif pattern == DisturbancePattern.SINUSOIDAL.value:
            dx = amp * math.sin(1.2 * t)
            dy = amp * math.sin(0.7 * t)
        else: # RANDOM
            dx = float(self.rng.normal(0.0, amp * 0.4))
            dy = float(self.rng.normal(0.0, amp * 0.4))

        return max(-20.0, min(20.0, dx)), max(-20.0, min(20.0, dy))
