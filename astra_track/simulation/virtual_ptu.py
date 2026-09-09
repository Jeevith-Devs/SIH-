"""
Kinematic Pan-Tilt Unit (PTU) simulation.
Models physical motor velocity limits (5-10 deg/s), acceleration slew rates, and hard mechanical stops.
"""

import math
from typing import Tuple
from astra_track.app.config import CameraConfig


class VirtualPTU:
    """Simulated high-precision optical gimbal / pan-tilt mechanism."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.pan_deg: float = 0.0
        self.tilt_deg: float = 0.0
        
        self.pan_vel_deg_s: float = 0.0
        self.tilt_vel_deg_s: float = 0.0

        self.max_speed_pan = config.max_pan_speed
        self.max_speed_tilt = config.max_tilt_speed
        self.max_accel_pan = config.max_pan_accel
        self.max_accel_tilt = config.max_tilt_accel

        self.pan_limit = config.pan_limit
        self.tilt_limit = config.tilt_limit

    def reset(self, pan: float = 0.0, tilt: float = 0.0):
        self.pan_deg = pan
        self.tilt_deg = tilt
        self.pan_vel_deg_s = 0.0
        self.tilt_vel_deg_s = 0.0

    def set_pan_tilt(self, pan: float, tilt: float):
        """Directly sets pan/tilt angles (hardware interface override)."""
        self.pan_deg = max(-self.pan_limit, min(self.pan_limit, pan))
        self.tilt_deg = max(-self.tilt_limit, min(self.tilt_limit, tilt))

    def step_velocity(self, target_pan_rate: float, target_tilt_rate: float, dt: float) -> Tuple[float, float]:
        """
        Applies rate commands to the PTU subject to physical acceleration and velocity limits.
        """
        # Clamp commanded rates to max motor speed
        target_pan_rate = max(-self.max_speed_pan, min(self.max_speed_pan, target_pan_rate))
        target_tilt_rate = max(-self.max_speed_tilt, min(self.max_speed_tilt, target_tilt_rate))

        # Acceleration slew limiting for Pan
        pan_accel_req = (target_pan_rate - self.pan_vel_deg_s) / max(1e-4, dt)
        pan_accel_clamped = max(-self.max_accel_pan, min(self.max_accel_pan, pan_accel_req))
        self.pan_vel_deg_s += pan_accel_clamped * dt

        # Acceleration slew limiting for Tilt
        tilt_accel_req = (target_tilt_rate - self.tilt_vel_deg_s) / max(1e-4, dt)
        tilt_accel_clamped = max(-self.max_accel_tilt, min(self.max_accel_tilt, tilt_accel_req))
        self.tilt_vel_deg_s += tilt_accel_clamped * dt

        # Integrate positions
        new_pan = self.pan_deg + self.pan_vel_deg_s * dt
        new_tilt = self.tilt_deg + self.tilt_vel_deg_s * dt

        # Enforce mechanical hard limits
        if new_pan > self.pan_limit:
            self.pan_deg = self.pan_limit
            self.pan_vel_deg_s = 0.0
        elif new_pan < -self.pan_limit:
            self.pan_deg = -self.pan_limit
            self.pan_vel_deg_s = 0.0
        else:
            self.pan_deg = new_pan

        if new_tilt > self.tilt_limit:
            self.tilt_deg = self.tilt_limit
            self.tilt_vel_deg_s = 0.0
        elif new_tilt < -self.tilt_limit:
            self.tilt_deg = -self.tilt_limit
            self.tilt_vel_deg_s = 0.0
        else:
            self.tilt_deg = new_tilt

        return self.pan_deg, self.tilt_deg

    def get_angles(self) -> Tuple[float, float]:
        return self.pan_deg, self.tilt_deg

    def get_rates(self) -> Tuple[float, float]:
        return self.pan_vel_deg_s, self.tilt_vel_deg_s
