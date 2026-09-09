"""
Kinematics analysis: velocity, acceleration, heading angle, and angular rate estimators.
"""

import math
from typing import Tuple


class KinematicsEstimator:
    """Computes derived speed, acceleration, and heading telemetry."""

    @staticmethod
    def compute_motion_vector(vx: float, vy: float, ax: float, ay: float) -> Tuple[float, float, float]:
        """
        Returns (speed_px_s, heading_deg, accel_px_s2).
        Heading: 0 deg = moving right (+x), 90 deg = moving down (+y).
        """
        speed = math.hypot(vx, vy)
        heading_rad = math.atan2(vy, vx)
        heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
        accel = math.hypot(ax, ay)
        return speed, heading_deg, accel
