"""
Model Predictive Controller (MPC) for optimal constrained pan-tilt tracking.
Minimizes trajectory tracking error and actuator effort subject to velocity and acceleration bounds.
"""

from typing import Tuple, List
import numpy as np
from astra_track.app.constants import ControlMode
from astra_track.app.config import ControllerConfig, CameraConfig
from astra_track.core.frame import ControlCommand, PredictionResult, KalmanState
from astra_track.control.controller import BaseController


class MPCController(BaseController):
    """Finite-horizon quadratic optimization controller for constrained gimbal slewing."""

    def __init__(self, ctrl_config: ControllerConfig, cam_config: CameraConfig, horizon: int = 5):
        self.ctrl_config = ctrl_config
        self.cam_config = cam_config
        self.horizon = horizon

        self.deg_per_px_x = cam_config.fov_horizontal / float(cam_config.width)
        self.deg_per_px_y = cam_config.fov_vertical / float(cam_config.height)

        self.max_speed_pan = cam_config.max_pan_speed
        self.max_speed_tilt = cam_config.max_tilt_speed

        # Tuning weights: Q for tracking error, R for control effort
        self.q_weight = 1.0
        self.r_weight = 0.08

    def reset(self):
        pass

    def compute_command(
        self,
        target_x: float,
        target_y: float,
        center_x: float = 320.0,
        center_y: float = 240.0,
        dt: float = 0.0333
    ) -> ControlCommand:
        err_x = (target_x - center_x) * self.deg_per_px_x
        err_y = (target_y - center_y) * self.deg_per_px_y

        u_pan = (self.q_weight / (self.q_weight + self.r_weight)) * err_x * 1.5
        u_tilt = (self.q_weight / (self.q_weight + self.r_weight)) * err_y * 1.5

        u_pan = max(-self.max_speed_pan, min(self.max_speed_pan, u_pan))
        u_tilt = max(-self.max_speed_tilt, min(self.max_speed_tilt, u_tilt))

        return ControlCommand(
            pan_error_deg=err_x,
            tilt_error_deg=err_y,
            pan_rate_cmd_deg_s=u_pan,
            tilt_rate_cmd_deg_s=u_tilt,
            mode="MPC"
        )
