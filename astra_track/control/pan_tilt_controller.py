"""
Proportional coarse alignment pan/tilt controller with dead-zone and physical velocity limiting.
"""

from typing import Tuple
from astra_track.app.constants import ControlMode
from astra_track.app.config import ControllerConfig, CameraConfig
from astra_track.core.frame import ControlCommand
from astra_track.control.controller import BaseController


class PanTiltController(BaseController):
    """Reactive proportional feedback controller for optical gimbal coarse pointing."""

    def __init__(self, ctrl_config: ControllerConfig, cam_config: CameraConfig):
        self.ctrl_config = ctrl_config
        self.cam_config = cam_config

        self.deg_per_px_x = cam_config.fov_horizontal / float(cam_config.width)
        self.deg_per_px_y = cam_config.fov_vertical / float(cam_config.height)

        self.kp_pan = ctrl_config.kp_pan
        self.kp_tilt = ctrl_config.kp_tilt
        self.deadzone = ctrl_config.deadzone_deg
        self.max_speed_pan = cam_config.max_pan_speed
        self.max_speed_tilt = cam_config.max_tilt_speed

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
        """
        Translates measured target pixel offset to rate commands.
        """
        # Pixel error from optical boresight
        err_px_x = target_x - center_x
        err_px_y = target_y - center_y

        # Angular error in degrees
        pan_err_deg = err_px_x * self.deg_per_px_x
        tilt_err_deg = err_px_y * self.deg_per_px_y

        # Deadzone filtering
        effective_pan_err = 0.0 if abs(pan_err_deg) < self.deadzone else pan_err_deg
        effective_tilt_err = 0.0 if abs(tilt_err_deg) < self.deadzone else tilt_err_deg

        # Proportional velocity commands (deg/s)
        # Positive pan error (target right of center) -> positive pan command (camera pans right)
        pan_rate = self.kp_pan * effective_pan_err
        tilt_rate = self.kp_tilt * effective_tilt_err

        # Clamp to physical motor speed limits
        pan_rate_clamped = max(-self.max_speed_pan, min(self.max_speed_pan, pan_rate))
        tilt_rate_clamped = max(-self.max_speed_tilt, min(self.max_speed_tilt, tilt_rate))

        return ControlCommand(
            pan_error_deg=float(pan_err_deg),
            tilt_error_deg=float(tilt_err_deg),
            pan_rate_cmd_deg_s=float(pan_rate_clamped),
            tilt_rate_cmd_deg_s=float(tilt_rate_clamped),
            mode=ControlMode.REACTIVE.value
        )
