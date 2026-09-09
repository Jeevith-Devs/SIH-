"""
Predictive feedforward & lead-position camera controller.
Controls camera toward predicted beacon location to eliminate dynamic lag.
"""

from typing import Tuple, Optional
from astra_track.app.constants import ControlMode
from astra_track.app.config import ControllerConfig, CameraConfig
from astra_track.core.frame import ControlCommand, PredictionResult, KalmanState
from astra_track.control.controller import BaseController


class PredictiveController(BaseController):
    """Predictive feedforward coarse alignment controller for high-speed dynamic beacons."""

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
        """Standard fallback interface if prediction not provided."""
        err_px_x = target_x - center_x
        err_px_y = target_y - center_y
        pan_err_deg = err_px_x * self.deg_per_px_x
        tilt_err_deg = err_px_y * self.deg_per_px_y
        
        pan_rate = max(-self.max_speed_pan, min(self.max_speed_pan, self.kp_pan * pan_err_deg))
        tilt_rate = max(-self.max_speed_tilt, min(self.max_speed_tilt, self.kp_tilt * tilt_err_deg))

        return ControlCommand(
            pan_error_deg=pan_err_deg,
            tilt_error_deg=tilt_err_deg,
            pan_rate_cmd_deg_s=pan_rate,
            tilt_rate_cmd_deg_s=tilt_rate,
            mode=ControlMode.PREDICTIVE.value
        )

    def compute_predictive_command(
        self,
        prediction: PredictionResult,
        k_state: KalmanState,
        center_x: float = 320.0,
        center_y: float = 240.0,
        dt: float = 0.0333
    ) -> ControlCommand:
        """
        Uses forward-projected beacon position and velocity feedforward.
        """
        lead_x = prediction.lead_target_x
        lead_y = prediction.lead_target_y

        # Clamp lead target within plausible sensor acquisition range
        margin = 60.0
        clamped_lead_x = max(-margin, min(float(self.cam_config.width) + margin, lead_x))
        clamped_lead_y = max(-margin, min(float(self.cam_config.height) + margin, lead_y))

        # Error between future predicted target and camera boresight
        lead_err_x = clamped_lead_x - center_x
        lead_err_y = clamped_lead_y - center_y

        pan_err_deg = lead_err_x * self.deg_per_px_x
        tilt_err_deg = lead_err_y * self.deg_per_px_y

        # Deadband check
        eff_pan_err = 0.0 if abs(pan_err_deg) < self.deadzone else pan_err_deg
        eff_tilt_err = 0.0 if abs(tilt_err_deg) < self.deadzone else tilt_err_deg

        # Kinematic Feedforward Rate (deg/s) bounded to physical satellite orbital envelope
        max_ff_vel = 180.0 # px/s (~1.1 deg/s max orbital drift across FOV)
        clamped_vx = max(-max_ff_vel, min(max_ff_vel, k_state.vx))
        clamped_vy = max(-max_ff_vel, min(max_ff_vel, k_state.vy))
        ff_rate_pan = clamped_vx * self.deg_per_px_x
        ff_rate_tilt = clamped_vy * self.deg_per_px_y

        # Composite command: Feedback on lead position + Feedforward velocity
        pan_rate = 0.60 * ff_rate_pan + self.kp_pan * eff_pan_err
        tilt_rate = 0.60 * ff_rate_tilt + self.kp_tilt * eff_tilt_err

        # Clamp to physical motor speed limits
        pan_rate_clamped = max(-self.max_speed_pan, min(self.max_speed_pan, pan_rate))
        tilt_rate_clamped = max(-self.max_speed_tilt, min(self.max_speed_tilt, tilt_rate))

        return ControlCommand(
            pan_error_deg=float(pan_err_deg),
            tilt_error_deg=float(tilt_err_deg),
            pan_rate_cmd_deg_s=float(pan_rate_clamped),
            tilt_rate_cmd_deg_s=float(tilt_rate_clamped),
            mode=ControlMode.PREDICTIVE.value
        )
