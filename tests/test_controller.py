"""
Unit tests for pan/tilt controllers and speed limiting.
"""

from astra_track.app.config import ControllerConfig, CameraConfig
from astra_track.control.pan_tilt_controller import PanTiltController
from astra_track.simulation.virtual_ptu import VirtualPTU


def test_pan_speed_limit():
    cam_cfg = CameraConfig(max_pan_speed=5.0, max_tilt_speed=5.0)
    ctrl_cfg = ControllerConfig(kp_pan=10.0, kp_tilt=10.0)
    controller = PanTiltController(ctrl_cfg, cam_cfg)

    # Large target error (far right: 600px vs center 320px)
    cmd = controller.compute_command(600.0, 240.0, center_x=320.0, center_y=240.0)

    # Commanded speed must be clamped to max_pan_speed (5.0 deg/s)
    assert abs(cmd.pan_rate_cmd_deg_s - 5.0) < 1e-4
    assert cmd.tilt_rate_cmd_deg_s == 0.0


def test_tilt_speed_limit():
    cam_cfg = CameraConfig(max_pan_speed=5.0, max_tilt_speed=6.0)
    ctrl_cfg = ControllerConfig(kp_pan=5.0, kp_tilt=5.0)
    controller = PanTiltController(ctrl_cfg, cam_cfg)

    # Large vertical error (bottom: 450px vs center 240px)
    cmd = controller.compute_command(320.0, 450.0, center_x=320.0, center_y=240.0)
    assert abs(cmd.tilt_rate_cmd_deg_s - 6.0) < 1e-4


def test_ptu_acceleration_limits():
    cam_cfg = CameraConfig(max_pan_speed=5.0, max_pan_accel=10.0)
    ptu = VirtualPTU(cam_cfg)
    
    # Step from 0 to 5 deg/s in 0.1s -> max possible delta is 10.0 * 0.1 = 1.0 deg/s
    ptu.step_velocity(5.0, 0.0, dt=0.1)
    rates = ptu.get_rates()
    assert abs(rates[0] - 1.0) < 1e-3
