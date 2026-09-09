"""
Unit and integration tests for Physical Hardware-in-the-Loop (HIL) adapters.
Tests RealCameraAdapter, SerialServoPTUAdapter, and live frame tracking pipeline.
"""

import numpy as np
import pytest
from astra_track.io.camera_interface import RealCameraAdapter
from astra_track.io.ptu_interface import SerialServoPTUAdapter, RealPTUAdapter
from astra_track.core.pipeline import TrackingPipeline
from astra_track.app.config import AstraConfig


def test_camera_adapter_probing():
    """Verify camera probing executes without raising exceptions."""
    cams = RealCameraAdapter.list_available_cameras(max_devices=2)
    assert isinstance(cams, list)


def test_serial_ptu_port_probing():
    """Verify serial port probing executes cleanly."""
    ports = SerialServoPTUAdapter.list_available_ports()
    assert isinstance(ports, list)


def test_serial_servo_ptu_commands():
    """Test command format and safety clamping on SerialServoPTUAdapter."""
    ptu = SerialServoPTUAdapter(port="COM_NONEXISTENT", baudrate=115200)
    assert ptu.is_connected is False

    # Setting pan/tilt should clamp within safety limits
    ptu.set_pan_tilt(60.0, -45.0)
    assert ptu.pan == 45.0  # Clamped to +45 deg max
    assert ptu.tilt == -30.0 # Clamped to -30 deg min

    ptu.set_rates(5.0, -2.0, dt=0.1)
    assert abs(ptu.pan - 45.0) < 1e-3 # Clamped
    assert abs(ptu.tilt - (-30.0)) < 1e-3

    ptu.release()
    assert ptu.is_connected is False


def test_live_frame_pipeline_tracking():
    """Verify process_live_frame correctly detects and tracks optical spots."""
    config = AstraConfig.preset_easy()
    pipeline = TrackingPipeline(config)

    # Frame 0: flat background (no spot)
    dark_frame = np.zeros((480, 640), dtype=np.uint8) + 10
    proc0 = pipeline.process_live_frame(dark_frame)
    assert proc0.detection.detected is False

    # Frame 1: Introduce bright 10x10 optical beacon at (200, 150)
    beacon_frame = dark_frame.copy()
    beacon_frame[145:155, 195:205] = 250
    proc1 = pipeline.process_live_frame(beacon_frame)
    assert proc1.detection.detected is True
    assert abs(proc1.detection.x - 200.0) < 3.0
    assert abs(proc1.detection.y - 150.0) < 3.0
    assert proc1.tracking_error_px > 0.0 # Distance from center (320, 240)
