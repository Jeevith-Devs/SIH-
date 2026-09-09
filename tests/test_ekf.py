"""
Unit tests for 9-State Extended Kalman Filter (EKF) with 6-DOF IMU Sensor Fusion.
"""

import math
import numpy as np
import pytest

from astra_track.tracking.ekf_tracker import ExtendedKalmanTracker, EKFState


def test_ekf_initialization():
    """Verify spherical to Cartesian initialization and coordinate extraction."""
    tracker = ExtendedKalmanTracker(dt=0.0333)
    tracker.init_state(range_m=1000.0, azimuth_deg=30.0, elevation_deg=10.0)

    state = tracker.get_state()
    assert abs(state.range_m - 1000.0) < 1.0
    assert abs(state.azimuth_deg - 30.0) < 0.5
    assert abs(state.elevation_deg - 10.0) < 0.5
    assert state.x > 0.0
    assert state.y > 0.0
    assert state.z > 0.0


def test_ekf_prediction_kinematics():
    """Verify constant-velocity kinematic propagation across time steps."""
    tracker = ExtendedKalmanTracker(dt=0.1)
    tracker.init_state(range_m=500.0, azimuth_deg=0.0, elevation_deg=0.0)
    
    # Inject 20 m/s velocity along X
    tracker.x[3, 0] = 20.0
    initial_x = tracker.x[0, 0]

    for _ in range(10):
        tracker.predict(dt=0.1)

    state = tracker.get_state()
    expected_x = initial_x + 20.0 * 1.0  # 1.0s elapsed
    assert abs(state.x - expected_x) < 1e-4


def test_ekf_optical_bearing_convergence():
    """Verify filter converges and uncertainty shrinks under optical LOS angle measurements."""
    tracker = ExtendedKalmanTracker(dt=0.0333)
    tracker.init_state(range_m=1000.0, azimuth_deg=0.0, elevation_deg=0.0)

    initial_uncertainty = tracker.get_state().uncertainty_radius

    # True target is at 5 deg Azimuth, 3 deg Elevation
    target_az = math.radians(5.0)
    target_el = math.radians(3.0)

    for _ in range(25):
        tracker.predict(dt=0.0333)
        tracker.update_optical_los(target_az, target_el)

    final_state = tracker.get_state()
    assert abs(final_state.azimuth_deg - 5.0) < 0.5
    assert abs(final_state.elevation_deg - 3.0) < 0.5
    assert final_state.uncertainty_radius < initial_uncertainty


def test_ekf_imu_platform_attitude_fusion():
    """Verify platform roll/pitch IMU fusion and inertial decoupling."""
    tracker = ExtendedKalmanTracker(dt=0.0333)
    tracker.init_state(range_m=1000.0, azimuth_deg=10.0, elevation_deg=0.0)

    # Simulate platform pitch disturbance of 3 degrees
    pitch_disturb_rad = math.radians(3.0)
    tracker.update_imu_attitude(roll_rad=0.0, pitch_rad=pitch_disturb_rad, yaw_rad=0.0)

    state = tracker.get_state()
    assert abs(state.pitch_deg - 3.0) < 0.2
