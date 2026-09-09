"""
Unit tests for mathematical trajectory generators.
"""

import math
import pytest
from astra_track.app.config import TrajectoryConfig
from astra_track.app.constants import TrajectoryType
from astra_track.simulation.trajectories import (
    StraightTrajectory, CircularTrajectory, Figure8Trajectory,
    SinusoidalTrajectory, RandomTrajectory
)


def test_straight_trajectory():
    cfg = TrajectoryConfig(type=TrajectoryType.STRAIGHT.value, speed=50.0)
    traj = StraightTrajectory(cfg, origin=(1000.0, 1000.0))
    x1, y1, vx1, vy1, ax1, ay1 = traj.step(0.0)
    x2, y2, vx2, vy2, ax2, ay2 = traj.step(1.0)
    
    assert 100.0 <= x1 <= 1900.0
    assert 100.0 <= y1 <= 1900.0
    # Speed magnitude check
    measured_speed = math.hypot(vx1, vy1)
    assert abs(measured_speed - 50.0) < 1.0


def test_circle_trajectory():
    cfg = TrajectoryConfig(type=TrajectoryType.CIRCULAR.value, speed=60.0, radius=200.0)
    traj = CircularTrajectory(cfg, origin=(1000.0, 1000.0))
    x0, y0, vx0, vy0, ax0, ay0 = traj.step(0.0)
    # At t=0, x = 1000 + R = 1200, y = 1000
    assert abs(x0 - 1200.0) < 1e-3
    assert abs(y0 - 1000.0) < 1e-3
    assert abs(vx0) < 1e-3
    assert abs(abs(vy0) - 60.0) < 1e-3


def test_figure8_trajectory():
    cfg = TrajectoryConfig(type=TrajectoryType.FIGURE8.value, speed=80.0, amplitude=250.0)
    traj = Figure8Trajectory(cfg, origin=(1000.0, 1000.0))
    x0, y0, vx0, vy0, ax0, ay0 = traj.step(0.0)
    assert abs(x0 - 1000.0) < 1e-3
    assert abs(y0 - 1000.0) < 1e-3
    # Check that it moves both positive and negative x
    states = [traj.step(t) for t in [1.0, 2.0, 3.0, 4.0, 5.0]]
    xs = [s[0] for s in states]
    assert max(xs) > 1000.0
    assert len(traj.trail) == 6


def test_sinusoidal_trajectory():
    cfg = TrajectoryConfig(type=TrajectoryType.SINUSOIDAL.value, speed=40.0, amplitude=150.0, frequency=0.1)
    traj = SinusoidalTrajectory(cfg, origin=(1000.0, 1000.0))
    x, y, vx, vy, ax, ay = traj.step(2.5) # At quarter period (2.5s for 0.1Hz), sin(pi/2) = 1
    assert abs(y - 1150.0) < 1.0


def test_random_trajectory():
    cfg = TrajectoryConfig(type=TrajectoryType.RANDOM.value, speed=50.0)
    traj = RandomTrajectory(cfg, origin=(1000.0, 1000.0), seed=123)
    for t in range(30):
        x, y, vx, vy, ax, ay = traj.step(t * 0.033)
        assert 100.0 <= x <= 1900.0
        assert 100.0 <= y <= 1900.0
