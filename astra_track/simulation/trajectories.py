"""
Mathematical trajectory generators for mobile FSOC optical beacon targets.
Supports Straight Line, Circular, Figure-8, Random, Spiral, Sinusoidal, and Custom waypoints.
Calculates ground truth position, velocity, and acceleration analytically.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
from astra_track.app.constants import TrajectoryType
from astra_track.app.config import TrajectoryConfig


class TrajectoryBase:
    """Base class for beacon motion trajectories."""

    def __init__(self, config: TrajectoryConfig, origin: Tuple[float, float] = (1000.0, 1000.0)):
        self.config = config
        self.origin_x, self.origin_y = origin
        self.trail: List[Tuple[float, float]] = []
        self.max_trail_len = 250

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        """
        Returns (x, y, vx, vy, ax, ay) at simulation time t.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self, t: float) -> Tuple[float, float, float, float, float, float]:
        state = self.get_state(t)
        self.trail.append((state[0], state[1]))
        if len(self.trail) > self.max_trail_len:
            self.trail.pop(0)
        return state

    def reset(self):
        self.trail.clear()


class StraightTrajectory(TrajectoryBase):
    """Linear movement starting at origin and moving smoothly back and forth."""

    def __init__(self, config: TrajectoryConfig, origin: Tuple[float, float] = (1000.0, 1000.0), world_bounds: Tuple[int, int] = (2000, 2000)):
        super().__init__(config, origin)
        self.world_w, self.world_h = world_bounds
        self.angle = math.radians(35.0)
        self.speed = config.speed
        self.half_span = 350.0

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        period = (4.0 * self.half_span) / max(1.0, self.speed)
        phase = (t % period) / period

        if phase < 0.25:
            tri = 4.0 * phase               # 0 -> 1
            v_dir = 1.0
        elif phase < 0.75:
            tri = 1.0 - 4.0 * (phase - 0.25) # 1 -> -1
            v_dir = -1.0
        else:
            tri = -1.0 + 4.0 * (phase - 0.75) # -1 -> 0
            v_dir = 1.0

        x = self.origin_x + self.half_span * tri * math.cos(self.angle)
        y = self.origin_y + self.half_span * tri * math.sin(self.angle)
        vx = self.speed * math.cos(self.angle) * v_dir
        vy = self.speed * math.sin(self.angle) * v_dir
        ax = 0.0
        ay = 0.0

        return x, y, vx, vy, ax, ay


class CircularTrajectory(TrajectoryBase):
    """Circular trajectory around origin with configurable radius and angular speed."""

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        R = max(20.0, self.config.radius)
        # omega = speed / R (rad/s)
        omega = (self.config.speed / R) * self.config.direction
        
        x = self.origin_x + R * math.cos(omega * t)
        y = self.origin_y + R * math.sin(omega * t)

        vx = -R * omega * math.sin(omega * t)
        vy = R * omega * math.cos(omega * t)

        ax = -R * (omega**2) * math.cos(omega * t)
        ay = -R * (omega**2) * math.sin(omega * t)

        return x, y, vx, vy, ax, ay


class Figure8Trajectory(TrajectoryBase):
    """Lemniscate / Lissajous Figure-8 trajectory testing multi-axis direction reversals."""

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        A = max(20.0, self.config.amplitude)
        omega = (self.config.speed / A)
        
        x = self.origin_x + A * math.sin(omega * t)
        y = self.origin_y + 0.5 * A * math.sin(2.0 * omega * t)

        vx = A * omega * math.cos(omega * t)
        vy = A * omega * math.cos(2.0 * omega * t)

        ax = -A * (omega**2) * math.sin(omega * t)
        ay = -2.0 * A * (omega**2) * math.sin(2.0 * omega * t)

        return x, y, vx, vy, ax, ay


class SinusoidalTrajectory(TrajectoryBase):
    """Sinusoidal wave trajectory oscillating across origin."""

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        A = self.config.amplitude
        speed = self.config.speed
        half_span = 350.0
        period_x = (4.0 * half_span) / max(1.0, speed)
        phase_x = (t % period_x) / period_x

        if phase_x < 0.25:
            tri = 4.0 * phase_x
            vx = speed
        elif phase_x < 0.75:
            tri = 1.0 - 4.0 * (phase_x - 0.25)
            vx = -speed
        else:
            tri = -1.0 + 4.0 * (phase_x - 0.75)
            vx = speed

        x = self.origin_x + half_span * tri
        omega_y = 2.0 * math.pi * self.config.frequency
        y = self.origin_y + A * math.sin(omega_y * t)

        vy = A * omega_y * math.cos(omega_y * t)
        ax = 0.0
        ay = -A * (omega_y**2) * math.sin(omega_y * t)

        return x, y, vx, vy, ax, ay


class SpiralTrajectory(TrajectoryBase):
    """Expanding and contracting spiral trajectory."""

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        max_r = max(50.0, self.config.radius)
        # Breathe in and out smoothly
        cycle_period = 20.0
        r_phase = (t % cycle_period) / cycle_period
        r_scale = math.sin(r_phase * math.pi)
        R = 40.0 + (max_r - 40.0) * r_scale

        omega = 0.8 * self.config.direction
        x = self.origin_x + R * math.cos(omega * t)
        y = self.origin_y + R * math.sin(omega * t)

        # Approximate velocity
        vx = -R * omega * math.sin(omega * t)
        vy = R * omega * math.cos(omega * t)
        ax = -R * (omega**2) * math.cos(omega * t)
        ay = -R * (omega**2) * math.sin(omega * t)

        return x, y, vx, vy, ax, ay


class RandomTrajectory(TrajectoryBase):
    """Smooth random walk with momentum and soft boundary repulsion."""

    def __init__(self, config: TrajectoryConfig, origin: Tuple[float, float] = (1000.0, 1000.0), seed: int = 42):
        super().__init__(config, origin)
        self.rng = np.random.default_rng(seed)
        self.cur_x = origin[0]
        self.cur_y = origin[1]
        self.cur_vx = (self.rng.random() - 0.5) * config.speed
        self.cur_vy = (self.rng.random() - 0.5) * config.speed
        self.cur_ax = 0.0
        self.cur_ay = 0.0
        self.last_t = 0.0

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        dt = max(0.001, min(0.1, t - self.last_t if self.last_t > 0 else 0.033))
        self.last_t = t

        # Random steering acceleration
        accel_noise_std = 30.0
        target_ax = self.rng.normal(0.0, accel_noise_std)
        target_ay = self.rng.normal(0.0, accel_noise_std)

        # Soft boundary repulsion from [250, 1750]
        min_bound, max_bound = 250.0, 1750.0
        if self.cur_x < min_bound:
            target_ax += 80.0
        elif self.cur_x > max_bound:
            target_ax -= 80.0
            
        if self.cur_y < min_bound:
            target_ay += 80.0
        elif self.cur_y > max_bound:
            target_ay -= 80.0

        # Update velocity with damping
        damping = 0.98
        self.cur_ax = target_ax
        self.cur_ay = target_ay
        self.cur_vx = (self.cur_vx + self.cur_ax * dt) * damping
        self.cur_vy = (self.cur_vy + self.cur_ay * dt) * damping

        # Speed clamp
        current_speed = math.hypot(self.cur_vx, self.cur_vy)
        max_speed = self.config.speed
        if current_speed > max_speed:
            self.cur_vx = (self.cur_vx / current_speed) * max_speed
            self.cur_vy = (self.cur_vy / current_speed) * max_speed

        self.cur_x += self.cur_vx * dt
        self.cur_y += self.cur_vy * dt

        return self.cur_x, self.cur_y, self.cur_vx, self.cur_vy, self.cur_ax, self.cur_ay


class CustomTrajectory(TrajectoryBase):
    """Traverses a custom list of waypoints with linear/spline interpolation."""

    def __init__(self, config: TrajectoryConfig, origin: Tuple[float, float] = (1000.0, 1000.0)):
        super().__init__(config, origin)
        self.waypoints = config.waypoints if config.waypoints else [
            (origin[0] - 200, origin[1] - 200),
            (origin[0] + 200, origin[1] - 200),
            (origin[0] + 200, origin[1] + 200),
            (origin[0] - 200, origin[1] + 200),
        ]

    def get_state(self, t: float) -> Tuple[float, float, float, float, float, float]:
        if len(self.waypoints) < 2:
            return self.origin_x, self.origin_y, 0.0, 0.0, 0.0, 0.0

        # Total distance
        segments = []
        total_dist = 0.0
        for i in range(len(self.waypoints)):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % len(self.waypoints)]
            d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            segments.append((p1, p2, d))
            total_dist += d

        if total_dist == 0:
            return self.origin_x, self.origin_y, 0.0, 0.0, 0.0, 0.0

        loop_time = total_dist / max(5.0, self.config.speed)
        progress_dist = ((t % loop_time) / loop_time) * total_dist

        accum = 0.0
        for p1, p2, d in segments:
            if accum + d >= progress_dist:
                seg_ratio = (progress_dist - accum) / d if d > 0 else 0.0
                x = p1[0] + (p2[0] - p1[0]) * seg_ratio
                y = p1[1] + (p2[1] - p1[1]) * seg_ratio
                unit_dx = (p2[0] - p1[0]) / d if d > 0 else 0.0
                unit_dy = (p2[1] - p1[1]) / d if d > 0 else 0.0
                vx = unit_dx * self.config.speed
                vy = unit_dy * self.config.speed
                return x, y, vx, vy, 0.0, 0.0
            accum += d

        p = self.waypoints[0]
        return p[0], p[1], 0.0, 0.0, 0.0, 0.0


def create_trajectory(config: TrajectoryConfig, origin: Tuple[float, float] = (1000.0, 1000.0), world_bounds: Tuple[int, int] = (2000, 2000)) -> TrajectoryBase:
    """Factory helper to instantiate trajectory generator according to config."""
    t_type = config.type
    if t_type == TrajectoryType.STRAIGHT.value:
        return StraightTrajectory(config, origin, world_bounds)
    elif t_type == TrajectoryType.CIRCULAR.value:
        return CircularTrajectory(config, origin)
    elif t_type == TrajectoryType.FIGURE8.value:
        return Figure8Trajectory(config, origin)
    elif t_type == TrajectoryType.SINUSOIDAL.value:
        return SinusoidalTrajectory(config, origin)
    elif t_type == TrajectoryType.SPIRAL.value:
        return SpiralTrajectory(config, origin)
    elif t_type == TrajectoryType.RANDOM.value:
        return RandomTrajectory(config, origin)
    elif t_type == TrajectoryType.CUSTOM.value:
        return CustomTrajectory(config, origin)
    else:
        return Figure8Trajectory(config, origin)
