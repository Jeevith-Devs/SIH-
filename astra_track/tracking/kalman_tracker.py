"""
6-State Kinematic Discrete Kalman Filter for optical beacon tracking.
State: X = [x, y, vx, vy, ax, ay]^T (position, velocity, acceleration).
Incorporates constant acceleration kinematic transitions, process noise Q,
measurement noise R, covariance propagation, and uncertainty radius extraction.
"""

import math
from typing import Tuple, Optional
import numpy as np

from astra_track.core.frame import KalmanState


class KalmanTracker:
    """Discrete 6-state Kalman Filter with continuous jerk process noise."""

    def __init__(self, dt: float = 0.0333, q_noise: float = 120.0, r_noise: float = 4.0):
        self.dt = dt
        self.q_noise = q_noise
        self.r_noise = r_noise

        # State vector: [x, y, vx, vy, ax, ay]^T
        self.x = np.zeros((6, 1), dtype=np.float64)
        
        # State transition matrix F
        self.F = self._build_transition_matrix(dt)

        # Measurement matrix H: maps state to [x, y]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * (r_noise**2)

        # Process noise covariance Q
        self.Q = self._build_process_noise(dt, q_noise)

        # Estimation error covariance matrix P
        self.P = np.eye(6, dtype=np.float64) * 100.0

        self.initialized = False
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        dt2 = 0.5 * (dt**2)
        return np.array([
            [1.0, 0.0,  dt, 0.0, dt2, 0.0],
            [0.0, 1.0, 0.0,  dt, 0.0, dt2],
            [0.0, 0.0, 1.0, 0.0,  dt, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _build_process_noise(self, dt: float, q: float) -> np.ndarray:
        """Piecewise constant jerk process noise."""
        dt2 = 0.5 * (dt**2)
        dt3 = dt**3 / 6.0
        # Noise coupling matrix G
        G = np.array([
            [dt3, 0.0],
            [0.0, dt3],
            [dt2, 0.0],
            [0.0, dt2],
            [ dt, 0.0],
            [0.0,  dt]
        ], dtype=np.float64)
        return np.dot(G, G.T) * q

    def init_state(self, x: float, y: float):
        """Initializes state with first detection measurement."""
        self.x = np.array([[x], [y], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.P = np.diag([5.0, 5.0, 400.0, 400.0, 1000.0, 1000.0]).astype(np.float64)
        self.initialized = True

    initialize = init_state

    def reset(self):
        self.initialized = False
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Kalman time update step.
        Advances state prior X_pred = F * X, P_pred = F * P * F^T + Q.
        """
        if dt is not None and abs(dt - self.dt) > 1e-4:
            self.dt = dt
            self.F = self._build_transition_matrix(dt)
            self.Q = self._build_process_noise(dt, self.q_noise)

        # State extrapolation
        self.x = np.dot(self.F, self.x)
        # Covariance extrapolation
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x: float, z_y: float) -> KalmanState:
        """
        Kalman measurement update step with sensor measurement [z_x, z_y].
        """
        if not self.initialized:
            self.init_state(z_x, z_y)
            return self.get_state()

        z = np.array([[z_x], [z_y]], dtype=np.float64)

        # Innovation (measurement residual)
        y = z - np.dot(self.H, self.x)
        self.last_innovation = y

        # Innovation covariance S
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Optimal Kalman Gain K
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state estimate
        self.x = self.x + np.dot(K, y)

        # Updated estimate covariance (Joseph form for numerical stability)
        I = np.eye(6, dtype=np.float64)
        I_KH = I - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)

        return self.get_state()

    def get_state(self) -> KalmanState:
        """Extracts structured Kalman state and uncertainty metrics."""
        x_val = float(self.x[0, 0])
        y_val = float(self.x[1, 0])
        vx_val = float(self.x[2, 0])
        vy_val = float(self.x[3, 0])
        ax_val = float(self.x[4, 0])
        ay_val = float(self.x[5, 0])

        speed = math.hypot(vx_val, vy_val)
        accel = math.hypot(ax_val, ay_val)

        var_x = float(self.P[0, 0])
        var_y = float(self.P[1, 1])

        # Maximum eigenvalue of 2x2 position covariance matrix
        cov_pos = self.P[0:2, 0:2]
        eigvals = np.linalg.eigvalsh(cov_pos)
        max_eig = max(0.01, float(np.max(eigvals)))
        uncertainty_radius = 2.0 * math.sqrt(max_eig) # 2-sigma boundary (95.4% confidence)

        return KalmanState(
            x=x_val,
            y=y_val,
            vx=vx_val,
            vy=vy_val,
            ax=ax_val,
            ay=ay_val,
            speed=speed,
            accel=accel,
            var_x=var_x,
            var_y=var_y,
            uncertainty_radius=uncertainty_radius,
            innovation_x=float(self.last_innovation[0, 0]),
            innovation_y=float(self.last_innovation[1, 0])
        )
