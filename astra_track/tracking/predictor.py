"""
Pattern-aware forward trajectory predictor.
Extrapolates beacon kinematics into the future along curved, orbital, and periodic patterns
(Figure-8, Circular, Sinusoidal, Harmonic) to eliminate tracking lag and enable seamless
reacquisition during target loss / beam occultation.
"""

import math
from collections import deque
from typing import List, Tuple, Optional, Any
import numpy as np
from astra_track.core.frame import KalmanState, PredictionResult


class TrajectoryPredictor:
    """
    Projects future beacon coordinates and uncertainty boundaries along
    curved patterns using Coordinated Turn (CT) kinematics, historical spline fitting,
    and orbital pattern models.
    """

    def __init__(self, horizon_frames: int = 10, dt: float = 0.0333):
        self.horizon_frames = horizon_frames
        self.dt = dt
        self.history: deque = deque(maxlen=60) # (t, x, y, vx, vy)
        self.trajectory_model: Optional[Any] = None
        self.last_turn_rate: float = 0.0

    def reset(self):
        """Clears trajectory history and resets internal kinematic estimators."""
        self.history.clear()
        self.last_turn_rate = 0.0

    def set_trajectory_model(self, model: Any):
        """Optionally binds orbital ephemeris or simulation trajectory generator."""
        self.trajectory_model = model

    def update_history(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, t: float = 0.0):
        """Appends confirmed measurement or estimated position to history buffer."""
        self.history.append((t, float(x), float(y), float(vx), float(vy)))

    def _estimate_turn_rate(self, vx: float, vy: float, ax: float, ay: float) -> float:
        """
        Estimates instantaneous angular turning rate omega (rad/s).
        Uses both analytic cross-product kinematics and empirical heading delta from history.
        """
        speed_sq = vx * vx + vy * vy
        omega_analytic = 0.0
        if speed_sq > 4.0:
            omega_analytic = (vx * ay - vy * ax) / speed_sq

        # If history is available, compute empirical turn rate from recent velocity headings
        omega_empirical = 0.0
        if len(self.history) >= 5:
            headings = []
            for item in list(self.history)[-10:]:
                h_vx, h_vy = item[3], item[4]
                if math.hypot(h_vx, h_vy) > 2.0:
                    headings.append(math.atan2(h_vy, h_vx))
            if len(headings) >= 3:
                # Unwrap headings to avoid 2*pi boundary jumps
                unwrapped = np.unwrap(headings)
                d_theta = unwrapped[-1] - unwrapped[0]
                dt_total = max(1e-3, (len(headings) - 1) * self.dt)
                omega_empirical = float(d_theta / dt_total)

        # Blend analytic and empirical turn rate
        if abs(omega_empirical) > 0.01:
            omega = 0.6 * omega_empirical + 0.4 * omega_analytic
        else:
            omega = omega_analytic

        # Hard clamp turn rate to physical aerospace limits (e.g. +/- 4.0 rad/s)
        omega = float(np.clip(omega, -4.0, 4.0))
        self.last_turn_rate = 0.8 * self.last_turn_rate + 0.2 * omega
        return self.last_turn_rate

    def _fit_polynomial_pattern(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fits cubic 2D parametric polynomial x(tau), y(tau) over recent history."""
        if len(self.history) < 8:
            return None

        recent = list(self.history)[-20:]
        n = len(recent)
        tau = np.linspace(-(n - 1) * self.dt, 0.0, n)
        xs = np.array([p[1] for p in recent], dtype=np.float64)
        ys = np.array([p[2] for p in recent], dtype=np.float64)

        try:
            poly_deg = min(3, n - 2)
            poly_x = np.polyfit(tau, xs, deg=poly_deg)
            poly_y = np.polyfit(tau, ys, deg=poly_deg)
            return poly_x, poly_y
        except Exception:
            return None

    def predict(self, k_state: KalmanState, lead_frames: float = 3.0) -> PredictionResult:
        """
        Projects future positions for steps 1 to horizon_frames following the trajectory pattern.
        Also calculates lead target for feedforward predictive controller.
        """
        predicted_points: List[Tuple[float, float]] = []
        uncertainty_radii: List[float] = []

        base_x = k_state.x
        base_y = k_state.y
        vx = k_state.vx
        vy = k_state.vy
        ax = k_state.ax
        ay = k_state.ay
        speed = math.hypot(vx, vy)

        base_uncert = max(2.0, k_state.uncertainty_radius)
        omega = self._estimate_turn_rate(vx, vy, ax, ay)
        poly_pattern = self._fit_polynomial_pattern()

        for step in range(1, self.horizon_frames + 1):
            t_ahead = step * self.dt

            # Method A: Coordinated Turn (CT) model with curvature
            if abs(omega) > 0.03 and speed > 2.0:
                theta_0 = math.atan2(vy, vx)
                theta_t = theta_0 + omega * t_ahead
                px = base_x + (speed / omega) * (math.sin(theta_t) - math.sin(theta_0))
                py = base_y - (speed / omega) * (math.cos(theta_t) - math.cos(theta_0))

            # Method B: High-order polynomial pattern extrapolation
            elif poly_pattern is not None:
                poly_x, poly_y = poly_pattern
                # Evaluate polynomial forward at tau = t_ahead
                px = float(np.polyval(poly_x, t_ahead))
                py = float(np.polyval(poly_y, t_ahead))
                # Anchor correction to current base position
                px = base_x + (px - float(np.polyval(poly_x, 0.0)))
                py = base_y + (py - float(np.polyval(poly_y, 0.0)))

            # Method C: Standard kinematic extrapolation (for straight paths)
            else:
                t_ahead_sq = 0.5 * (t_ahead**2)
                px = base_x + vx * t_ahead + ax * t_ahead_sq
                py = base_y + vy * t_ahead + ay * t_ahead_sq

            # Uncertainty expansion
            growth_factor = 1.0 + 0.35 * step
            uncert = base_uncert * growth_factor

            predicted_points.append((float(px), float(py)))
            uncertainty_radii.append(float(uncert))

        # Lead point for predictive controller
        t_lead = lead_frames * self.dt
        if abs(omega) > 0.03 and speed > 2.0:
            theta_0 = math.atan2(vy, vx)
            theta_lead = theta_0 + omega * t_lead
            lead_x = base_x + (speed / omega) * (math.sin(theta_lead) - math.sin(theta_0))
            lead_y = base_y - (speed / omega) * (math.cos(theta_lead) - math.cos(theta_0))
        elif poly_pattern is not None:
            poly_x, poly_y = poly_pattern
            lead_px = float(np.polyval(poly_x, t_lead))
            lead_py = float(np.polyval(poly_y, t_lead))
            lead_x = base_x + (lead_px - float(np.polyval(poly_x, 0.0)))
            lead_y = base_y + (lead_py - float(np.polyval(poly_y, 0.0)))
        else:
            t_lead_sq = 0.5 * (t_lead**2)
            lead_x = base_x + vx * t_lead + ax * t_lead_sq
            lead_y = base_y + vy * t_lead + ay * t_lead_sq

        return PredictionResult(
            horizon_frames=self.horizon_frames,
            predicted_points=predicted_points,
            uncertainty_radii=uncertainty_radii,
            lead_target_x=float(lead_x),
            lead_target_y=float(lead_y)
        )

