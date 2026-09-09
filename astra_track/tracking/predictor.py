"""
N-step forward trajectory predictor.
Extrapolates beacon kinematics into the future to eliminate coarse alignment lag.
"""

import math
from typing import List, Tuple
from astra_track.core.frame import KalmanState, PredictionResult


class TrajectoryPredictor:
    """Projects future beacon coordinates and uncertainty boundaries."""

    def __init__(self, horizon_frames: int = 5, dt: float = 0.0333):
        self.horizon_frames = horizon_frames
        self.dt = dt

    def predict(self, k_state: KalmanState, lead_frames: float = 3.0) -> PredictionResult:
        """
        Calculates predicted positions for steps 1 to horizon_frames.
        Also calculates lead target for feedforward controller.
        """
        predicted_points: List[Tuple[float, float]] = []
        uncertainty_radii: List[float] = []

        base_x = k_state.x
        base_y = k_state.y
        vx = k_state.vx
        vy = k_state.vy
        ax = k_state.ax
        ay = k_state.ay

        base_uncert = max(2.0, k_state.uncertainty_radius)

        for step in range(1, self.horizon_frames + 1):
            t_ahead = step * self.dt
            t_ahead_sq = 0.5 * (t_ahead**2)

            px = base_x + vx * t_ahead + ax * t_ahead_sq
            py = base_y + vy * t_ahead + ay * t_ahead_sq

            # Uncertainty expands with prediction horizon
            growth_factor = 1.0 + 0.35 * step
            uncert = base_uncert * growth_factor

            predicted_points.append((float(px), float(py)))
            uncertainty_radii.append(float(uncert))

        # Feedforward lead point for camera controller
        t_lead = lead_frames * self.dt
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
