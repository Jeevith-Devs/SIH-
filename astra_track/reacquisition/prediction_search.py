"""
Prediction-based search region generation for target loss reacquisition.
Uses extrapolated Kalman state and uncertainty covariance ellipse to bound the initial search ROI.
"""

from typing import Tuple
from astra_track.core.frame import KalmanState


class PredictionSearchROI:
    """Generates uncertainty-scaled ROI bounding boxes around extrapolated target position."""

    @staticmethod
    def compute_roi(
        k_state: KalmanState,
        frame_w: int = 640,
        frame_h: int = 480,
        expansion_factor: float = 3.5,
        min_box_size: float = 60.0
    ) -> Tuple[int, int, int, int]:
        """
        Returns (x, y, w, h) bounding box centered at Kalman prediction.
        """
        center_x = k_state.x
        center_y = k_state.y

        half_span = max(min_box_size / 2.0, expansion_factor * max(5.0, k_state.uncertainty_radius))

        rx1 = int(max(0, center_x - half_span))
        ry1 = int(max(0, center_y - half_span))
        rx2 = int(min(frame_w, center_x + half_span))
        ry2 = int(min(frame_h, center_y + half_span))

        rw = max(10, rx2 - rx1)
        rh = max(10, ry2 - ry1)

        return rx1, ry1, rw, rh
