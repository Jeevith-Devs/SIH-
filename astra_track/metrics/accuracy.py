"""
Accuracy metrics: Euclidean tracking error, angular error, and Root Mean Square Error (RMSE).
"""

import math
from typing import List, Tuple, Dict, Any
import numpy as np


class AccuracyEvaluator:
    """Calculates spatial alignment precision metrics against ground truth."""

    @staticmethod
    def compute_frame_error(actual_x: float, actual_y: float, tracked_x: float, tracked_y: float) -> float:
        """Euclidean distance in camera sensor pixels."""
        if actual_x < 0 or actual_y < 0 or tracked_x < 0 or tracked_y < 0:
            return 0.0
        return float(math.hypot(actual_x - tracked_x, actual_y - tracked_y))

    @staticmethod
    def compute_summary(errors_px: List[float]) -> Dict[str, float]:
        """Calculates mean, maximum, standard deviation, and RMSE."""
        if not errors_px:
            return {
                "avg_error_px": 0.0,
                "max_error_px": 0.0,
                "min_error_px": 0.0,
                "rmse_px": 0.0,
                "std_error_px": 0.0
            }

        arr = np.array(errors_px, dtype=np.float64)
        avg_err = float(np.mean(arr))
        max_err = float(np.max(arr))
        min_err = float(np.min(arr))
        rmse = float(np.sqrt(np.mean(arr**2)))
        std_err = float(np.std(arr))

        return {
            "avg_error_px": avg_err,
            "max_error_px": max_err,
            "min_error_px": min_err,
            "rmse_px": rmse,
            "std_error_px": std_err
        }
