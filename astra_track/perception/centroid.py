"""
Beacon centroid estimation algorithms: Geometric centroid vs Intensity-weighted centroid.
"""

from typing import Tuple
import numpy as np
from astra_track.core.frame import CentroidResult


class CentroidCalculator:
    """Computes and compares geometric bounding box centroid vs sub-pixel intensity-weighted centroid."""

    @staticmethod
    def compute(patch: np.ndarray, bbox: Tuple[int, int, int, int]) -> CentroidResult:
        """
        Calculates geometric and intensity-weighted centroids.
        Args:
            patch: 2D numpy array (grayscale sub-image of candidate region)
            bbox: (x, y, w, h) bounding box on sensor image plane
        Returns:
            CentroidResult
        """
        x, y, w, h = bbox

        # 1. Geometric Centroid (Center of Bounding Box)
        geom_x = float(x + (w / 2.0))
        geom_y = float(y + (h / 2.0))

        if patch is None or patch.size == 0 or np.sum(patch) == 0:
            return CentroidResult(
                geometric_x=geom_x,
                geometric_y=geom_y,
                intensity_x=geom_x,
                intensity_y=geom_y,
                delta_px=0.0
            )

        # 2. Intensity-Weighted Centroid (Center of Mass)
        # cx = sum(I * x) / sum(I)
        # cy = sum(I * y) / sum(I)
        h_p, w_p = patch.shape
        y_indices, x_indices = np.indices((h_p, w_p))
        
        # Apply soft background suppression to isolate spot peak
        threshold = np.percentile(patch, 40)
        weights = np.maximum(0.0, patch.astype(np.float64) - threshold)
        total_weight = np.sum(weights)

        if total_weight > 1e-6:
            local_int_x = np.sum(x_indices * weights) / total_weight
            local_int_y = np.sum(y_indices * weights) / total_weight
            int_x = float(x + local_int_x)
            int_y = float(y + local_int_y)
        else:
            int_x = geom_x
            int_y = geom_y

        delta_px = float(np.hypot(int_x - geom_x, int_y - geom_y))

        return CentroidResult(
            geometric_x=geom_x,
            geometric_y=geom_y,
            intensity_x=int_x,
            intensity_y=int_y,
            delta_px=delta_px
        )
