"""
Confidence estimator for beacon detection candidates in adverse optical conditions.
Synthesizes Peak SNR, shape compactness, intensity contrast, and Kalman spatial gating.
"""

import math
from typing import Optional, Tuple
import numpy as np


class ConfidenceScorer:
    """Computes multi-parametric confidence metric [0.0, 1.0] for coarse alignment verification."""

    @staticmethod
    def calculate_confidence(
        peak_intensity: float,
        mean_intensity: float,
        bg_mean: float,
        bg_std: float,
        area: float,
        circularity: float,
        expected_size: float = 10.0,
        distance_from_prediction: Optional[float] = None,
        gating_sigma: float = 25.0
    ) -> float:
        """
        Synthesizes confidence from SNR, shape, size match, and temporal spatial prior.
        """
        # 1. Optical Peak SNR: (I_peak - bg_mean) / (bg_std + 1)
        effective_std = max(1.5, bg_std)
        snr = max(0.0, (peak_intensity - bg_mean) / effective_std)
        # Map SNR to [0, 1] using sigmoid (typical strong beacon SNR >= 8)
        snr_score = 1.0 / (1.0 + math.exp(-0.4 * (snr - 4.5)))

        # 2. Normalized peak brightness (beacon spots are intense, typically > 120 DN)
        brightness_score = max(0.0, min(1.0, (peak_intensity - 30.0) / 180.0))

        # 3. Shape Circularity / Compactness (beacon spot is compact, typical circularity > 0.4)
        circ_score = max(0.0, min(1.0, circularity / 0.8))

        # 4. Size consistency match
        expected_area = expected_size * expected_size
        size_ratio = min(area, expected_area) / max(area, expected_area, 1.0)
        size_score = max(0.1, size_ratio)

        # Baseline visual confidence
        visual_conf = (
            0.35 * snr_score +
            0.30 * brightness_score +
            0.20 * circ_score +
            0.15 * size_score
        )
        
        # Penalize sub-resolution impulse noise specks (smallest physical beacon is 5x5 = 25px)
        if area < 8.0:
            visual_conf *= ((area / 8.0) ** 2)

        # 5. Spatial Prior from Kalman Prediction Gating
        if distance_from_prediction is not None and not math.isnan(distance_from_prediction):
            spatial_prior = math.exp(-0.5 * (distance_from_prediction / max(5.0, gating_sigma))**2)
            # Blend visual detection confidence with spatial track prior
            total_conf = 0.70 * visual_conf + 0.30 * spatial_prior
        else:
            total_conf = visual_conf

        return float(max(0.0, min(1.0, total_conf)))
