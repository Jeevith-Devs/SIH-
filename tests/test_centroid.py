"""
Unit tests for centroid algorithms: Geometric centroid vs Intensity-weighted centroid.
"""

import numpy as np
from astra_track.perception.centroid import CentroidCalculator


def test_geometric_centroid():
    patch = np.ones((11, 11), dtype=np.uint8) * 200
    bbox = (100, 200, 11, 11)
    res = CentroidCalculator.compute(patch, bbox)
    assert abs(res.geometric_x - 105.5) < 1e-4
    assert abs(res.geometric_y - 205.5) < 1e-4


def test_intensity_weighted_centroid_asymmetric():
    # Place hot spot off-center to verify sub-pixel pull
    patch = np.zeros((15, 15), dtype=np.uint8)
    # Bright spot at local (10, 10)
    patch[9:12, 9:12] = 250
    bbox = (50, 50, 15, 15)
    res = CentroidCalculator.compute(patch, bbox)
    
    # Geometric center is 50 + 7.5 = 57.5
    assert abs(res.geometric_x - 57.5) < 1e-3
    # Intensity center should be pulled toward 50 + 10 = 60.0
    assert res.intensity_x > res.geometric_x
    assert res.intensity_y > res.geometric_y
    assert res.delta_px > 1.0
