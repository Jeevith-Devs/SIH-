"""
Unit and integration tests for temporal carrier frequency discrimination and decoy rejection.
"""

import math
import numpy as np
import pytest

from astra_track.app.config import TrackerConfig
from astra_track.perception.temporal_filter import TemporalBeaconFilter, TemporalAnalysisResult
from astra_track.perception.detector import UnifiedDetector


def test_temporal_filter_static_glint():
    """Verify that unmodulated DC optical sources (sun glint, streetlights) are flagged as static."""
    filt = TemporalBeaconFilter(fps=30.0, buffer_size=24, target_carrier_hz=15.0)

    # Feed 24 identical bright DC samples (e.g. bright unmodulated glint)
    for _ in range(24):
        filt.update_sample(track_id=1, intensity=220.0, position=(100.0, 100.0))

    res = filt.analyze_track(1)
    assert res.is_static_glint is True
    assert res.is_carrier_matched is False
    assert res.modulation_depth < 0.05
    # Should penalize confidence for unmodulated DC source when carrier is expected
    assert res.confidence_factor < 0.50


def test_temporal_filter_carrier_match():
    """Verify detection of a 10 Hz modulated beacon at 30 FPS."""
    carrier_freq = 10.0
    fps = 30.0
    filt = TemporalBeaconFilter(fps=fps, buffer_size=30, target_carrier_hz=carrier_freq, frequency_tolerance_hz=2.0)

    # Generate 30 samples of a 10 Hz sinusoidal optical pulse
    for i in range(30):
        t = i / fps
        intensity = 150.0 + 80.0 * math.sin(2.0 * math.pi * carrier_freq * t)
        filt.update_sample(track_id=1, intensity=intensity, position=(150.0, 150.0))

    res = filt.analyze_track(1)
    assert abs(res.dominant_frequency_hz - carrier_freq) <= 2.0
    assert res.is_carrier_matched is True
    assert res.is_static_glint is False
    assert res.spectral_purity > 0.40
    assert res.modulation_depth > 0.20
    assert res.confidence_factor > 1.0


def test_temporal_filter_carrier_mismatch():
    """Verify that a decoy modulating at the wrong frequency (e.g. 25 Hz vs 10 Hz) is rejected."""
    fps = 60.0
    target_carrier = 10.0
    decoy_freq = 25.0
    filt = TemporalBeaconFilter(fps=fps, buffer_size=32, target_carrier_hz=target_carrier, frequency_tolerance_hz=2.0)

    for i in range(32):
        t = i / fps
        intensity = 140.0 + 70.0 * math.sin(2.0 * math.pi * decoy_freq * t)
        filt.update_sample(track_id=1, intensity=intensity, position=(200.0, 200.0))

    res = filt.analyze_track(1)
    assert abs(res.dominant_frequency_hz - decoy_freq) <= 2.0
    assert res.is_carrier_matched is False
    assert res.confidence_factor < 0.85


def test_unified_detector_temporal_integration():
    """Verify that UnifiedDetector correctly updates and reports temporal metrics."""
    config = TrackerConfig(
        temporal_filtering_enabled=True,
        beacon_carrier_freq_hz=10.0,
        roi_search_enabled=False
    )
    detector = UnifiedDetector(config, frame_w=100, frame_h=100)

    # Generate frames with a bright spot at (50, 50)
    for i in range(16):
        frame = np.zeros((100, 100), dtype=np.uint8)
        # Spot intensity modulated around 10 Hz
        intensity = int(120 + 80 * math.sin(2.0 * math.pi * 10.0 * (i / 30.0)))
        frame[48:53, 48:53] = np.clip(intensity, 40, 255)

        result = detector.detect(frame)
        assert result.detected is True

    # After 16 frames, temporal analysis should be active
    assert result.carrier_freq_hz > 0.0
    assert result.confidence > 0.0
