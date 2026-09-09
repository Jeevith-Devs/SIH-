"""
Unit tests for classical CV detector and noise robustness.
"""

import numpy as np
from astra_track.perception.classical_detector import ClassicalDetector
from astra_track.simulation.noise import NoiseSimulator
from astra_track.app.config import NoiseConfig


def test_beacon_detection_clean():
    detector = ClassicalDetector()
    frame = np.zeros((480, 640), dtype=np.uint8) + 12
    # Place clean 10x10 beacon at (320, 240)
    frame[235:245, 315:325] = 240

    res = detector.detect(frame)
    assert res.detected is True
    assert abs(res.x - 320.0) < 3.0
    assert abs(res.y - 240.0) < 3.0
    assert res.confidence > 0.70


def test_beacon_detection_noise():
    detector = ClassicalDetector()
    frame = np.zeros((480, 640), dtype=np.uint8) + 12
    # Beacon
    frame[235:245, 315:325] = 250
    # Add 8% salt & pepper noise + Gaussian noise
    noise_sim = NoiseSimulator(NoiseConfig(salt_pepper=0.08, gaussian_std=8.0), seed=42)
    noisy_frame = noise_sim.apply(frame)

    res = detector.detect(noisy_frame, predicted_pos=(320.0, 240.0))
    assert res.detected is True
    assert abs(res.x - 320.0) < 6.0
    assert abs(res.y - 240.0) < 6.0


def test_no_beacon_empty():
    detector = ClassicalDetector()
    frame = np.zeros((480, 640), dtype=np.uint8) + 15
    res = detector.detect(frame)
    assert res.detected is False
    assert res.confidence == 0.0
