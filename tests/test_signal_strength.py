"""
Unit tests for optical laser signal strength, RSSI (dBm), SNR (dB), and coupling efficiency.
"""

import numpy as np
import pytest
from astra_track.perception.classical_detector import ClassicalDetector
from astra_track.core.frame import DetectionResult, ProcessedFrame
from astra_track.core.pipeline import TrackingPipeline
from astra_track.app.config import AstraConfig


def test_clean_laser_beacon_signal_strength():
    detector = ClassicalDetector()
    frame = np.zeros((480, 640), dtype=np.uint8) + 10
    # Clean bright beacon spot at (320, 240)
    frame[235:245, 315:325] = 250

    res = detector.detect(frame)
    assert res.detected is True
    assert res.snr_db > 15.0 # High optical SNR in dB
    assert res.rssi_dbm > -20.0 # High received power in dBm
    assert res.signal_strength_pct > 80.0 # High signal percentage
    assert res.coupling_efficiency > 0.85 # Strong coupling efficiency


def test_dimmed_beacon_lower_signal():
    detector = ClassicalDetector()
    frame_bright = np.zeros((480, 640), dtype=np.uint8) + 10
    frame_bright[235:245, 315:325] = 240
    res_bright = detector.detect(frame_bright)

    frame_dim = np.zeros((480, 640), dtype=np.uint8) + 10
    frame_dim[235:245, 315:325] = 90
    res_dim = detector.detect(frame_dim)

    assert res_bright.detected is True
    assert res_dim.detected is True
    # Dimmed beacon must have strictly lower RSSI and signal percentage
    assert res_dim.rssi_dbm < res_bright.rssi_dbm
    assert res_dim.signal_strength_pct < res_bright.signal_strength_pct
    assert res_dim.snr_db < res_bright.snr_db


def test_empty_frame_signal_floor():
    detector = ClassicalDetector()
    empty_frame = np.zeros((480, 640), dtype=np.uint8) + 10
    res = detector.detect(empty_frame)

    assert res.detected is False
    assert res.signal_strength_pct == 0.0
    assert res.rssi_dbm == -50.0
    assert res.coupling_efficiency == 0.0


def test_pipeline_processed_frame_signal_strength():
    cfg = AstraConfig.preset_easy()
    pipe = TrackingPipeline(cfg)
    proc = pipe.process_simulation_tick()

    assert isinstance(proc, ProcessedFrame)
    assert hasattr(proc, "signal_strength_pct")
    assert hasattr(proc, "rssi_dbm")
    assert hasattr(proc, "snr_db")
    assert hasattr(proc, "coupling_efficiency")
    assert proc.signal_strength_pct >= 0.0
    assert proc.rssi_dbm >= -50.0
