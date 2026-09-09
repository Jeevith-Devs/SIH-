"""
Unit tests for SlideExporter, AudioAlertManager, and 3D Gimbal math.
"""

import os
import pytest
from astra_track.storage.slide_exporter import SlideExporter
from astra_track.ui.audio_alerts import AudioAlertManager
from astra_track.ui.widgets.gimbal_3d_widget import Gimbal3DWidget


def test_slide_exporter_generation(tmp_path):
    """Verify that SlideExporter generates a valid standalone HTML pitch deck."""
    test_html = os.path.join(str(tmp_path), "test_pitch.html")
    benchmark_data = {
        "avg_fps": 85.0,
        "avg_error_px": 3.2,
        "passed_scenarios": 5,
        "total_scenarios": 5
    }

    path = SlideExporter.export_presentation_html(output_path=test_html, benchmark_results=benchmark_data)
    assert os.path.isfile(path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ASTRA-Track" in content
    assert "Problem Statement 26169" in content
    assert "85.0 FPS" in content
    assert "3.20 px" in content
    assert "Slide 1 of 6" in content


def test_audio_alerts_manager():
    """Verify AudioAlertManager executes without raising exceptions."""
    mgr = AudioAlertManager(enabled=False)  # Muted in test runner
    mgr.play_lock_acquired()
    mgr.play_target_lost()
    mgr.play_reacquired()
    mgr.play_calibration_beep()

    # Even if enabled, should catch and suppress any audio driver errors
    mgr_enabled = AudioAlertManager(enabled=True)
    mgr_enabled.play_calibration_beep()
