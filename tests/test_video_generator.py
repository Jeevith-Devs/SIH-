"""
Unit tests for synthetic benchmark video generator and ground truth alignment.
"""

import os
import csv
from astra_track.app.config import AstraConfig
from astra_track.simulation.synthetic_generator import SyntheticVideoGenerator


def test_synthetic_video_generator(tmp_path):
    mp4_path = str(tmp_path / "test_out.mp4")
    cfg = AstraConfig.preset_easy()
    cfg.camera.fps = 15 # lower FPS for rapid test execution

    paths = SyntheticVideoGenerator.generate_benchmark(
        config=cfg,
        output_mp4=mp4_path,
        duration_sec=1.0 # 15 frames
    )

    assert os.path.exists(paths["video"])
    assert os.path.exists(paths["csv"])
    assert os.path.exists(paths["json"])

    # Verify CSV rows match frame count
    with open(paths["csv"], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Header + 15 frames = 16 rows
        assert len(rows) == 16
        assert rows[0][0] == "frame_id"
        assert rows[0][2] == "actual_x"
