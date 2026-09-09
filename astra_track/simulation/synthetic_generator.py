"""
Synthetic Test Video Benchmark Generator.
Generates MP4 benchmark videos alongside matching Ground Truth CSV logs and JSON configuration metadata.
Includes 9 pre-configured test scenarios.
"""

import os
import csv
import json
import time
from typing import Callable, Optional, Dict, Any
import cv2
import numpy as np

from astra_track.app.config import AstraConfig
from astra_track.app.constants import TrajectoryType, AtmosphereMode, DisturbancePattern
from astra_track.core.pipeline import TrackingPipeline


class SyntheticVideoGenerator:
    """Generates benchmark-ready optical videos with synchronized ground truth files."""

    @staticmethod
    def generate_benchmark(
        config: AstraConfig,
        output_mp4: str,
        duration_sec: float = 10.0,
        loss_trigger_time_s: Optional[float] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, str]:
        """
        Synthesizes an MP4 video, a CSV ground-truth file, and a JSON config file.
        Returns:
            Dict with 'video', 'csv', 'json' paths.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
        base_name = os.path.splitext(output_mp4)[0]
        csv_path = f"{base_name}_ground_truth.csv"
        json_path = f"{base_name}_config.json"

        # Save configuration JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

        fps = config.camera.fps
        total_frames = int(round(duration_sec * fps))
        width = config.camera.width
        height = config.camera.height

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_mp4, fourcc, float(fps), (width, height), isColor=False)
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_mp4, fourcc, float(fps), (width, height), isColor=False)

        # Initialize CSV logging
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_id", "timestamp", "actual_x", "actual_y", "detected_x", "detected_y",
            "filtered_x", "filtered_y", "predicted_x", "predicted_y", "pan", "tilt",
            "confidence", "tracking_status"
        ])

        pipeline = TrackingPipeline(config)

        try:
            for f_idx in range(total_frames):
                sim_time = f_idx / float(fps)

                # Check if target loss should be triggered
                if loss_trigger_time_s is not None and abs(sim_time - loss_trigger_time_s) < (0.5 / fps):
                    pipeline.sim_engine.trigger_target_loss(duration_frames=int(fps * 0.8))

                proc = pipeline.process_simulation_tick()

                # Write raw sensor frame to MP4
                writer.write(proc.raw_frame)

                # Write Ground Truth Row
                pred_x = proc.prediction.predicted_points[0][0] if proc.prediction.predicted_points else proc.kalman.x
                pred_y = proc.prediction.predicted_points[0][1] if proc.prediction.predicted_points else proc.kalman.y

                csv_writer.writerow([
                    proc.frame_id,
                    f"{proc.timestamp:.4f}",
                    f"{proc.ground_truth.camera_x:.2f}",
                    f"{proc.ground_truth.camera_y:.2f}",
                    f"{proc.detection.x:.2f}" if proc.detection.detected else "-1.00",
                    f"{proc.detection.y:.2f}" if proc.detection.detected else "-1.00",
                    f"{proc.kalman.x:.2f}",
                    f"{proc.kalman.y:.2f}",
                    f"{pred_x:.2f}",
                    f"{pred_y:.2f}",
                    f"{proc.ptu_pan:.3f}",
                    f"{proc.ptu_tilt:.3f}",
                    f"{proc.detection.confidence:.3f}",
                    proc.tracking_status.value
                ])

                if progress_cb and (f_idx % 15 == 0 or f_idx == total_frames - 1):
                    pct = int(((f_idx + 1) / total_frames) * 100)
                    progress_cb(f_idx + 1, total_frames, f"Generating frame {f_idx + 1}/{total_frames} ({pct}%)")

        finally:
            writer.release()
            csv_file.close()

        return {
            "video": output_mp4,
            "csv": csv_path,
            "json": json_path
        }

    # 9 One-Click Test Presets
    @classmethod
    def generate_straight_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_easy()
        cfg.trajectory.type = TrajectoryType.STRAIGHT.value
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_straight.mp4"), duration_sec=8.0)

    @classmethod
    def generate_circular_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_moderate()
        cfg.trajectory.type = TrajectoryType.CIRCULAR.value
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_circular.mp4"), duration_sec=10.0)

    @classmethod
    def generate_figure8_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_hard()
        cfg.trajectory.type = TrajectoryType.FIGURE8.value
        cfg.noise.salt_pepper = 0.0
        cfg.atmosphere.intensity = 0.0
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_figure8.mp4"), duration_sec=12.0)

    @classmethod
    def generate_random_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig()
        cfg.trajectory.type = TrajectoryType.RANDOM.value
        cfg.trajectory.speed = 70.0
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_random.mp4"), duration_sec=10.0)

    @classmethod
    def generate_noise_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_moderate()
        cfg.noise.salt_pepper = 0.10
        cfg.noise.gaussian_std = 10.0
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_noise.mp4"), duration_sec=8.0)

    @classmethod
    def generate_fog_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_hard()
        cfg.atmosphere.mode = AtmosphereMode.FOG.value
        cfg.atmosphere.intensity = 0.50
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_fog.mp4"), duration_sec=8.0)

    @classmethod
    def generate_jitter_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig()
        cfg.disturbance.camera_jitter_amp = 12.0
        cfg.disturbance.camera_jitter_pattern = DisturbancePattern.SINUSOIDAL.value
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_jitter.mp4"), duration_sec=8.0)

    @classmethod
    def generate_target_loss_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_reacquisition()
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_target_loss.mp4"), duration_sec=10.0, loss_trigger_time_s=3.5)

    @classmethod
    def generate_full_stress_test(cls, output_dir: str = "reports/videos") -> Dict[str, str]:
        cfg = AstraConfig.preset_extreme()
        return cls.generate_benchmark(cfg, os.path.join(output_dir, "test_stress.mp4"), duration_sec=12.0, loss_trigger_time_s=4.0)
