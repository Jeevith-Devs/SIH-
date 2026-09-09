"""
Automated Benchmark and Stress Test Runner for SIH 2026 Problem Statement 26169.
Executes batch evaluations (5, 10, 50, 100 scenarios) and randomized multi-variable stress tests.
"""

import time
import random
from typing import List, Dict, Any, Callable, Optional
import numpy as np

from astra_track.app.config import AstraConfig
from astra_track.app.constants import TrajectoryType, AtmosphereMode, DisturbancePattern
from astra_track.core.pipeline import TrackingPipeline
from astra_track.storage.session import SessionRecorder


class BenchmarkRunner:
    """Automated benchmark test battery evaluating FSOC coarse alignment against SIH criteria."""

    @staticmethod
    def run_scenario(
        config: AstraConfig,
        duration_s: float = 6.0,
        scenario_name: str = "Benchmark Run",
        inject_loss: bool = False
    ) -> Dict[str, Any]:
        """Runs a single scenario headlessly and returns its scorecard."""
        pipeline = TrackingPipeline(config)
        session = SessionRecorder(scenario_name)

        fps = config.camera.fps
        total_frames = int(round(duration_s * fps))

        for f_idx in range(total_frames):
            sim_time = f_idx / float(fps)
            if inject_loss and abs(sim_time - 2.5) < (0.5 / fps):
                pipeline.sim_engine.trigger_target_loss(duration_frames=int(fps * 0.7))

            proc = pipeline.process_simulation_tick()
            session.record_frame(proc, log_frame_detail=False)

        summary = session.finalize_summary(pipeline.fsm, config.to_dict())
        return summary

    @classmethod
    def run_batch(
        cls,
        count: int = 5,
        duration_per_scenario_s: float = 5.0,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes a batch of N scenarios (5, 10, 50, 100) covering easy, moderate, hard, and extreme presets.
        """
        presets = [
            ("Easy - Straight", AstraConfig.preset_easy, False),
            ("Moderate - Circular + Jitter", AstraConfig.preset_moderate, False),
            ("Hard - Figure-8 + Fog + Noise", AstraConfig.preset_hard, False),
            ("Reacquisition Stress", AstraConfig.preset_reacquisition, True),
            ("Extreme - Full Stress", AstraConfig.preset_extreme, False)
        ]

        results = []
        passed_count = 0

        for i in range(count):
            name_base, preset_fn, has_loss = presets[i % len(presets)]
            name = f"Scenario {i+1:02d}: {name_base}"

            if progress_cb:
                progress_cb(i, count, f"Running {name}...")

            cfg = preset_fn()
            res = cls.run_scenario(cfg, duration_s=duration_per_scenario_s, scenario_name=name, inject_loss=has_loss)
            results.append(res)
            if res["overall_pass"]:
                passed_count += 1

        if progress_cb:
            progress_cb(count, count, "Batch benchmark completed.")

        pass_rate = (passed_count / max(1, count)) * 100.0
        avg_errors = [r["avg_error_px"] for r in results]
        avg_fps_list = [r["avg_fps"] for r in results]
        loss_pcts = [r["target_loss_pct"] for r in results]

        summary = {
            "total_scenarios": count,
            "passed": passed_count,
            "failed": count - passed_count,
            "pass_rate_pct": pass_rate,
            "avg_tracking_error_px": float(np.mean(avg_errors)),
            "max_tracking_error_px": float(np.max([r["max_error_px"] for r in results])),
            "avg_target_loss_pct": float(np.mean(loss_pcts)),
            "avg_fps": float(np.mean(avg_fps_list)),
            "scenarios": results
        }
        return summary

    @classmethod
    def run_stress_test(
        cls,
        count: int = 10,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Randomly varies beacon speed, trajectory, noise, fog, jitter, platform movement.
        Produces Best case, Average case, Worst case, and Pass rate.
        """
        trajectories = [
            TrajectoryType.STRAIGHT.value, TrajectoryType.CIRCULAR.value,
            TrajectoryType.FIGURE8.value, TrajectoryType.SINUSOIDAL.value,
            TrajectoryType.RANDOM.value
        ]
        atmospheres = [
            AtmosphereMode.CLEAR.value, AtmosphereMode.HAZE.value,
            AtmosphereMode.FOG.value, AtmosphereMode.RAIN.value
        ]

        results = []
        rng = random.Random(42)

        for i in range(count):
            if progress_cb:
                progress_cb(i, count, f"Stress Scenario {i+1}/{count}...")

            cfg = AstraConfig()
            cfg.trajectory.type = rng.choice(trajectories)
            cfg.trajectory.speed = rng.uniform(50.0, 110.0)
            cfg.noise.salt_pepper = rng.uniform(0.0, 0.12)
            cfg.noise.gaussian_std = rng.uniform(0.0, 15.0)
            cfg.atmosphere.mode = rng.choice(atmospheres)
            cfg.atmosphere.intensity = rng.uniform(0.1, 0.65)
            cfg.disturbance.camera_jitter_amp = rng.uniform(0.0, 14.0)
            cfg.disturbance.platform_motion_amp = rng.uniform(0.0, 12.0)

            res = cls.run_scenario(cfg, duration_s=4.5, scenario_name=f"Stress #{i+1}", inject_loss=(i % 3 == 0))
            results.append(res)

        if progress_cb:
            progress_cb(count, count, "Stress testing completed.")

        # Rank by average tracking error
        results.sort(key=lambda r: r["avg_error_px"])
        best_case = results[0]
        worst_case = results[-1]
        passed_count = sum(1 for r in results if r["overall_pass"])

        errors = [r["avg_error_px"] for r in results]
        fps_vals = [r["avg_fps"] for r in results]
        loss_vals = [r["target_loss_pct"] for r in results]

        return {
            "total_runs": count,
            "passed": passed_count,
            "failed": count - passed_count,
            "pass_rate_pct": (passed_count / max(1, count)) * 100.0,
            "best_case": best_case,
            "average_case": {
                "avg_error_px": float(np.mean(errors)),
                "avg_fps": float(np.mean(fps_vals)),
                "target_loss_pct": float(np.mean(loss_vals))
            },
            "worst_case": worst_case,
            "runs": results
        }
