"""
Session manager recording real-time telemetry series and compiling simulation run metrics.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np

from astra_track.core.frame import ProcessedFrame
from astra_track.metrics.accuracy import AccuracyEvaluator
from astra_track.metrics.performance import PerformanceEvaluator
from astra_track.metrics.fps import FPSProfiler
from astra_track.metrics.report import PerformanceReportGenerator
from astra_track.storage.database import DatabaseManager


class SessionRecorder:
    """Collects runtime telemetry series for plotting, analysis, and database logging."""

    def __init__(self, scenario_name: str = "Simulation", db_manager: Optional[DatabaseManager] = None):
        self.scenario_name = scenario_name
        self.db = db_manager if db_manager else DatabaseManager()

        # Telemetry Time Series
        self.timestamps: List[float] = []
        self.errors_px: List[float] = []
        self.fps_series: List[float] = []
        self.pan_series: List[float] = []
        self.tilt_series: List[float] = []
        self.confidence_series: List[float] = []
        
        self.gt_points: List[Dict[str, float]] = []
        self.tracked_points: List[Dict[str, float]] = []
        self.predicted_points: List[Dict[str, float]] = []

        self.frames_log: List[Dict[str, Any]] = []
        self.fps_profiler = FPSProfiler()

        self.total_frames = 0
        self.lost_frames = 0
        self.locked_frames = 0

        self.acquisition_time_s: Optional[float] = None
        self.reacquisition_times_s: List[float] = []

    def record_frame(self, proc: ProcessedFrame, log_frame_detail: bool = True):
        self.total_frames += 1
        self.timestamps.append(proc.timestamp)
        self.errors_px.append(proc.tracking_error_px)
        self.fps_series.append(proc.fps)
        self.pan_series.append(proc.ptu_pan)
        self.tilt_series.append(proc.ptu_tilt)
        self.confidence_series.append(proc.detection.confidence)

        self.fps_profiler.record(proc.fps, proc.processing_time_ms)

        if "TRACK" in proc.tracking_status.value or "LOCK" in proc.tracking_status.value or "PREDICT" in proc.tracking_status.value:
            self.locked_frames += 1
        elif "LOST" in proc.tracking_status.value:
            self.lost_frames += 1

        if proc.ground_truth:
            self.gt_points.append({"t": proc.timestamp, "x": proc.ground_truth.camera_x, "y": proc.ground_truth.camera_y})
        self.tracked_points.append({"t": proc.timestamp, "x": proc.kalman.x, "y": proc.kalman.y})
        if proc.prediction.predicted_points:
            p0 = proc.prediction.predicted_points[0]
            self.predicted_points.append({"t": proc.timestamp, "x": p0[0], "y": p0[1]})

        if log_frame_detail:
            self.frames_log.append({
                "frame_id": proc.frame_id,
                "timestamp": proc.timestamp,
                "actual_x": proc.ground_truth.camera_x if proc.ground_truth else -1.0,
                "actual_y": proc.ground_truth.camera_y if proc.ground_truth else -1.0,
                "detected_x": proc.detection.x if proc.detection.detected else -1.0,
                "detected_y": proc.detection.y if proc.detection.detected else -1.0,
                "filtered_x": proc.kalman.x,
                "filtered_y": proc.kalman.y,
                "predicted_x": proc.prediction.predicted_points[0][0] if proc.prediction.predicted_points else proc.kalman.x,
                "predicted_y": proc.prediction.predicted_points[0][1] if proc.prediction.predicted_points else proc.kalman.y,
                "pan": proc.ptu_pan,
                "tilt": proc.ptu_tilt,
                "error_px": proc.tracking_error_px,
                "confidence": proc.detection.confidence,
                "status": proc.tracking_status.value
            })

    def finalize_summary(self, fsm_state=None, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculates end-of-run statistical scorecard."""
        acc_summary = AccuracyEvaluator.compute_summary(self.errors_px)
        fps_summary = self.fps_profiler.get_summary()

        acq_time = fsm_state.initial_acquisition_duration if fsm_state else 0.8
        reacq_list = fsm_state.total_reacquisition_durations if fsm_state else []

        perf_summary = PerformanceEvaluator.evaluate(
            acquisition_time_s=acq_time,
            reacquisition_times_s=reacq_list,
            total_frames=self.total_frames,
            lost_frames=self.lost_frames,
            locked_frames=self.locked_frames
        )

        is_reacq = ("reacquisition" in self.scenario_name.lower()) or ("stress" in self.scenario_name.lower() and perf_summary["avg_reacquisition_time_s"] > 0)
        overall_pass, criteria_verdicts = PerformanceReportGenerator.evaluate_benchmark(
            acq_time_s=perf_summary["acquisition_time_s"],
            avg_err_px=acc_summary["avg_error_px"],
            loss_pct=perf_summary["target_loss_pct"],
            reacq_time_s=perf_summary["avg_reacquisition_time_s"],
            avg_fps=fps_summary["avg_fps"],
            retention_pct=perf_summary["lock_retention_pct"],
            is_reacquisition_test=is_reacq
        )

        duration = self.timestamps[-1] if self.timestamps else 0.0

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": self.scenario_name,
            "duration_s": duration,
            "resolution": "640x480",
            "overall_pass": overall_pass,
            "criteria": criteria_verdicts,
            "config": config_dict or {},
            **acc_summary,
            **perf_summary,
            **fps_summary
        }

        # Save to SQLite
        try:
            self.db.save_simulation(summary, self.frames_log)
        except Exception as e:
            print(f"[Database Error]: {e}")

        return summary
