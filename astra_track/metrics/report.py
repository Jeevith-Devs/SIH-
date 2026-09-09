"""
Report synthesizer and SIH 2026 Problem Statement 26169 benchmark evaluator.
"""

from typing import Dict, Any, Tuple
from astra_track.app.constants import SIHBenchmarks


class PerformanceReportGenerator:
    """Aggregates metrics and determines SIH benchmark PASS/FAIL verdict."""

    @staticmethod
    def evaluate_benchmark(
        acq_time_s: float,
        avg_err_px: float,
        loss_pct: float,
        reacq_time_s: float,
        avg_fps: float,
        retention_pct: float,
        is_reacquisition_test: bool = False
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Evaluates metrics against SIH benchmarks.
        Returns:
            (overall_pass, criteria_verdicts)
        """
        c_acq = acq_time_s <= SIHBenchmarks.MAX_ACQUISITION_TIME_SEC
        c_err = avg_err_px <= SIHBenchmarks.MAX_TRACKING_ERROR_PX
        c_loss = True if is_reacquisition_test else (loss_pct <= SIHBenchmarks.MAX_TARGET_LOSS_PCT)
        c_reacq = (reacq_time_s <= SIHBenchmarks.MAX_REACQUISITION_TIME_SEC) or (reacq_time_s == 0.0)
        c_fps = avg_fps >= SIHBenchmarks.MIN_PROCESSING_FPS
        c_ret = True if is_reacquisition_test else (retention_pct >= SIHBenchmarks.MIN_LOCK_RETENTION_PCT)

        verdicts = {
            f"Acquisition (<= {SIHBenchmarks.MAX_ACQUISITION_TIME_SEC}s)": c_acq,
            f"Tracking Error (<= {SIHBenchmarks.MAX_TRACKING_ERROR_PX}px)": c_err,
            f"Target Loss (<= {SIHBenchmarks.MAX_TARGET_LOSS_PCT}%)": c_loss,
            f"Reacquisition (<= {SIHBenchmarks.MAX_REACQUISITION_TIME_SEC}s)": c_reacq,
            f"FPS (>= {SIHBenchmarks.MIN_PROCESSING_FPS})": c_fps,
            f"Lock Retention (>= {SIHBenchmarks.MIN_LOCK_RETENTION_PCT}%)": c_ret
        }

        overall_pass = all(verdicts.values())
        return overall_pass, verdicts

    @staticmethod
    def format_text_report(data: Dict[str, Any]) -> str:
        """
        Produces human-readable summary report matching Prompt specifications.
        """
        scenario = data.get("scenario", "Custom Scenario")
        duration = data.get("duration_s", 0.0)
        res = data.get("resolution", "640 x 480")
        fps = data.get("avg_fps", 30.0)
        acq = data.get("acquisition_time_s", 0.0)
        avg_err = data.get("avg_error_px", 0.0)
        max_err = data.get("max_error_px", 0.0)
        loss = data.get("target_loss_pct", 0.0)
        reacq = data.get("avg_reacquisition_time_s", 0.0)
        retention = data.get("lock_retention_pct", 100.0)
        passed = data.get("overall_pass", True)

        verdict_str = "PASS" if passed else "FAIL"

        report = f"""====================================================
ASTRA-TRACK PERFORMANCE REPORT
Hardware-Agnostic FSOC Virtual Camera Tracking System
SIH 2026 Problem Statement 26169
====================================================

Scenario:
  {scenario}

Duration:
  {duration:.1f} seconds

Resolution:
  {res}

FPS:
  {fps:.1f}

Acquisition Time:
  {acq:.2f} sec (Benchmark: <= {SIHBenchmarks.MAX_ACQUISITION_TIME_SEC}s)

Average Tracking Error:
  {avg_err:.2f} px (Benchmark: <= {SIHBenchmarks.MAX_TRACKING_ERROR_PX}px)

Maximum Tracking Error:
  {max_err:.2f} px

Target Loss:
  {loss:.1f}% (Benchmark: <= {SIHBenchmarks.MAX_TARGET_LOSS_PCT}%)

Reacquisition Time:
  {reacq:.2f} sec (Benchmark: <= {SIHBenchmarks.MAX_REACQUISITION_TIME_SEC}s)

Lock Retention:
  {retention:.1f}% (Benchmark: >= {SIHBenchmarks.MIN_LOCK_RETENTION_PCT}%)

----------------------------------------------------
Final Result:
  {verdict_str}
====================================================
"""
        return report
