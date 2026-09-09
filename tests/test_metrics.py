"""
Unit tests for accuracy and SIH benchmark evaluation.
"""

from astra_track.metrics.accuracy import AccuracyEvaluator
from astra_track.metrics.report import PerformanceReportGenerator


def test_tracking_error_metrics():
    # Actual (320, 240), Tracked (323, 244) -> hypot(3, 4) = 5.0 px
    err = AccuracyEvaluator.compute_frame_error(320.0, 240.0, 323.0, 244.0)
    assert abs(err - 5.0) < 1e-4

    summary = AccuracyEvaluator.compute_summary([3.0, 4.0, 5.0, 6.0])
    assert abs(summary["avg_error_px"] - 4.5) < 1e-4
    assert abs(summary["max_error_px"] - 6.0) < 1e-4


def test_sih_benchmark_verdicts():
    # Case 1: Excellent performance satisfying all criteria
    passed, verdicts = PerformanceReportGenerator.evaluate_benchmark(
        acq_time_s=0.85,
        avg_err_px=4.5,
        loss_pct=1.2,
        reacq_time_s=0.62,
        avg_fps=28.5,
        retention_pct=98.8
    )
    assert passed is True
    assert all(verdicts.values())

    # Case 2: Excessive tracking error (> 10px) -> FAIL
    passed_fail, verdicts_fail = PerformanceReportGenerator.evaluate_benchmark(
        acq_time_s=0.85,
        avg_err_px=14.5,
        loss_pct=1.2,
        reacq_time_s=0.62,
        avg_fps=28.5,
        retention_pct=98.8
    )
    assert passed_fail is False
