"""
Unit tests for automated benchmark runner and stress test execution.
"""

from astra_track.metrics.benchmark_runner import BenchmarkRunner


def test_batch_benchmark():
    # Rapid 2-scenario batch test
    summary = BenchmarkRunner.run_batch(count=2, duration_per_scenario_s=1.5)
    assert summary["total_scenarios"] == 2
    assert summary["pass_rate_pct"] >= 50.0
    assert summary["avg_tracking_error_px"] <= 12.0
    assert summary["avg_fps"] >= 20.0
