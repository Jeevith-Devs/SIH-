"""
Temporal tracking performance: acquisition time, reacquisition time, lock retention, and target loss rate.
"""

from typing import List, Optional, Dict, Any


class PerformanceEvaluator:
    """Computes operational PAT (Pointing, Acquisition, Tracking) timing benchmarks."""

    @staticmethod
    def evaluate(
        acquisition_time_s: Optional[float],
        reacquisition_times_s: List[float],
        total_frames: int,
        lost_frames: int,
        locked_frames: int
    ) -> Dict[str, Any]:
        """
        Calculates performance scorecard.
        """
        acq_time = acquisition_time_s if acquisition_time_s is not None else 0.0

        if reacquisition_times_s:
            avg_reacq = sum(reacquisition_times_s) / len(reacquisition_times_s)
            max_reacq = max(reacquisition_times_s)
        else:
            avg_reacq = 0.0
            max_reacq = 0.0

        loss_pct = (lost_frames / max(1, total_frames)) * 100.0
        retention_pct = (locked_frames / max(1, total_frames)) * 100.0

        return {
            "acquisition_time_s": acq_time,
            "avg_reacquisition_time_s": avg_reacq,
            "max_reacquisition_time_s": max_reacq,
            "target_loss_pct": loss_pct,
            "lock_retention_pct": retention_pct,
            "total_frames": total_frames,
            "lost_frames": lost_frames,
            "locked_frames": locked_frames
        }
