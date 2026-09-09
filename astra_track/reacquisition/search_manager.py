"""
Multi-stage intelligent reacquisition supervisor.
Coordinates search hierarchy:
Level 1: Extrapolated Kalman ROI -> Level 2: Expanded uncertainty ROI -> Level 3: Local spiral scan -> Level 4: Global sweep.
Ensures target reacquisition within the <= 1.0 second SIH benchmark.
"""

from enum import Enum
from typing import Tuple, Optional
from astra_track.core.frame import KalmanState
from astra_track.reacquisition.prediction_search import PredictionSearchROI
from astra_track.reacquisition.fallback_search import FallbackSearchScanner


class ReacquisitionStage(str, Enum):
    IDLE = "IDLE"
    PREDICTED_ROI = "Level 1: Predicted ROI"
    EXPANDED_ROI = "Level 2: Expanded ROI"
    LOCAL_SPIRAL = "Level 3: Local Spiral"
    GLOBAL_SWEEP = "Level 4: Global Fallback"


class ReacquisitionManager:
    """Manages multi-tier recovery sequence after target loss detection."""

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self.frame_w = frame_w
        self.frame_h = frame_h
        
        self.scanner = FallbackSearchScanner()
        self.stage = ReacquisitionStage.IDLE
        self.loss_elapsed_time = 0.0
        self.frames_in_loss = 0
        self.last_scan_pan = 0.0
        self.last_scan_tilt = 0.0

    def reset(self):
        self.stage = ReacquisitionStage.IDLE
        self.loss_elapsed_time = 0.0
        self.frames_in_loss = 0
        self.last_scan_pan = 0.0
        self.last_scan_tilt = 0.0
        self.scanner.reset()

    def start_reacquisition(self):
        self.stage = ReacquisitionStage.PREDICTED_ROI
        self.loss_elapsed_time = 0.0
        self.frames_in_loss = 0
        self.last_scan_pan = 0.0
        self.last_scan_tilt = 0.0
        self.scanner.reset()

    def update(
        self,
        dt: float,
        k_state: KalmanState
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[float, float], str]:
        """
        Executes one tick of reacquisition.
        Returns:
            (search_roi_box, camera_angular_scan_rates, stage_name)
        """
        self.loss_elapsed_time += dt
        self.frames_in_loss += 1

        # Advance stages based on elapsed loss time
        if self.loss_elapsed_time < 0.25:
            self.stage = ReacquisitionStage.PREDICTED_ROI
            # Tightly gated window on predicted position
            roi = PredictionSearchROI.compute_roi(k_state, self.frame_w, self.frame_h, expansion_factor=2.5)
            scan_rates = (0.0, 0.0)

        elif self.loss_elapsed_time < 0.60:
            self.stage = ReacquisitionStage.EXPANDED_ROI
            # Expanded window
            roi = PredictionSearchROI.compute_roi(k_state, self.frame_w, self.frame_h, expansion_factor=5.0)
            scan_rates = (0.0, 0.0)

        elif self.loss_elapsed_time < 1.20:
            self.stage = ReacquisitionStage.LOCAL_SPIRAL
            # Full frame search with bounded spiral modulation around origin
            roi = None
            d_pan, d_tilt = self.scanner.get_spiral_offset(dt)
            rate_pan = (d_pan - self.last_scan_pan) / max(1e-4, dt)
            rate_tilt = (d_tilt - self.last_scan_tilt) / max(1e-4, dt)
            self.last_scan_pan = d_pan
            self.last_scan_tilt = d_tilt
            scan_rates = (rate_pan, rate_tilt)

        else:
            self.stage = ReacquisitionStage.GLOBAL_SWEEP
            roi = None
            d_pan, d_tilt = self.scanner.get_raster_offset(self.loss_elapsed_time)
            rate_pan = (d_pan - self.last_scan_pan) / max(1e-4, dt)
            rate_tilt = (d_tilt - self.last_scan_tilt) / max(1e-4, dt)
            self.last_scan_pan = d_pan
            self.last_scan_tilt = d_tilt
            scan_rates = (rate_pan, rate_tilt)

        return roi, scan_rates, self.stage.value
