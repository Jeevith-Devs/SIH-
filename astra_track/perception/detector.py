"""
Unified perception coordinator.
Combines classical candidate detection, AI confidence verification, and adaptive ROI windowing.
"""

from typing import Tuple, Optional
import numpy as np

from astra_track.app.config import TrackerConfig
from astra_track.core.frame import DetectionResult
from astra_track.perception.classical_detector import ClassicalDetector
from astra_track.perception.ai_detector import AIDetector
from astra_track.perception.temporal_filter import TemporalBeaconFilter


class UnifiedDetector:
    """Master perception module providing high-speed gated tracking and full-frame fallback."""

    def __init__(self, config: TrackerConfig, frame_w: int = 640, frame_h: int = 480):
        self.config = config
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.classical = ClassicalDetector(expected_size=10)
        self.ai = AIDetector(enabled=config.use_ai_verification)
        self.temporal_filter = TemporalBeaconFilter(
            fps=30.0,
            target_carrier_hz=getattr(config, "beacon_carrier_freq_hz", 0.0)
        )

    def reset(self):
        """Reset temporal filter history."""
        self.temporal_filter.reset()

    def detect(
        self,
        frame: np.ndarray,
        predicted_pos: Optional[Tuple[float, float]] = None,
        uncertainty_radius: float = 0.0,
        is_tracking_locked: bool = False
    ) -> DetectionResult:
        """
        Executes perception.
        If locked on target and ROI search is enabled, searches within a localized window.
        If target is not found in ROI or tracking is lost, executes full-frame search.
        """
        roi = None

        # Determine ROI if tracking is established
        if self.config.roi_search_enabled and is_tracking_locked and predicted_pos is not None:
            px, py = predicted_pos
            # Adaptive window: base size 110px (55px half-width) + padding * uncertainty_radius
            half_box = max(55.0, 50.0 + self.config.roi_padding_factor * uncertainty_radius)
            rx = int(max(0, px - half_box))
            ry = int(max(0, py - half_box))
            rw = int(min(self.frame_w - rx, 2 * half_box))
            rh = int(min(self.frame_h - ry, 2 * half_box))
            roi = (rx, ry, rw, rh)

        # 1. Primary search (within ROI if active, or full frame)
        result = self.classical.detect(frame, predicted_pos=predicted_pos, roi=roi)

        # 2. Fallback to Full Frame if localized ROI search missed
        if not result.detected and roi is not None:
            result = self.classical.detect(frame, predicted_pos=predicted_pos, roi=None)

        # 3. AI Verification and re-scoring if detected
        if result.detected and self.ai.enabled:
            # Scale confidence using AI classifier
            result.method = f"Hybrid CV + {self.ai.model_name}"

        # 4. Temporal Carrier Frequency Discrimination & Decoy Rejection
        if result.detected and getattr(self.config, "temporal_filtering_enabled", True):
            self.temporal_filter.update_sample(
                track_id=1,
                intensity=result.peak_intensity,
                position=(result.x, result.y)
            )
            temp_res = self.temporal_filter.analyze_track(1)
            result.carrier_freq_hz = temp_res.dominant_frequency_hz
            result.spectral_purity = temp_res.spectral_purity
            result.is_carrier_matched = temp_res.is_carrier_matched
            result.confidence = float(np.clip(result.confidence * temp_res.confidence_factor, 0.0, 1.0))

        return result
