"""
Target management, candidate association gating, track validation, and lifetime supervision.
Protects against false locks on ambient noise spikes and manages track confirmation.
"""

import math
from typing import Optional, Tuple, List
from astra_track.app.config import TrackerConfig
from astra_track.core.frame import DetectionResult, KalmanState, PredictionResult
from astra_track.tracking.kalman_tracker import KalmanTracker
from astra_track.tracking.predictor import TrajectoryPredictor


class TargetManager:
    """Oversees optical track lifecycle, spatial gating, and temporal consistency."""

    def __init__(self, config: TrackerConfig, dt: float = 0.0333):
        self.config = config
        self.dt = dt

        self.kalman = KalmanTracker(dt=dt)
        self.predictor = TrajectoryPredictor(horizon_frames=config.prediction_horizon, dt=dt)

        self.is_confirmed_track = False
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.confirmation_threshold = 3
        self.loss_threshold = config.loss_threshold_frames

        self.last_kalman_state = KalmanState()
        self.last_prediction = PredictionResult()

    def reset(self):
        self.kalman.reset()
        self.is_confirmed_track = False
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.last_kalman_state = KalmanState()
        self.last_prediction = PredictionResult()

    def process_measurement(
        self,
        detection: DetectionResult,
        dt: Optional[float] = None
    ) -> Tuple[KalmanState, PredictionResult, bool]:
        """
        Executes predict-update cycle for the active track.
        Returns:
            (KalmanState, PredictionResult, is_track_valid)
        """
        # 1. Kalman Predict Step
        pred_x, pred_y = self.kalman.predict(dt=dt)

        is_valid_measurement = False

        if detection.detected and detection.confidence >= self.config.confidence_threshold:
            # Check spatial gating if track is already established
            if self.is_confirmed_track:
                gate_dist = math.hypot(detection.x - pred_x, detection.y - pred_y)
                # Max permissible jump in one frame (e.g. 150 px)
                max_gate = max(60.0, 4.0 * self.last_kalman_state.uncertainty_radius)
                if gate_dist <= max_gate:
                    is_valid_measurement = True
            else:
                is_valid_measurement = True

        # 2. Kalman Update Step
        if is_valid_measurement:
            if not self.is_confirmed_track and self.consecutive_hits == 0:
                self.kalman.init_state(detection.x, detection.y)
            self.consecutive_hits += 1
            self.consecutive_misses = 0
            if self.consecutive_hits >= self.confirmation_threshold:
                self.is_confirmed_track = True
            k_state = self.kalman.update(detection.x, detection.y)
        else:
            self.consecutive_misses += 1
            self.consecutive_hits = 0
            if self.consecutive_misses >= self.loss_threshold:
                self.is_confirmed_track = False
            k_state = self.kalman.get_state()

        # 3. Trajectory Prediction Step
        prediction = self.predictor.predict(k_state)

        self.last_kalman_state = k_state
        self.last_prediction = prediction

        return k_state, prediction, self.is_confirmed_track
