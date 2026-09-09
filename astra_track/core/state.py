"""
Finite State Machine managing optical beacon tracking lifecycle.
States: INITIALIZING -> SEARCHING -> ACQUIRING -> TRACKING -> PREDICTING -> TARGET_LOST -> REACQUIRING -> LOCK_RESTORED -> ERROR
"""

import time
from typing import List, Tuple, Callable, Optional
from astra_track.app.constants import TrackingStatus


class TrackingFSM:
    """Deterministic state machine governing FSOC coarse alignment tracking."""

    def __init__(
        self,
        acquisition_hits_threshold: int = 3,
        loss_misses_threshold: int = 5,
        confidence_threshold: float = 0.50,
        reacquisition_timeout_s: float = 2.0,
    ):
        self.status = TrackingStatus.INITIALIZING
        self.acquisition_hits_threshold = acquisition_hits_threshold
        self.loss_misses_threshold = loss_misses_threshold
        self.confidence_threshold = confidence_threshold
        self.reacquisition_timeout_s = reacquisition_timeout_s

        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.state_enter_time = time.time()
        self.sim_time = 0.0

        # Timing analytics
        self.initial_acquisition_start_time: Optional[float] = None
        self.initial_acquisition_duration: Optional[float] = None
        self.current_loss_start_time: Optional[float] = None
        self.last_reacquisition_duration: Optional[float] = None
        self.total_reacquisition_durations: List[float] = []
        self.target_loss_count: int = 0
        self.total_frames: int = 0
        self.locked_frames: int = 0

        self.history: List[Tuple[float, TrackingStatus, str]] = []
        self._record_transition(TrackingStatus.INITIALIZING, "System initialized")

    def _record_transition(self, new_status: TrackingStatus, reason: str):
        prev = self.status
        self.status = new_status
        self.state_enter_time = self.sim_time
        self.history.append((self.sim_time, new_status, f"{prev.value} -> {new_status.value}: {reason}"))

    def reset(self):
        self.status = TrackingStatus.INITIALIZING
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.sim_time = 0.0
        self.initial_acquisition_start_time = None
        self.initial_acquisition_duration = None
        self.current_loss_start_time = None
        self.last_reacquisition_duration = None
        self.total_reacquisition_durations.clear()
        self.target_loss_count = 0
        self.total_frames = 0
        self.locked_frames = 0
        self.history.clear()
        self._record_transition(TrackingStatus.INITIALIZING, "System reset")

    def update(self, detected: bool, confidence: float, dt: float) -> TrackingStatus:
        """
        Evaluate single frame observation and transition FSM states.
        """
        self.sim_time += dt
        self.total_frames += 1

        is_valid_detection = detected and (confidence >= self.confidence_threshold)

        if is_valid_detection:
            self.consecutive_hits += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_hits = 0

        # State transition logic
        if self.status == TrackingStatus.INITIALIZING:
            if self.initial_acquisition_start_time is None:
                self.initial_acquisition_start_time = self.sim_time
            self._record_transition(TrackingStatus.SEARCHING, "Started optical search")

        elif self.status == TrackingStatus.SEARCHING:
            if is_valid_detection:
                self._record_transition(TrackingStatus.ACQUIRING, f"Candidate spot detected (conf={confidence:.2f})")

        elif self.status == TrackingStatus.ACQUIRING:
            if is_valid_detection:
                if self.consecutive_hits >= self.acquisition_hits_threshold:
                    self._record_transition(TrackingStatus.TRACKING, f"Coarse lock established ({self.consecutive_hits} hits)")
                    if self.initial_acquisition_duration is None and self.initial_acquisition_start_time is not None:
                        self.initial_acquisition_duration = max(0.01, self.sim_time - self.initial_acquisition_start_time)
            else:
                self._record_transition(TrackingStatus.SEARCHING, "Candidate spot lost before lock confirmation")

        elif self.status in (TrackingStatus.TRACKING, TrackingStatus.LOCK_RESTORED):
            self.locked_frames += 1
            if is_valid_detection:
                # Normal healthy tracking
                pass
            else:
                if self.consecutive_misses >= self.loss_misses_threshold:
                    self.target_loss_count += 1
                    self.current_loss_start_time = self.sim_time
                    self._record_transition(TrackingStatus.TARGET_LOST, f"Target lost: {self.consecutive_misses} consecutive misses")
                else:
                    self._record_transition(TrackingStatus.PREDICTING, f"Missed frame, running Kalman prediction ({self.consecutive_misses} misses)")

        elif self.status == TrackingStatus.PREDICTING:
            if is_valid_detection:
                self.locked_frames += 1
                self._record_transition(TrackingStatus.TRACKING, "Target redetected on predicted path")
            else:
                if self.consecutive_misses >= self.loss_misses_threshold:
                    self.target_loss_count += 1
                    self.current_loss_start_time = self.sim_time
                    self._record_transition(TrackingStatus.TARGET_LOST, f"Target lost after {self.consecutive_misses} misses")

        elif self.status == TrackingStatus.TARGET_LOST:
            self._record_transition(TrackingStatus.REACQUIRING, "Initiated multi-stage predictive reacquisition")

        elif self.status == TrackingStatus.REACQUIRING:
            if is_valid_detection:
                if self.current_loss_start_time is not None:
                    reacq_time = max(0.01, self.sim_time - self.current_loss_start_time)
                    self.last_reacquisition_duration = reacq_time
                    self.total_reacquisition_durations.append(reacq_time)
                    self.current_loss_start_time = None
                self.locked_frames += 1
                self._record_transition(TrackingStatus.LOCK_RESTORED, f"Lock restored via predictive search in {self.last_reacquisition_duration or 0.0:.2f}s")
            else:
                # Check timeout for reacquisition
                time_in_reacq = self.sim_time - self.state_enter_time
                if time_in_reacq > self.reacquisition_timeout_s:
                    self._record_transition(TrackingStatus.SEARCHING, "Reacquisition window timed out, fallback to wide search")

        return self.status

    @property
    def is_locked(self) -> bool:
        return self.status in (TrackingStatus.TRACKING, TrackingStatus.LOCK_RESTORED, TrackingStatus.PREDICTING)

    @property
    def lock_retention_pct(self) -> float:
        if self.total_frames == 0:
            return 100.0
        return (self.locked_frames / self.total_frames) * 100.0

    @property
    def average_reacquisition_time(self) -> float:
        if not self.total_reacquisition_durations:
            return 0.0
        return sum(self.total_reacquisition_durations) / len(self.total_reacquisition_durations)
