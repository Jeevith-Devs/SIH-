"""
Unit tests for target loss detection and multi-stage reacquisition.
"""

from astra_track.core.state import TrackingFSM
from astra_track.app.constants import TrackingStatus
from astra_track.reacquisition.search_manager import ReacquisitionManager, ReacquisitionStage
from astra_track.core.frame import KalmanState


def test_target_loss_and_recovery():
    fsm = TrackingFSM(acquisition_hits_threshold=3, loss_misses_threshold=5)
    
    # 1. Acquire track with 3 hits
    for _ in range(3):
        fsm.update(detected=True, confidence=0.90, dt=0.033)
    assert fsm.status == TrackingStatus.TRACKING

    # 2. Miss 4 frames -> status enters PREDICTING
    for _ in range(4):
        fsm.update(detected=False, confidence=0.0, dt=0.033)
    assert fsm.status == TrackingStatus.PREDICTING

    # 3. Miss 5th frame -> declares TARGET_LOST
    fsm.update(detected=False, confidence=0.0, dt=0.033)
    assert fsm.status == TrackingStatus.TARGET_LOST

    # 4. Next tick transitions to REACQUIRING
    fsm.update(detected=False, confidence=0.0, dt=0.033)
    assert fsm.status == TrackingStatus.REACQUIRING

    # 5. Redetected on predicted path -> LOCK_RESTORED
    fsm.update(detected=True, confidence=0.88, dt=0.033)
    assert fsm.status == TrackingStatus.LOCK_RESTORED
    assert fsm.last_reacquisition_duration is not None
    assert fsm.last_reacquisition_duration <= 1.0


def test_reacquisition_manager_stages():
    mgr = ReacquisitionManager(frame_w=640, frame_h=480)
    mgr.start_reacquisition()
    k_state = KalmanState(x=320, y=240, uncertainty_radius=4.0)

    # Initial tick (< 0.25s) is PREDICTED_ROI
    roi, rates, stage = mgr.update(0.033, k_state)
    assert mgr.stage == ReacquisitionStage.PREDICTED_ROI
    assert roi is not None

    # Advance beyond 0.70s -> LOCAL_SPIRAL
    for _ in range(25):
        roi, rates, stage = mgr.update(0.033, k_state)
    assert mgr.stage == ReacquisitionStage.LOCAL_SPIRAL
    assert roi is None
