"""
Unit tests for 6-state Kalman filter and forward trajectory predictor.
"""

from astra_track.tracking.kalman_tracker import KalmanTracker
from astra_track.tracking.predictor import TrajectoryPredictor


def test_kalman_convergence():
    tracker = KalmanTracker(dt=0.0333)
    # Moving beacon: x(t) = 100 + 60*t, y(t) = 200 + 30*t
    # dt = 0.0333, dx per frame = 2.0, dy per frame = 1.0
    tracker.init_state(100.0, 200.0)

    for i in range(1, 40):
        t = i * 0.0333
        meas_x = 100.0 + 60.0 * t
        meas_y = 200.0 + 30.0 * t

        tracker.predict()
        state = tracker.update(meas_x, meas_y)

    # Check converged velocity estimate
    assert abs(state.vx - 60.0) < 5.0
    assert abs(state.vy - 30.0) < 5.0
    assert state.uncertainty_radius < 5.0


def test_kalman_prediction():
    tracker = KalmanTracker(dt=0.0333)
    tracker.init_state(200.0, 300.0)
    for i in range(1, 30):
        t = i * 0.0333
        tracker.predict()
        state = tracker.update(200.0 + 90.0 * t, 300.0 - 45.0 * t)

    predictor = TrajectoryPredictor(horizon_frames=5, dt=0.0333)
    pred_res = predictor.predict(state)

    assert len(pred_res.predicted_points) == 5
    # 5 frames ahead = 5 * 0.0333 = 0.1665 s
    # expected dx ~ 90 * 0.1665 ~ 15.0 px
    p5 = pred_res.predicted_points[4]
    assert p5[0] > state.x
    assert p5[1] < state.y
    assert pred_res.uncertainty_radii[4] > pred_res.uncertainty_radii[0]
