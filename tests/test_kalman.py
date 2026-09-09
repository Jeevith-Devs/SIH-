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


def test_pattern_curvilinear_prediction():
    """Validates that circular / orbital patterns produce curved predictions, not straight lines."""
    import math
    tracker = KalmanTracker(dt=0.0333)
    predictor = TrajectoryPredictor(horizon_frames=10, dt=0.0333)
    
    # Circular orbital beacon motion: R=150, omega=1.2 rad/s
    R = 150.0
    omega = 1.2
    for i in range(40):
        t = i * 0.0333
        mx = 320.0 + R * math.cos(omega * t)
        my = 240.0 + R * math.sin(omega * t)
        tracker.predict()
        state = tracker.update(mx, my)
        predictor.update_history(mx, my, state.vx, state.vy, t)

    pred_res = predictor.predict(state)
    pts = pred_res.predicted_points
    assert len(pts) == 10

    # Test curvature: vector from pt 0->2 vs vector from pt 2->6
    v1_x = pts[2][0] - pts[0][0]
    v1_y = pts[2][1] - pts[0][1]
    v2_x = pts[6][0] - pts[4][0]
    v2_y = pts[6][1] - pts[4][1]

    # Cross product of tangent vectors: non-zero for curved pattern
    cross_prod = v1_x * v2_y - v1_y * v2_x
    assert abs(cross_prod) > 1.0, f"Expected curved prediction, but cross product was {cross_prod}"


def test_kalman_turn_rate_extrapolation():
    """Validates that during target loss, coordinated turn rotates velocity along the pattern."""
    tracker = KalmanTracker(dt=0.0333)
    tracker.init_state(320.0, 240.0)
    tracker.x[2, 0] = 100.0  # vx = 100 px/s
    tracker.x[3, 0] = 0.0    # vy = 0 px/s

    # Advance 15 frames with turn_rate = 1.5 rad/s
    for _ in range(15):
        tracker.predict(turn_rate=1.5)

    st = tracker.get_state()
    # Velocity must have rotated: vy should now be positive
    assert st.vy > 10.0, f"Expected velocity rotation, but vy is {st.vy}"

