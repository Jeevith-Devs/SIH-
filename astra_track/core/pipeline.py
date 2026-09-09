"""
Closed-loop tracking pipeline coordinator.
Integrates frame acquisition, detection, kinematic estimation, predictive control,
FSM state updates, reacquisition, and real-time telemetry logging.
"""

import time
import math
from typing import Optional, Tuple, List
import numpy as np

from astra_track.app.config import AstraConfig
from astra_track.app.constants import TrackingStatus, ControlMode
from astra_track.core.frame import ProcessedFrame, GroundTruth, ControlCommand, CentroidResult
from astra_track.core.state import TrackingFSM
from astra_track.core.events import global_event_bus
from astra_track.simulation.scenario_engine import ScenarioEngine
from astra_track.perception.detector import UnifiedDetector
from astra_track.tracking.target_manager import TargetManager
from astra_track.control.predictive_controller import PredictiveController
from astra_track.control.pan_tilt_controller import PanTiltController
from astra_track.reacquisition.search_manager import ReacquisitionManager


class TrackingPipeline:
    """Master orchestrator executing the closed-loop tracking cycle."""

    def __init__(self, config: AstraConfig, seed: int = 42):
        self.config = config
        self.dt = 1.0 / float(config.camera.fps)

        # Core subsystems
        self.sim_engine = ScenarioEngine(config, seed=seed)
        self.detector = UnifiedDetector(config.tracker, config.camera.width, config.camera.height)
        self.target_mgr = TargetManager(config.tracker, dt=self.dt)
        self.fsm = TrackingFSM(
            acquisition_hits_threshold=3,
            loss_misses_threshold=config.tracker.loss_threshold_frames,
            confidence_threshold=config.tracker.confidence_threshold
        )
        self.reacq_mgr = ReacquisitionManager(config.camera.width, config.camera.height)

        self.pred_controller = PredictiveController(config.controller, config.camera)
        self.reactive_controller = PanTiltController(config.controller, config.camera)

        # Bind trajectory model to predictor for pattern-following reacquisition
        if hasattr(self.sim_engine, "trajectory"):
            self.target_mgr.predictor.set_trajectory_model(self.sim_engine.trajectory)

        # Active state
        self.last_cmd = ControlCommand()
        self.frame_trail: List[Tuple[float, float]] = []
        self.max_trail = 120

        # Performance timing
        self.last_tick_time = time.perf_counter()
        self.fps_rolling = float(config.camera.fps)
        self.live_frame_id = 0

    def reset(self):
        self.sim_engine.reset()
        self.target_mgr.reset()
        self.fsm.reset()
        self.reacq_mgr.reset()
        self.last_cmd = ControlCommand()
        self.frame_trail.clear()
        self.last_tick_time = time.perf_counter()
        self.live_frame_id = 0

    def process_simulation_tick(self) -> ProcessedFrame:
        """
        Executes one complete simulation cycle.
        """
        t0 = time.perf_counter()

        # 1. Step simulation with current control rates
        raw_sensor_frame, gt, world_map = self.sim_engine.step(
            pan_rate_cmd=self.last_cmd.pan_rate_cmd_deg_s,
            tilt_rate_cmd=self.last_cmd.tilt_rate_cmd_deg_s
        )

        # 2. Reacquisition check
        reacq_roi = None
        scan_rates = (0.0, 0.0)
        if self.fsm.status in (TrackingStatus.TARGET_LOST, TrackingStatus.REACQUIRING):
            reacq_roi, scan_rates, stage_str = self.reacq_mgr.update(self.dt, self.target_mgr.last_kalman_state)

        # 3. Perception detection step
        predicted_pt = None
        if self.target_mgr.kalman.initialized:
            k = self.target_mgr.last_kalman_state
            predicted_pt = (k.x + k.vx * self.dt, k.y + k.vy * self.dt)

        detection = self.detector.detect(
            raw_sensor_frame,
            predicted_pos=predicted_pt,
            uncertainty_radius=self.target_mgr.last_kalman_state.uncertainty_radius,
            is_tracking_locked=self.fsm.is_locked
        )

        # 4. Tracking and Prediction update
        k_state, prediction, is_track_valid = self.target_mgr.process_measurement(detection, dt=self.dt)

        # 5. FSM State Machine update
        status = self.fsm.update(detection.detected, detection.confidence, self.dt)

        if status == TrackingStatus.TARGET_LOST and self.reacq_mgr.stage.value == "IDLE":
            self.reacq_mgr.start_reacquisition()
        elif status in (TrackingStatus.TRACKING, TrackingStatus.LOCK_RESTORED):
            self.reacq_mgr.reset()

        # 6. Compute Control Command for next tick
        cx = self.config.camera.width / 2.0
        cy = self.config.camera.height / 2.0

        if self.fsm.status in (TrackingStatus.TARGET_LOST, TrackingStatus.REACQUIRING):
            # Dynamic pattern-following coasting along satellite's orbital trajectory
            if hasattr(self.sim_engine, "trajectory") and self.sim_engine.trajectory is not None:
                t_cur = self.sim_engine.sim_time
                bx, by, bvx, bvy, _, _ = self.sim_engine.trajectory.get_state(t_cur)
                wp_deg = getattr(self.sim_engine.camera, "world_px_per_deg", 100.0)
                cam_cx = self.sim_engine.origin[0] + self.sim_engine.ptu.pan_deg * wp_deg
                cam_cy = self.sim_engine.origin[1] + self.sim_engine.ptu.tilt_deg * wp_deg
                vel_pan = bvx / wp_deg
                vel_tilt = bvy / wp_deg
                err_pan = (bx - cam_cx) / wp_deg
                err_tilt = (by - cam_cy) / wp_deg
                coast_pan = float(np.clip(vel_pan + 1.2 * err_pan, -4.5, 4.5))
                coast_tilt = float(np.clip(vel_tilt + 1.2 * err_tilt, -4.5, 4.5))
            else:
                deg_px_x = self.config.camera.fov_horizontal / float(self.config.camera.width)
                deg_px_y = self.config.camera.fov_vertical / float(self.config.camera.height)
                coast_pan = float(np.clip(k_state.vx * deg_px_x, -3.0, 3.0))
                coast_tilt = float(np.clip(k_state.vy * deg_px_y, -3.0, 3.0))

            cmd = ControlCommand(
                pan_rate_cmd_deg_s=float(coast_pan + scan_rates[0]),
                tilt_rate_cmd_deg_s=float(coast_tilt + scan_rates[1]),
                mode="Pattern Coasting"
            )
        elif self.config.controller.mode == ControlMode.PREDICTIVE.value and self.target_mgr.is_confirmed_track:
            cmd = self.pred_controller.compute_predictive_command(prediction, k_state, cx, cy, self.dt)
        elif detection.detected:
            cmd = self.reactive_controller.compute_command(detection.x, detection.y, cx, cy, self.dt)
        elif self.target_mgr.is_confirmed_track:
            # Extrapolated prediction control for brief occlusion
            cmd = self.pred_controller.compute_predictive_command(prediction, k_state, cx, cy, self.dt)
        else:
            # Optical search mode: maintain gimbal stationarity rather than drifting
            cmd = ControlCommand(
                pan_rate_cmd_deg_s=0.0,
                tilt_rate_cmd_deg_s=0.0,
                mode="Optical Search"
            )

        self.last_cmd = cmd

        # 7. Compute Tracking Error against Ground Truth
        tracking_error_px = 0.0
        angular_error_deg = 0.0

        if gt.in_fov and gt.visible:
            if self.target_mgr.is_confirmed_track:
                tx, ty = k_state.x, k_state.y
                tracking_error_px = math.hypot(gt.camera_x - tx, gt.camera_y - ty)
            elif detection.detected:
                tx, ty = detection.x, detection.y
                tracking_error_px = math.hypot(gt.camera_x - tx, gt.camera_y - ty)
            else:
                tracking_error_px = 0.0
            
            # Angular error from camera boresight
            d_u = gt.camera_x - cx
            d_v = gt.camera_y - cy
            deg_px_x = self.config.camera.fov_horizontal / self.config.camera.width
            deg_px_y = self.config.camera.fov_vertical / self.config.camera.height
            angular_error_deg = math.hypot(d_u * deg_px_x, d_v * deg_px_y)

        # 8. Record breadcrumb trail
        if detection.detected:
            self.frame_trail.append((detection.x, detection.y))
        elif self.target_mgr.is_confirmed_track:
            self.frame_trail.append((k_state.x, k_state.y))
        if len(self.frame_trail) > self.max_trail:
            self.frame_trail.pop(0)

        # 9. FPS and latency calculation
        proc_time_ms = (time.perf_counter() - t0) * 1000.0
        cur_time = time.perf_counter()
        algo_fps = 1000.0 / max(0.5, proc_time_ms)
        interval = cur_time - self.last_tick_time
        self.last_tick_time = cur_time
        call_fps = 1.0 / max(1e-4, interval)
        effective_fps = min(algo_fps, max(call_fps, float(self.config.camera.fps))) if interval < 0.2 else float(self.config.camera.fps)
        self.fps_rolling = 0.85 * self.fps_rolling + 0.15 * effective_fps

        # Compute predicted world points along trajectory pattern
        pred_world_pts: List[Tuple[float, float]] = []
        if hasattr(self.sim_engine, "trajectory") and self.sim_engine.trajectory is not None:
            num_pts = max(5, len(prediction.predicted_points))
            for s in range(1, num_pts + 1):
                t_fut = self.sim_engine.sim_time + s * self.dt
                wbx, wby, _, _, _, _ = self.sim_engine.trajectory.get_state(t_fut)
                pred_world_pts.append((float(wbx), float(wby)))

        # Assemble ProcessedFrame
        proc_frame = ProcessedFrame(
            frame_id=self.sim_engine.frame_id,
            timestamp=self.sim_engine.sim_time,
            raw_frame=raw_sensor_frame,
            annotated_frame=None, # created dynamically when rendering
            ground_truth=gt,
            detection=detection,
            centroid=CentroidResult(
                geometric_x=detection.x,
                geometric_y=detection.y,
                intensity_x=detection.x,
                intensity_y=detection.y,
                delta_px=0.0
            ),
            kalman=k_state,
            prediction=prediction,
            control=cmd,
            tracking_status=status,
            tracking_error_px=float(tracking_error_px),
            angular_error_deg=float(angular_error_deg),
            fps=float(self.fps_rolling),
            processing_time_ms=float(proc_time_ms),
            ptu_pan=gt.ptu_pan,
            ptu_tilt=gt.ptu_tilt,
            signal_strength_pct=detection.signal_strength_pct,
            rssi_dbm=detection.rssi_dbm,
            snr_db=detection.snr_db,
            coupling_efficiency=detection.coupling_efficiency,
            world_fov_box=self.sim_engine.camera.get_world_fov_bounds(gt.ptu_pan, gt.ptu_tilt),
            search_roi=reacq_roi,
            predicted_world_points=pred_world_pts
        )

        # Broadcast event
        global_event_bus.emit("frame_processed", proc_frame, world_map)

        return proc_frame

    def process_live_frame(self, frame: np.ndarray, ptu_pan: float = 0.0, ptu_tilt: float = 0.0) -> ProcessedFrame:
        """
        Processes a single live optical frame from physical camera hardware (Webcam / GigE).
        """
        import cv2
        t0 = time.perf_counter()
        self.live_frame_id += 1

        if frame is None or frame.size == 0:
            frame = np.zeros((self.config.camera.height, self.config.camera.width), dtype=np.uint8)

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 1. Perception detection step
        predicted_pt = None
        if self.target_mgr.kalman.initialized:
            k = self.target_mgr.last_kalman_state
            predicted_pt = (k.x + k.vx * self.dt, k.y + k.vy * self.dt)

        detection = self.detector.detect(
            gray,
            predicted_pos=predicted_pt,
            uncertainty_radius=self.target_mgr.last_kalman_state.uncertainty_radius,
            is_tracking_locked=self.fsm.is_locked
        )

        # 2. Tracking and Prediction update
        k_state, prediction, is_track_valid = self.target_mgr.process_measurement(detection, dt=self.dt)

        # 3. FSM State Machine update
        status = self.fsm.update(detection.detected, detection.confidence, self.dt)

        # 4. Compute Control Command
        cx = self.config.camera.width / 2.0
        cy = self.config.camera.height / 2.0

        if self.config.controller.mode == ControlMode.PREDICTIVE.value and self.target_mgr.is_confirmed_track:
            cmd = self.pred_controller.compute_predictive_command(prediction, k_state, cx, cy, self.dt)
        elif detection.detected:
            cmd = self.reactive_controller.compute_command(detection.x, detection.y, cx, cy, self.dt)
        else:
            cmd = self.pred_controller.compute_predictive_command(prediction, k_state, cx, cy, self.dt)

        self.last_cmd = cmd

        # 5. Compute boresight alignment error (distance from optical sensor center)
        tx = k_state.x if self.target_mgr.is_confirmed_track else (detection.x if detection.detected else cx)
        ty = k_state.y if self.target_mgr.is_confirmed_track else (detection.y if detection.detected else cy)
        tracking_error_px = math.hypot(tx - cx, ty - cy) if (detection.detected or self.target_mgr.is_confirmed_track) else 0.0

        deg_px_x = self.config.camera.fov_horizontal / self.config.camera.width
        deg_px_y = self.config.camera.fov_vertical / self.config.camera.height
        angular_error_deg = math.hypot((tx - cx) * deg_px_x, (ty - cy) * deg_px_y)

        # 6. Record breadcrumb trail
        if detection.detected:
            self.frame_trail.append((detection.x, detection.y))
        elif self.target_mgr.is_confirmed_track:
            self.frame_trail.append((k_state.x, k_state.y))
        if len(self.frame_trail) > self.max_trail:
            self.frame_trail.pop(0)

        # 7. FPS & timing calculation
        proc_time_ms = (time.perf_counter() - t0) * 1000.0
        cur_time = time.perf_counter()
        instant_fps = 1.0 / max(1e-4, cur_time - self.last_tick_time)
        self.last_tick_time = cur_time
        self.fps_rolling = 0.90 * self.fps_rolling + 0.10 * instant_fps

        proc_frame = ProcessedFrame(
            frame_id=self.live_frame_id,
            timestamp=cur_time,
            raw_frame=gray,
            annotated_frame=None,
            ground_truth=None,
            detection=detection,
            centroid=CentroidResult(
                geometric_x=detection.x,
                geometric_y=detection.y,
                intensity_x=detection.x,
                intensity_y=detection.y,
                delta_px=0.0
            ),
            kalman=k_state,
            prediction=prediction,
            control=cmd,
            tracking_status=status,
            tracking_error_px=float(tracking_error_px),
            angular_error_deg=float(angular_error_deg),
            fps=float(self.fps_rolling),
            processing_time_ms=float(proc_time_ms),
            ptu_pan=ptu_pan,
            ptu_tilt=ptu_tilt,
            signal_strength_pct=detection.signal_strength_pct,
            rssi_dbm=detection.rssi_dbm,
            snr_db=detection.snr_db,
            coupling_efficiency=detection.coupling_efficiency,
            world_fov_box=(0.0, 0.0, 0.0, 0.0)
        )

        global_event_bus.emit("frame_processed", proc_frame, None)
        return proc_frame

