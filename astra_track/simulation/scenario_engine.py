"""
Scenario Engine orchestrating virtual environment, beacon motion, optical camera rendering, disturbances, and ground truth.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
from astra_track.app.config import AstraConfig
from astra_track.core.frame import GroundTruth
from astra_track.simulation.environment import VirtualEnvironment
from astra_track.simulation.beacon import Beacon
from astra_track.simulation.trajectories import create_trajectory, TrajectoryBase
from astra_track.simulation.virtual_camera import VirtualCamera
from astra_track.simulation.virtual_ptu import VirtualPTU
from astra_track.simulation.noise import NoiseSimulator
from astra_track.simulation.atmosphere import AtmosphereSimulator
from astra_track.simulation.disturbances import DisturbanceSimulator


class ScenarioEngine:
    """End-to-end synthetic optical scene simulation and sensor frame synthesizer."""

    def __init__(self, config: AstraConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.frame_id = 0
        self.sim_time = 0.0
        self.dt = 1.0 / float(config.camera.fps)

        # Occultation / target loss override
        self.occultation_frames_remaining = 0

        # Subsystems
        self.env = VirtualEnvironment(config.environment)
        self.origin = (config.environment.width / 2.0, config.environment.height / 2.0)
        
        self.beacon = Beacon(
            id=1,
            size=config.beacon.size,
            brightness=config.beacon.brightness,
            shape=config.beacon.shape,
            visible=True,
            is_designated=True,
            modulation_frequency=config.beacon.modulation_frequency,
            modulation_depth=config.beacon.modulation_depth
        )

        self.trajectory: TrajectoryBase = create_trajectory(
            config.trajectory,
            origin=self.origin,
            world_bounds=(config.environment.width, config.environment.height)
        )

        self.camera = VirtualCamera(config.camera, world_origin=self.origin)
        self.ptu = VirtualPTU(config.camera)
        self.noise = NoiseSimulator(config.noise, seed=seed)
        self.atmosphere = AtmosphereSimulator(config.atmosphere, seed=seed + 1)
        self.disturbances = DisturbanceSimulator(config.disturbance, seed=seed + 2)

    def reset(self):
        self.frame_id = 0
        self.sim_time = 0.0
        self.occultation_frames_remaining = 0
        self.beacon.visible = True
        self.trajectory.reset()
        self.ptu.reset()
        self.disturbances.reset()

    def trigger_target_loss(self, duration_frames: int = 30):
        """Simulate temporary optical blockage / beam loss."""
        self.occultation_frames_remaining = duration_frames

    def step(self, pan_rate_cmd: float = 0.0, tilt_rate_cmd: float = 0.0) -> Tuple[np.ndarray, GroundTruth, np.ndarray]:
        """
        Advance simulation by one frame interval dt.
        Returns:
            (sensor_frame, ground_truth, world_map)
        """
        self.frame_id += 1
        self.sim_time += self.dt

        # 1. Update PTU with commanded motor rates
        current_pan, current_tilt = self.ptu.step_velocity(pan_rate_cmd, tilt_rate_cmd, self.dt)

        # 2. Update Beacon Trajectory
        bx, by, bvx, bvy, bax, bay = self.trajectory.step(self.sim_time)
        self.beacon.x = bx
        self.beacon.y = by
        self.beacon.vx = bvx
        self.beacon.vy = bvy
        self.beacon.ax = bax
        self.beacon.ay = bay

        # 3. Handle Occultation / Target Loss
        is_visible = True
        if self.occultation_frames_remaining > 0:
            self.occultation_frames_remaining -= 1
            is_visible = False
        self.beacon.visible = is_visible

        # 4. Update Platform Disturbance and Camera Jitter
        platform_dx, platform_dy = self.disturbances.step_platform(self.dt, self.sim_time)
        jitter_dx, jitter_dy = self.disturbances.get_jitter(self.sim_time)

        # 5. Project Beacon to Camera Sensor
        # Platform movement shifts the perceived relative target position
        effective_bx = bx + platform_dx
        effective_by = by + platform_dy

        cam_u, cam_v, in_fov = self.camera.world_to_sensor(effective_bx, effective_by, current_pan, current_tilt)
        # Apply camera mounting jitter to image plane
        cam_u += jitter_dx
        cam_v += jitter_dy
        # Re-evaluate in_fov after jitter
        in_fov = (0.0 <= cam_u < self.camera.width) and (0.0 <= cam_v < self.camera.height)

        # 6. Render Sensor Frame (640x480 grayscale)
        sensor_frame = np.zeros((self.camera.height, self.camera.width), dtype=np.uint8)
        # Baseline electronic dark current (typical 10-15 DN)
        sensor_frame[:, :] = 12

        if in_fov and is_visible:
            patch = self.beacon.render_patch(sim_time=self.sim_time)
            ph, pw = patch.shape
            half_w = pw // 2
            half_h = ph // 2

            cu = int(round(cam_u))
            cv = int(round(cam_v))

            x1 = max(0, cu - half_w)
            y1 = max(0, cv - half_h)
            x2 = min(self.camera.width, cu + half_w + 1)
            y2 = min(self.camera.height, cv + half_h + 1)

            px1 = x1 - (cu - half_w)
            py1 = y1 - (cv - half_h)
            px2 = px1 + (x2 - x1)
            py2 = py1 + (y2 - y1)

            if x2 > x1 and y2 > y1 and px2 > px1 and py2 > py1:
                # Additive optical power on sensor
                target_region = sensor_frame[y1:y2, x1:x2].astype(np.float32)
                beacon_region = patch[py1:py2, px1:px2].astype(np.float32)
                blended = np.clip(target_region + beacon_region, 0, 255).astype(np.uint8)
                sensor_frame[y1:y2, x1:x2] = blended

        # 7. Apply Environmental Degradation (Atmosphere & Noise)
        degraded_frame = self.atmosphere.apply(sensor_frame)
        noisy_frame = self.noise.apply(degraded_frame)

        # 8. Assemble Exact Ground Truth
        gt = GroundTruth(
            world_x=bx,
            world_y=by,
            world_vx=bvx,
            world_vy=bvy,
            world_ax=bax,
            world_ay=bay,
            camera_x=cam_u if (in_fov and is_visible) else -1.0,
            camera_y=cam_v if (in_fov and is_visible) else -1.0,
            in_fov=in_fov and is_visible,
            visible=is_visible,
            ptu_pan=current_pan,
            ptu_tilt=current_tilt
        )

        # 9. Render World Map Overview
        fov_bounds = self.camera.get_world_fov_bounds(current_pan, current_tilt)
        world_map = self.env.render_world_map(
            beacon_world_pos=(bx, by),
            trajectory_trail=self.trajectory.trail,
            fov_bounds=fov_bounds,
            camera_origin=self.origin
        )

        return noisy_frame, gt, world_map
