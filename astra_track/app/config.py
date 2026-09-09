"""
Configuration management for ASTRA-Track.
Supports YAML, JSON, and typed dataclass models.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import os
import yaml
import json

from astra_track.app.constants import (
    TrajectoryType, AtmosphereMode, DisturbancePattern,
    CentroidMethod, ControlMode
)


@dataclass
class EnvironmentConfig:
    width: int = 2000
    height: int = 2000
    background_stars: int = 60
    star_seed: int = 42


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_horizontal: float = 4.0   # degrees
    fov_vertical: float = 3.0     # degrees
    fps: int = 30
    max_pan_speed: float = 5.0    # deg/s (default 5.0, up to 10.0)
    max_tilt_speed: float = 5.0   # deg/s
    max_pan_accel: float = 15.0   # deg/s^2
    max_tilt_accel: float = 15.0  # deg/s^2
    pan_limit: float = 45.0       # +/- degrees from center
    tilt_limit: float = 45.0      # +/- degrees from center


@dataclass
class BeaconConfig:
    size: int = 10                # pixels (5 to 20 px)
    brightness: int = 255         # 0-255
    shape: str = "square"         # "square" or "gaussian"
    count: int = 1
    initial_x: float = 1000.0     # center of default world
    initial_y: float = 1000.0
    modulation_frequency: float = 0.0  # Hz (0.0 for continuous wave, e.g. 15.0 for pulsed beacon)
    modulation_depth: float = 1.0      # 0.0 to 1.0


@dataclass
class TrajectoryConfig:
    type: str = TrajectoryType.FIGURE8.value
    speed: float = 60.0           # pixels per second in world space
    radius: float = 350.0         # pixels for circular / spiral
    amplitude: float = 250.0      # for figure-8 / sinusoidal
    frequency: float = 0.05       # Hz
    direction: int = 1            # 1 for clockwise, -1 for counter-clockwise
    waypoints: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class NoiseConfig:
    salt_pepper: float = 0.0      # probability 0.0 - 0.3 (default 0.10 for noise tests)
    gaussian_std: float = 0.0     # standard deviation 0 - 50
    poisson: bool = False


@dataclass
class DisturbanceConfig:
    camera_jitter_amp: float = 0.0       # pixels (+/- 20 max)
    camera_jitter_pattern: str = DisturbancePattern.RANDOM.value
    platform_motion_amp: float = 0.0     # pixels/frame (+/- 20 max)
    platform_motion_pattern: str = DisturbancePattern.LINEAR.value
    platform_speed: float = 20.0         # pixels/sec


@dataclass
class AtmosphereConfig:
    mode: str = AtmosphereMode.CLEAR.value
    intensity: float = 0.0               # 0.0 to 1.0


@dataclass
class TrackerConfig:
    kalman_enabled: bool = True
    prediction_horizon: int = 5          # frames ahead
    confidence_threshold: float = 0.60
    loss_threshold_frames: int = 5       # frames of missed detection to declare TARGET_LOST
    centroid_method: str = CentroidMethod.INTENSITY_WEIGHTED.value
    roi_search_enabled: bool = True
    roi_padding_factor: float = 3.0      # multiplier for Kalman uncertainty ellipse
    reacquisition_timeout_s: float = 1.0
    use_ai_verification: bool = True
    temporal_filtering_enabled: bool = True
    beacon_carrier_freq_hz: float = 0.0  # Expected beacon carrier (0.0 = adaptive/CW)
    use_ekf: bool = False                # 3D 6-DOF IMU EKF fusion mode


@dataclass
class ControllerConfig:
    mode: str = ControlMode.PREDICTIVE.value
    kp_pan: float = 1.2
    kp_tilt: float = 1.2
    deadzone_deg: float = 0.02           # dead zone to avoid motor jitter near center
    lead_time_frames: float = 3.0        # horizon frames for predictive control lead


@dataclass
class AstraConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    beacon: BeaconConfig = field(default_factory=BeaconConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AstraConfig":
        env = EnvironmentConfig(**data.get("environment", {}))
        cam = CameraConfig(**data.get("camera", {}))
        beacon = BeaconConfig(**data.get("beacon", {}))
        traj = TrajectoryConfig(**data.get("trajectory", {}))
        noise = NoiseConfig(**data.get("noise", {}))
        dist = DisturbanceConfig(**data.get("disturbance", {}))
        atmo = AtmosphereConfig(**data.get("atmosphere", {}))
        tracker = TrackerConfig(**data.get("tracker", {}))
        ctrl = ControllerConfig(**data.get("controller", {}))
        return cls(
            environment=env, camera=cam, beacon=beacon, trajectory=traj,
            noise=noise, disturbance=dist, atmosphere=atmo,
            tracker=tracker, controller=ctrl
        )

    def to_yaml(self, file_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, file_path: str) -> "AstraConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    # Built-in Scenario Presets
    @classmethod
    def preset_easy(cls) -> "AstraConfig":
        cfg = cls()
        cfg.trajectory.type = TrajectoryType.STRAIGHT.value
        cfg.trajectory.speed = 40.0
        cfg.atmosphere.mode = AtmosphereMode.CLEAR.value
        cfg.noise.salt_pepper = 0.0
        cfg.noise.gaussian_std = 0.0
        cfg.disturbance.camera_jitter_amp = 0.0
        cfg.disturbance.platform_motion_amp = 0.0
        return cfg

    @classmethod
    def preset_moderate(cls) -> "AstraConfig":
        cfg = cls()
        cfg.trajectory.type = TrajectoryType.CIRCULAR.value
        cfg.trajectory.speed = 60.0
        cfg.trajectory.radius = 160.0
        cfg.noise.gaussian_std = 8.0
        cfg.disturbance.camera_jitter_amp = 3.0
        cfg.disturbance.camera_jitter_pattern = DisturbancePattern.RANDOM.value
        return cfg

    @classmethod
    def preset_hard(cls) -> "AstraConfig":
        cfg = cls()
        cfg.trajectory.type = TrajectoryType.FIGURE8.value
        cfg.trajectory.speed = 80.0
        cfg.trajectory.amplitude = 300.0
        cfg.noise.salt_pepper = 0.08
        cfg.atmosphere.mode = AtmosphereMode.FOG.value
        cfg.atmosphere.intensity = 0.40
        cfg.disturbance.platform_motion_amp = 8.0
        cfg.disturbance.camera_jitter_amp = 5.0
        return cfg

    @classmethod
    def preset_extreme(cls) -> "AstraConfig":
        cfg = cls()
        cfg.trajectory.type = TrajectoryType.FIGURE8.value
        cfg.trajectory.speed = 85.0
        cfg.trajectory.amplitude = 300.0
        cfg.noise.salt_pepper = 0.10
        cfg.noise.gaussian_std = 2.0
        cfg.noise.poisson = False
        cfg.atmosphere.mode = AtmosphereMode.FOG.value
        cfg.atmosphere.intensity = 0.45
        cfg.disturbance.camera_jitter_amp = 5.0
        cfg.disturbance.platform_motion_amp = 7.0
        cfg.camera.max_pan_speed = 8.0
        cfg.camera.max_tilt_speed = 8.0
        return cfg

    @classmethod
    def preset_reacquisition(cls) -> "AstraConfig":
        cfg = cls()
        cfg.trajectory.type = TrajectoryType.SINUSOIDAL.value
        cfg.trajectory.speed = 75.0
        cfg.trajectory.amplitude = 250.0
        cfg.noise.gaussian_std = 6.0
        cfg.disturbance.camera_jitter_amp = 4.0
        return cfg

    @classmethod
    def load_preset(cls, name: str) -> "AstraConfig":
        name = name.lower().strip()
        if name == "easy":
            return cls.preset_easy()
        elif name == "moderate":
            return cls.preset_moderate()
        elif name == "hard":
            return cls.preset_hard()
        elif name == "extreme":
            return cls.preset_extreme()
        elif name == "reacquisition":
            return cls.preset_reacquisition()
        else:
            return cls.preset_moderate()


# Alias for backwards compatibility
TrackingConfig = AstraConfig
