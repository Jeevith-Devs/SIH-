"""
Data models and state representations for frames, detections, and tracking telemetry.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import numpy as np
from astra_track.app.constants import TrackingStatus, ControlMode


@dataclass
class GroundTruth:
    """True simulation state for accuracy evaluation and benchmarking."""
    world_x: float = 0.0
    world_y: float = 0.0
    world_vx: float = 0.0
    world_vy: float = 0.0
    world_ax: float = 0.0
    world_ay: float = 0.0
    camera_x: float = -1.0          # Pixel x on camera sensor (-1 if out of FOV)
    camera_y: float = -1.0          # Pixel y on camera sensor (-1 if out of FOV)
    in_fov: bool = False
    visible: bool = True            # False during simulated occultation / target loss
    ptu_pan: float = 0.0
    ptu_tilt: float = 0.0


@dataclass
class DetectionCandidate:
    """Potential beacon candidate detected by classical CV."""
    x: float
    y: float
    width: float
    height: float
    area: float
    peak_intensity: float
    mean_intensity: float
    snr: float
    circularity: float
    score: float = 0.0
    snr_db: float = 0.0
    rssi_dbm: float = -50.0


@dataclass
class DetectionResult:
    """Perception output for a single frame."""
    detected: bool = False
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = 0.0
    candidate_count: int = 0
    method: str = "Classical CV"
    execution_time_ms: float = 0.0
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) searched
    peak_intensity: float = 0.0
    carrier_freq_hz: float = 0.0
    spectral_purity: float = 0.0
    is_carrier_matched: bool = False
    snr_db: float = 0.0
    signal_strength_pct: float = 0.0
    rssi_dbm: float = -50.0
    coupling_efficiency: float = 0.0


@dataclass
class CentroidResult:
    """Centroid computation results comparing geometric vs intensity-weighted."""
    geometric_x: float = 0.0
    geometric_y: float = 0.0
    intensity_x: float = 0.0
    intensity_y: float = 0.0
    delta_px: float = 0.0       # Euclidean difference between geometric & intensity centroid


@dataclass
class KalmanState:
    """6-state kinematic filter output [x, y, vx, vy, ax, ay]."""
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    speed: float = 0.0          # sqrt(vx^2 + vy^2) px/s
    accel: float = 0.0          # sqrt(ax^2 + ay^2) px/s^2
    var_x: float = 0.0
    var_y: float = 0.0
    uncertainty_radius: float = 0.0
    innovation_x: float = 0.0
    innovation_y: float = 0.0


@dataclass
class PredictionResult:
    """N-step forward trajectory prediction."""
    horizon_frames: int = 5
    predicted_points: List[Tuple[float, float]] = field(default_factory=list)
    uncertainty_radii: List[float] = field(default_factory=list)
    lead_target_x: float = 0.0
    lead_target_y: float = 0.0


@dataclass
class ControlCommand:
    """Pan/Tilt actuator commands issued by controller."""
    pan_error_deg: float = 0.0
    tilt_error_deg: float = 0.0
    pan_rate_cmd_deg_s: float = 0.0
    tilt_rate_cmd_deg_s: float = 0.0
    mode: str = ControlMode.PREDICTIVE.value


@dataclass
class ProcessedFrame:
    """Complete container encapsulating all data for a single processed tick."""
    frame_id: int
    timestamp: float
    raw_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    ground_truth: Optional[GroundTruth] = None
    detection: DetectionResult = field(default_factory=DetectionResult)
    centroid: CentroidResult = field(default_factory=CentroidResult)
    kalman: KalmanState = field(default_factory=KalmanState)
    prediction: PredictionResult = field(default_factory=PredictionResult)
    control: ControlCommand = field(default_factory=ControlCommand)
    tracking_status: TrackingStatus = TrackingStatus.INITIALIZING
    tracking_error_px: float = 0.0
    angular_error_deg: float = 0.0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    ptu_pan: float = 0.0
    ptu_tilt: float = 0.0
    signal_strength_pct: float = 0.0
    rssi_dbm: float = -50.0
    snr_db: float = 0.0
    coupling_efficiency: float = 0.0
    world_fov_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0) # (left, top, right, bottom)
    search_roi: Optional[Tuple[int, int, int, int]] = None
    predicted_world_points: List[Tuple[float, float]] = field(default_factory=list)

