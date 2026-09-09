"""
Constants and Enums for ASTRA-Track FSOC Virtual Tracking System.
"""

from enum import Enum


class TrackingStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    TARGET_LOST = "TARGET_LOST"
    REACQUIRING = "REACQUIRING"
    LOCK_RESTORED = "LOCK_RESTORED"
    ERROR = "ERROR"


class ControlMode(str, Enum):
    REACTIVE = "Reactive"
    PREDICTIVE = "Predictive"
    ADAPTIVE = "Adaptive"


class AtmosphereMode(str, Enum):
    CLEAR = "Clear"
    HAZE = "Haze"
    FOG = "Fog"
    RAIN = "Rain"
    LOW_LIGHT = "Low Light"


class TrajectoryType(str, Enum):
    STRAIGHT = "Straight Line"
    CIRCULAR = "Circular"
    FIGURE8 = "Figure-8"
    RANDOM = "Random Movement"
    SPIRAL = "Spiral"
    SINUSOIDAL = "Sinusoidal"
    CUSTOM = "Custom Waypoints"


class DisturbancePattern(str, Enum):
    RANDOM = "Random"
    LINEAR = "Linear"
    CIRCULAR = "Circular"
    SINUSOIDAL = "Sinusoidal"


class CentroidMethod(str, Enum):
    GEOMETRIC = "Geometric Centroid"
    INTENSITY_WEIGHTED = "Intensity-Weighted Centroid"


class SIHBenchmarks:
    """Smart India Hackathon 2026 Target Criteria for PS 26169"""
    MAX_ACQUISITION_TIME_SEC = 2.0
    MAX_TRACKING_ERROR_PX = 10.0
    MAX_TARGET_LOSS_PCT = 5.0
    MAX_REACQUISITION_TIME_SEC = 1.0
    MIN_PROCESSING_FPS = 20.0
    MIN_LOCK_RETENTION_PCT = 95.0


# Visual Palette - Aerospace Dark Technical Aesthetic
PALETTE = {
    "bg_dark": "#0d1117",
    "bg_panel": "#161b22",
    "bg_card": "#21262d",
    "border": "#30363d",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    
    # Status semantics
    "target_detected": "#3fb950",   # Green
    "target_tracked": "#238636",    # Deep Green
    "target_predicted": "#58a6ff",   # Cyan / Blue
    "target_lost": "#f85149",        # Crimson Red
    "reacquiring": "#d29922",        # Amber Warning
    "lock_restored": "#2ea043",      # Bright Emerald
    "camera_fov": "#79c0ff",         # Sky Blue
    "camera_center": "#f0883e",      # Precision Orange
    "trajectory_trail": "#a371f7",   # Aerospace Purple
    "uncertainty_ellipse": "#e3b341",# Gold Yellow
    "accent": "#1f6feb",            # Professional Blue
}
