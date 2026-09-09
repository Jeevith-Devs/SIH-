"""
9-State Extended Kalman Filter (EKF) with 6-DOF IMU Sensor Fusion.
Designed for Mobile FSOC Terminals (e.g. ship-to-shore, airborne, or vehicular mounts).

State Vector (9x1):
  X = [x, y, z, vx, vy, vz, phi (roll), theta (pitch), psi (yaw)]^T
Decouples terminal platform base disturbances (wave roll/pitch, vehicle vibrations)
from true target line-of-sight (LOS) kinematics, achieving inertial stabilization.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np


@dataclass
class EKFState:
    """Estimated 3D spherical and inertial platform state."""
    x: float = 0.0               # Target 3D Position X (m)
    y: float = 0.0               # Target 3D Position Y (m)
    z: float = 0.0               # Target 3D Position Z (m)
    vx: float = 0.0              # Target 3D Velocity X (m/s)
    vy: float = 0.0              # Target 3D Velocity Y (m/s)
    vz: float = 0.0              # Target 3D Velocity Z (m/s)
    roll_deg: float = 0.0        # Platform Roll (deg)
    pitch_deg: float = 0.0       # Platform Pitch (deg)
    yaw_deg: float = 0.0         # Platform Yaw (deg)
    azimuth_deg: float = 0.0     # Target Line-of-Sight Azimuth (deg)
    elevation_deg: float = 0.0   # Target Line-of-Sight Elevation (deg)
    range_m: float = 0.0         # Estimated Slant Range (m)
    uncertainty_radius: float = 1.0 # 2-sigma position uncertainty radius (m)


class ExtendedKalmanTracker:
    """
    9-State Non-Linear Extended Kalman Filter.
    Fuses camera optical bearing angles with 6-DOF platform IMU measurements.
    """

    def __init__(
        self,
        dt: float = 0.0333,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        q_att: float = 0.05,
        r_optical_deg: float = 0.05,
        r_imu_deg: float = 0.5
    ):
        self.dt = dt
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_att = q_att

        # State vector (9x1): [x, y, z, vx, vy, vz, roll, pitch, yaw]^T (angles in radians)
        self.x = np.zeros((9, 1), dtype=np.float64)

        # Covariance matrix (9x9)
        self.P = np.eye(9, dtype=np.float64) * 10.0
        # Higher initial variance for velocity and attitude
        self.P[3:6, 3:6] *= 50.0
        self.P[6:9, 6:9] *= 5.0

        # Process noise covariance Q (9x9)
        self.Q = np.zeros((9, 9), dtype=np.float64)
        for i in range(3):
            self.Q[i, i] = q_pos * (dt**2)
            self.Q[i + 3, i + 3] = q_vel * dt
            self.Q[i + 6, i + 6] = q_att * dt

        # Measurement noise covariance
        # Optical angle noise (rad^2)
        r_opt_rad = math.radians(r_optical_deg)
        self.R_optical = np.eye(2, dtype=np.float64) * (r_opt_rad**2)

        # IMU attitude noise (rad^2)
        r_imu_rad = math.radians(r_imu_deg)
        self.R_imu = np.eye(3, dtype=np.float64) * (r_imu_rad**2)

        self.initialized = False
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)

    def init_state(
        self,
        range_m: float = 1000.0,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0
    ):
        """Initializes 3D state from initial range and spherical bearing angles."""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        r = max(1.0, range_m)

        # Convert spherical (Range, Az, El) to Cartesian (X, Y, Z)
        x_init = r * math.cos(el) * math.cos(az)
        y_init = r * math.cos(el) * math.sin(az)
        z_init = r * math.sin(el)

        self.x = np.array([
            [x_init], [y_init], [z_init],
            [0.0], [0.0], [0.0],
            [math.radians(roll_deg)],
            [math.radians(pitch_deg)],
            [math.radians(yaw_deg)]
        ], dtype=np.float64)

        pos_var = max(400.0, (0.15 * r)**2)
        self.P = np.diag([
            pos_var, pos_var, pos_var,
            100.0, 100.0, 100.0,
            math.radians(2.0)**2, math.radians(2.0)**2, math.radians(2.0)**2
        ]).astype(np.float64)

        self.initialized = True

    def reset(self):
        """Reset filter state and covariance."""
        self.initialized = False
        self.x = np.zeros((9, 1), dtype=np.float64)
        self.P = np.eye(9, dtype=np.float64) * 10.0
        self.last_innovation = np.zeros((2, 1), dtype=np.float64)

    def predict(
        self,
        dt: Optional[float] = None,
        gyro_rates_rad_s: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        EKF Time Update (Prediction Step).
        Propagates 3D position and velocity with constant velocity kinematics,
        and propagates platform attitude using rate gyros if available.
        """
        step_dt = dt if dt is not None else self.dt

        # State Transition Jacobian F (9x9)
        F = np.eye(9, dtype=np.float64)
        # Position += Velocity * dt
        F[0, 3] = step_dt
        F[1, 4] = step_dt
        F[2, 5] = step_dt

        # Propagate State Mean
        # Position
        self.x[0, 0] += self.x[3, 0] * step_dt
        self.x[1, 0] += self.x[4, 0] * step_dt
        self.x[2, 0] += self.x[5, 0] * step_dt

        # Propagate Attitude using IMU rate gyro if provided
        if gyro_rates_rad_s is not None:
            p, q, r = gyro_rates_rad_s
            self.x[6, 0] += p * step_dt
            self.x[7, 0] += q * step_dt
            self.x[8, 0] += r * step_dt

        # Wrap attitude angles to [-pi, pi]
        for i in range(6, 9):
            self.x[i, 0] = (self.x[i, 0] + np.pi) % (2.0 * np.pi) - np.pi

        # Covariance propagation: P = F * P * F^T + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        return self.x.copy()

    def update_optical_los(self, measured_az_rad: float, measured_el_rad: float):
        """
        EKF Measurement Update with Optical Line-of-Sight Angles (Azimuth, Elevation).
        Measurement Model:
            h(X) = [ arctan2(y, x) - yaw,
                     arctan2(z, sqrt(x^2 + y^2)) - pitch ]^T
        """
        if not self.initialized:
            self.init_state(range_m=1000.0, azimuth_deg=math.degrees(measured_az_rad), elevation_deg=math.degrees(measured_el_rad))
            return

        x = float(self.x[0, 0])
        y = float(self.x[1, 0])
        z = float(self.x[2, 0])
        pitch = float(self.x[7, 0])
        yaw = float(self.x[8, 0])

        r_xy_sq = max(0.01, x**2 + y**2)
        r_xy = math.sqrt(r_xy_sq)
        rho_sq = max(0.01, r_xy_sq + z**2)

        # Expected spherical angles adjusted for platform attitude
        pred_az = math.atan2(y, x) - yaw
        pred_el = math.atan2(z, r_xy) - pitch

        # Wrap angle innovations
        diff_az = (measured_az_rad - pred_az + np.pi) % (2.0 * np.pi) - np.pi
        diff_el = (measured_el_rad - pred_el + np.pi) % (2.0 * np.pi) - np.pi

        y_innov = np.array([[diff_az], [diff_el]], dtype=np.float64)
        self.last_innovation = y_innov.copy()

        # Compute Jacobian H (2x9) with respect to target position [x, y, z]
        # Platform attitude is observed and stabilized by IMU
        H = np.zeros((2, 9), dtype=np.float64)
        # d(az)/dx, d(az)/dy, d(az)/dz
        H[0, 0] = -y / r_xy_sq
        H[0, 1] = x / r_xy_sq
        H[0, 2] = 0.0

        # d(el)/dx, d(el)/dy, d(el)/dz
        H[1, 0] = -(x * z) / (rho_sq * r_xy)
        H[1, 1] = -(y * z) / (rho_sq * r_xy)
        H[1, 2] = r_xy / rho_sq

        # Innovation covariance S = H * P * H^T + R
        S = np.dot(np.dot(H, self.P), H.T) + self.R_optical

        # Kalman gain K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # State correction
        self.x = self.x + np.dot(K, y_innov)

        # Joseph form covariance update
        I = np.eye(9, dtype=np.float64)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R_optical), K.T)

    def update_optical_pixels(
        self,
        u: float,
        v: float,
        fov_h_deg: float = 4.0,
        fov_v_deg: float = 3.0,
        img_w: int = 640,
        img_h: int = 480
    ):
        """Converts sensor pixel coordinate (u, v) into line-of-sight angles and updates filter."""
        norm_u = (u - img_w / 2.0) / (img_w / 2.0)
        norm_v = (v - img_h / 2.0) / (img_h / 2.0)

        los_az_rad = norm_u * math.radians(fov_h_deg / 2.0)
        los_el_rad = -norm_v * math.radians(fov_v_deg / 2.0)

        self.update_optical_los(los_az_rad, los_el_rad)

    def update_imu_attitude(self, roll_rad: float, pitch_rad: float, yaw_rad: float):
        """
        EKF Measurement Update with 3-DOF Platform Attitude from IMU.
        Directly observes the platform attitude states [roll, pitch, yaw].
        """
        if not self.initialized:
            return

        H_imu = np.zeros((3, 9), dtype=np.float64)
        H_imu[0, 6] = 1.0  # roll
        H_imu[1, 7] = 1.0  # pitch
        H_imu[2, 8] = 1.0  # yaw

        z_imu = np.array([[roll_rad], [pitch_rad], [yaw_rad]], dtype=np.float64)
        y_att = z_imu - np.dot(H_imu, self.x)

        # Wrap angle residuals
        for i in range(3):
            y_att[i, 0] = (y_att[i, 0] + np.pi) % (2.0 * np.pi) - np.pi

        S = np.dot(np.dot(H_imu, self.P), H_imu.T) + self.R_imu
        K = np.dot(np.dot(self.P, H_imu.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y_att)

        I = np.eye(9, dtype=np.float64)
        I_KH = I - np.dot(K, H_imu)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R_imu), K.T)

    def get_state(self) -> EKFState:
        """Extracts structured EKF state with spherical LOS angles and position uncertainty."""
        x = float(self.x[0, 0])
        y = float(self.x[1, 0])
        z = float(self.x[2, 0])
        vx = float(self.x[3, 0])
        vy = float(self.x[4, 0])
        vz = float(self.x[5, 0])
        roll_rad = float(self.x[6, 0])
        pitch_rad = float(self.x[7, 0])
        yaw_rad = float(self.x[8, 0])

        r_xy = math.hypot(x, y)
        range_m = math.sqrt(r_xy**2 + z**2)
        az_deg = math.degrees(math.atan2(y, x))
        el_deg = math.degrees(math.atan2(z, max(0.01, r_xy)))

        # 3D position covariance eigenvalues for uncertainty sphere
        cov_pos = self.P[0:3, 0:3]
        eigvals = np.linalg.eigvalsh(cov_pos)
        max_eig = max(0.001, float(np.max(eigvals)))
        uncertainty_radius = 2.0 * math.sqrt(max_eig)

        return EKFState(
            x=round(x, 3),
            y=round(y, 3),
            z=round(z, 3),
            vx=round(vx, 3),
            vy=round(vy, 3),
            vz=round(vz, 3),
            roll_deg=round(math.degrees(roll_rad), 3),
            pitch_deg=round(math.degrees(pitch_rad), 3),
            yaw_deg=round(math.degrees(yaw_rad), 3),
            azimuth_deg=round(az_deg, 3),
            elevation_deg=round(el_deg, 3),
            range_m=round(range_m, 2),
            uncertainty_radius=round(uncertainty_radius, 3)
        )
