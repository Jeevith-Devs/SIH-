"""
Virtual camera model for FSOC terminal tracking simulation.
Projects world space coordinates to sensor pixel space via PTU pan/tilt angles.
"""

from typing import Tuple, Optional
import numpy as np
from astra_track.app.config import CameraConfig


class VirtualCamera:
    """Simulated narrow-divergence optical receiver camera."""

    def __init__(self, config: CameraConfig, world_origin: Tuple[float, float] = (1000.0, 1000.0), world_px_per_deg: float = 100.0):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.fov_h = config.fov_horizontal
        self.fov_v = config.fov_vertical
        self.fps = config.fps

        self.origin_x, self.origin_y = world_origin
        self.world_px_per_deg = world_px_per_deg

        # World extent of camera FOV
        self.fov_world_w = self.fov_h * self.world_px_per_deg
        self.fov_world_h = self.fov_v * self.world_px_per_deg

        # Angular conversion factors
        self.deg_per_px_x = self.fov_h / float(self.width)
        self.deg_per_px_y = self.fov_v / float(self.height)
        self.px_per_deg_x = float(self.width) / self.fov_h
        self.px_per_deg_y = float(self.height) / self.fov_v

    def get_world_fov_bounds(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float, float, float]:
        """
        Returns (left, top, right, bottom) of camera FOV in world coordinates.
        """
        center_x = self.origin_x + pan_deg * self.world_px_per_deg
        center_y = self.origin_y + tilt_deg * self.world_px_per_deg

        left = center_x - self.fov_world_w / 2.0
        right = center_x + self.fov_world_w / 2.0
        top = center_y - self.fov_world_h / 2.0
        bottom = center_y + self.fov_world_h / 2.0

        return left, top, right, bottom

    def world_to_sensor(
        self,
        world_x: float,
        world_y: float,
        pan_deg: float,
        tilt_deg: float
    ) -> Tuple[float, float, bool]:
        """
        Project world point (world_x, world_y) to camera sensor pixel (u, v).
        Returns (u, v, in_fov).
        """
        center_x = self.origin_x + pan_deg * self.world_px_per_deg
        center_y = self.origin_y + tilt_deg * self.world_px_per_deg

        dx = world_x - center_x
        dy = world_y - center_y

        u = (self.width / 2.0) + (dx / self.fov_world_w) * self.width
        v = (self.height / 2.0) + (dy / self.fov_world_h) * self.height

        in_fov = (0.0 <= u < self.width) and (0.0 <= v < self.height)
        return u, v, in_fov

    def sensor_to_world(
        self,
        u: float,
        v: float,
        pan_deg: float,
        tilt_deg: float
    ) -> Tuple[float, float]:
        """
        Unproject sensor pixel (u, v) back to world coordinates.
        """
        center_x = self.origin_x + pan_deg * self.world_px_per_deg
        center_y = self.origin_y + tilt_deg * self.world_px_per_deg

        du = u - (self.width / 2.0)
        dv = v - (self.height / 2.0)

        world_x = center_x + (du / self.width) * self.fov_world_w
        world_y = center_y + (dv / self.height) * self.fov_world_h

        return world_x, world_y

    def pixel_error_to_angular_error(self, err_u: float, err_v: float) -> Tuple[float, float]:
        """
        Convert pixel offset from optical center (du, dv) to angular errors (d_pan, d_tilt) in degrees.
        """
        d_pan = err_u * self.deg_per_px_x
        d_tilt = err_v * self.deg_per_px_y
        return d_pan, d_tilt

    def angular_error_to_pixel_error(self, d_pan: float, d_tilt: float) -> Tuple[float, float]:
        """
        Convert angular error in degrees to pixel error on sensor.
        """
        err_u = d_pan * self.px_per_deg_x
        err_v = d_tilt * self.px_per_deg_y
        return err_u, err_v
