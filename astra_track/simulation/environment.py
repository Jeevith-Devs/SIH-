"""
2D Virtual Environment for FSOC simulation.
Manages world coordinate space, stars/background, and renders the technical world overview map.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from astra_track.app.config import EnvironmentConfig


class VirtualEnvironment:
    """Configurable 2D world space representing optical propagation theater."""

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Pre-generate background stars
        rng = np.random.default_rng(config.star_seed)
        num_stars = max(10, config.background_stars)
        self.stars_x = rng.integers(20, self.width - 20, num_stars)
        self.stars_y = rng.integers(20, self.height - 20, num_stars)
        self.stars_brightness = rng.integers(50, 180, num_stars)

    def render_world_map(
        self,
        beacon_world_pos: Tuple[float, float],
        trajectory_trail: List[Tuple[float, float]],
        fov_bounds: Tuple[float, float, float, float], # (left, top, right, bottom)
        camera_origin: Tuple[float, float] = (1000.0, 1000.0),
        output_size: Tuple[int, int] = (600, 600)
    ) -> np.ndarray:
        """
        Renders a clean, high-contrast technical 2D world map for the dashboard center panel.
        """
        out_w, out_h = output_size
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # Background: dark aerospace navy
        canvas[:, :] = (13, 17, 23)

        scale_x = out_w / float(self.width)
        scale_y = out_h / float(self.height)

        def to_canvas(wx: float, wy: float) -> Tuple[int, int]:
            cx = int(max(0, min(out_w - 1, wx * scale_x)))
            cy = int(max(0, min(out_h - 1, wy * scale_y)))
            return cx, cy

        # 1. Subtle coordinate grid
        grid_step = 250
        grid_color = (25, 35, 45)
        for gx in range(0, self.width, grid_step):
            x_screen = int(gx * scale_x)
            cv2.line(canvas, (x_screen, 0), (x_screen, out_h), grid_color, 1)
        for gy in range(0, self.height, grid_step):
            y_screen = int(gy * scale_y)
            cv2.line(canvas, (0, y_screen), (out_w, y_screen), grid_color, 1)

        # 2. Render background celestial stars
        for sx, sy, sb in zip(self.stars_x, self.stars_y, self.stars_brightness):
            scx, scy = to_canvas(sx, sy)
            b = int(sb)
            cv2.circle(canvas, (scx, scy), 1, (b, b, b), -1)

        # 3. Trajectory trail
        if len(trajectory_trail) > 1:
            points = np.array([to_canvas(px, py) for px, py in trajectory_trail], dtype=np.int32)
            cv2.polylines(canvas, [points], isClosed=False, color=(163, 113, 247), thickness=1, lineType=cv2.LINE_AA)

        # 4. Camera FOV Box
        left, top, right, bottom = fov_bounds
        p1 = to_canvas(left, top)
        p2 = to_canvas(right, bottom)
        # FOV rectangle
        cv2.rectangle(canvas, p1, p2, (121, 192, 255), thickness=2, lineType=cv2.LINE_AA)
        
        # Crosshair at center of Camera FOV
        fov_cx = int((p1[0] + p2[0]) / 2)
        fov_cy = int((p1[1] + p2[1]) / 2)
        cv2.drawMarker(canvas, (fov_cx, fov_cy), (240, 136, 62), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

        # Line of Sight line from terminal base to FOV center
        base_pt = to_canvas(camera_origin[0], camera_origin[1])
        cv2.circle(canvas, base_pt, 4, (240, 136, 62), -1)
        cv2.line(canvas, base_pt, (fov_cx, fov_cy), (60, 80, 110), 1, lineType=cv2.LINE_AA)

        # 5. Target beacon in world
        bx, by = to_canvas(beacon_world_pos[0], beacon_world_pos[1])
        # Glowing beacon circle
        cv2.circle(canvas, (bx, by), 6, (63, 185, 80), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 9, (46, 160, 67), 1, lineType=cv2.LINE_AA)

        # Labels
        cv2.putText(canvas, "WORLD: 2000x2000 px", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (139, 148, 158), 1, cv2.LINE_AA)
        cv2.putText(canvas, "TARGET BEACON", (bx + 10, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, "CAMERA FOV", (p1[0] + 5, p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (121, 192, 255), 1, cv2.LINE_AA)

        return canvas
