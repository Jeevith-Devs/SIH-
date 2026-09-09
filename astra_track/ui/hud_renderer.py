"""
Pixel-perfect aerospace HUD overlay renderer for camera and world views.
Operates directly on OpenCV BGR numpy arrays for maximum performance and cross-platform compatibility.
"""

import math
from typing import Tuple, List, Optional
import cv2
import numpy as np

from astra_track.core.frame import ProcessedFrame
from astra_track.app.constants import TrackingStatus, PALETTE


class HUDVisualizer:
    """Renders professional aerospace instrumentation overlays onto video frames."""

    @staticmethod
    def render_camera_hud(
        frame_gray: np.ndarray,
        proc: ProcessedFrame,
        trail: List[Tuple[float, float]],
        search_roi: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Renders technical aerospace HUD over the 640x480 camera view.
        """
        if frame_gray is None:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        elif len(frame_gray.shape) == 2:
            canvas = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        else:
            canvas = frame_gray.copy()

        h, w = canvas.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Corner Reticle Brackets
        bracket_len = 20
        bracket_col = (50, 70, 90)
        # Top-left
        cv2.line(canvas, (15, 45), (15 + bracket_len, 45), bracket_col, 1)
        cv2.line(canvas, (15, 45), (15, 45 + bracket_len), bracket_col, 1)
        # Top-right
        cv2.line(canvas, (w - 15, 45), (w - 15 - bracket_len, 45), bracket_col, 1)
        cv2.line(canvas, (w - 15, 45), (w - 15, 45 + bracket_len), bracket_col, 1)
        # Bottom-left
        cv2.line(canvas, (15, h - 35), (15 + bracket_len, h - 35), bracket_col, 1)
        cv2.line(canvas, (15, h - 35), (15, h - 35 - bracket_len), bracket_col, 1)
        # Bottom-right
        cv2.line(canvas, (w - 15, h - 35), (w - 15 - bracket_len, h - 35), bracket_col, 1)
        cv2.line(canvas, (w - 15, h - 35), (w - 15, h - 35 - bracket_len), bracket_col, 1)

        # 2. Optical Center (Optical Boresight)
        cv2.drawMarker(canvas, (cx, cy), (240, 136, 62), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1, line_type=cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 18, (240, 136, 62), 1, cv2.LINE_AA)

        # 3. Tracking Trail (Purple)
        if len(trail) > 1:
            pts = np.array([(int(round(px)), int(round(py))) for px, py in trail], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=(163, 113, 247), thickness=1, lineType=cv2.LINE_AA)

        # 4. Reacquisition Search ROI (Amber Yellow)
        if search_roi is not None:
            rx, ry, rw, rh = search_roi
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (227, 179, 65), 1, cv2.LINE_AA)
            cv2.putText(canvas, "PREDICTED SEARCH ROI", (rx, max(45, ry - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (227, 179, 65), 1, cv2.LINE_AA)

        # 5. Detected Target Bounding Box & Centroid (Emerald Green)
        if proc.detection.detected:
            dx = int(round(proc.detection.x - proc.detection.width / 2.0))
            dy = int(round(proc.detection.y - proc.detection.height / 2.0))
            dw = int(round(proc.detection.width))
            dh = int(round(proc.detection.height))
            
            # Corner accents on bounding box
            cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), (63, 185, 80), 1, cv2.LINE_AA)
            
            # Measured Centroid Point
            mx = int(round(proc.detection.x))
            my = int(round(proc.detection.y))
            cv2.circle(canvas, (mx, my), 3, (63, 185, 80), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"BEACON {proc.detection.confidence*100:.0f}%", (dx, max(45, dy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1, cv2.LINE_AA)

            # Error Vector connecting Center to Measured Centroid
            cv2.line(canvas, (cx, cy), (mx, my), (240, 136, 62), 1, cv2.LINE_AA)

        # 6. Kalman Filter State & Covariance Uncertainty Ellipse (Deep Green & Gold)
        if proc.kalman.speed > 0 or "TRACK" in proc.tracking_status.value:
            kx = int(round(proc.kalman.x))
            ky = int(round(proc.kalman.y))
            cv2.circle(canvas, (kx, ky), 4, (46, 160, 67), 1, cv2.LINE_AA)

            # Uncertainty Ellipse
            rad = max(4, int(round(proc.kalman.uncertainty_radius)))
            cv2.circle(canvas, (kx, ky), rad, (227, 179, 65), 1, cv2.LINE_AA)

        # 7. Future Trajectory Prediction & Lead Target (Cyan Blue)
        if proc.prediction.predicted_points:
            pred_pts = [(int(round(px)), int(round(py))) for px, py in proc.prediction.predicted_points]
            for i, p in enumerate(pred_pts):
                cv2.circle(canvas, p, 2 + i, (88, 166, 255), -1, cv2.LINE_AA)
            if len(pred_pts) > 1:
                cv2.polylines(canvas, [np.array(pred_pts, dtype=np.int32)], isClosed=False, color=(88, 166, 255), thickness=1, lineType=cv2.LINE_AA)

            # Lead point crosshair
            lx = int(round(proc.prediction.lead_target_x))
            ly = int(round(proc.prediction.lead_target_y))
            cv2.drawMarker(canvas, (lx, ly), (88, 166, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)

        # 8. Top Banner Background Overlay
        top_bar = canvas.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 36), (13, 17, 23), -1)
        cv2.rectangle(top_bar, (0, h - 28), (w, h), (13, 17, 23), -1)
        cv2.addWeighted(top_bar, 0.85, canvas, 0.15, 0, canvas)

        # 9. Top Telemetry Text
        status_val = proc.tracking_status.value
        if "TRACK" in status_val or "LOCK" in status_val:
            col_status = (63, 185, 80) # Green
        elif "LOST" in status_val:
            col_status = (248, 81, 73) # Red
        elif "REACQ" in status_val:
            col_status = (210, 153, 34) # Amber
        else:
            col_status = (88, 166, 255) # Blue

        cv2.putText(canvas, "ASTRA-TRACK", (12, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 246, 252), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"STATUS: {status_val}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_status, 1, cv2.LINE_AA)

        cv2.putText(canvas, f"FPS: {proc.fps:.1f}", (w - 190, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 246, 252), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"ERR: {proc.tracking_error_px:.1f}px ({proc.angular_error_deg*60:.1f}')", (w - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 246, 252), 1, cv2.LINE_AA)

        # 10. Bottom Actuator & Kinematics Bar
        bot_txt = f"PAN: {proc.ptu_pan:+.2f}*  TILT: {proc.ptu_tilt:+.2f}* | SPEED: {proc.kalman.speed:.1f} px/s | CONF: {proc.detection.confidence*100:.0f}% | MODE: {proc.control.mode}"
        cv2.putText(canvas, bot_txt, (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 148, 158), 1, cv2.LINE_AA)

        return canvas
