"""
Annotated video generator burning HUD overlays, Kalman predictions, and telemetry into MP4/AVI.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
from astra_track.core.frame import ProcessedFrame
from astra_track.app.constants import PALETTE, TrackingStatus


class AnnotatedVideoWriter:
    """Exports diagnostic-grade MP4 video files with embedded telemetry burns."""

    def __init__(self, output_path: str, width: int = 640, height: int = 480, fps: float = 30.0):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        if not self.writer.isOpened():
            # Fallback to XVID / AVI codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    @staticmethod
    def draw_hud(frame_bgr: np.ndarray, proc: ProcessedFrame, trail_history: List[Tuple[float, float]]) -> np.ndarray:
        """
        Renders technical HUD telemetry onto frame.
        """
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Optical Center Crosshair (Boresight)
        cv2.drawMarker(frame_bgr, (cx, cy), (240, 136, 62), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1, line_type=cv2.LINE_AA)
        cv2.circle(frame_bgr, (cx, cy), 20, (240, 136, 62), 1, cv2.LINE_AA)

        # 2. Tracking Trail History
        if len(trail_history) > 1:
            pts = np.array([(int(round(px)), int(round(py))) for px, py in trail_history], dtype=np.int32)
            cv2.polylines(frame_bgr, [pts], isClosed=False, color=(163, 113, 247), thickness=1, lineType=cv2.LINE_AA)

        # 3. Detection Bounding Box & Measured Centroid
        if proc.detection.detected:
            dx = int(round(proc.detection.x - proc.detection.width / 2.0))
            dy = int(round(proc.detection.y - proc.detection.height / 2.0))
            dw = int(round(proc.detection.width))
            dh = int(round(proc.detection.height))
            cv2.rectangle(frame_bgr, (dx, dy), (dx + dw, dy + dh), (63, 185, 80), 2, cv2.LINE_AA)

            # Measured centroid
            mx = int(round(proc.detection.x))
            my = int(round(proc.detection.y))
            cv2.circle(frame_bgr, (mx, my), 3, (63, 185, 80), -1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"BEACON {proc.detection.confidence*100:.0f}%", (dx, max(15, dy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1, cv2.LINE_AA)

            # Error Vector from Center to Target
            cv2.line(frame_bgr, (cx, cy), (mx, my), (240, 136, 62), 1, cv2.LINE_AA)

        # 4. Kalman Filtered Position
        if proc.kalman.speed > 0 or proc.tracking_status == TrackingStatus.TRACKING:
            kx = int(round(proc.kalman.x))
            ky = int(round(proc.kalman.y))
            cv2.circle(frame_bgr, (kx, ky), 4, (46, 160, 67), 1, cv2.LINE_AA)

        # 5. Future Predicted Trajectory
        if proc.prediction.predicted_points:
            pred_pts = [(int(round(px)), int(round(py))) for px, py in proc.prediction.predicted_points]
            for i, p in enumerate(pred_pts):
                cv2.circle(frame_bgr, p, 2 + i, (88, 166, 255), -1, cv2.LINE_AA)
            if len(pred_pts) > 1:
                cv2.polylines(frame_bgr, [np.array(pred_pts, dtype=np.int32)], isClosed=False, color=(88, 166, 255), thickness=1, lineType=cv2.LINE_AA)

            # Lead point crosshair
            lx = int(round(proc.prediction.lead_target_x))
            ly = int(round(proc.prediction.lead_target_y))
            cv2.drawMarker(frame_bgr, (lx, ly), (88, 166, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)

        # 6. Top Telemetry Banner
        status_color = (63, 185, 80) if "TRACK" in proc.tracking_status.value or "LOCK" in proc.tracking_status.value else (248, 81, 73)
        if "REACQ" in proc.tracking_status.value:
            status_color = (210, 153, 34)

        # Semitransparent banner overlay
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, 38), (13, 17, 23), -1)
        cv2.rectangle(overlay, (0, h - 26), (w, h), (13, 17, 23), -1)
        cv2.addWeighted(overlay, 0.75, frame_bgr, 0.25, 0, frame_bgr)

        # Banner text
        cv2.putText(frame_bgr, f"ASTRA-TRACK | F#{proc.frame_id:04d} | {proc.timestamp:.2f}s", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 246, 252), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"STATUS: {proc.tracking_status.value}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

        cv2.putText(frame_bgr, f"FPS: {proc.fps:.1f}", (w - 180, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 246, 252), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"ERR: {proc.tracking_error_px:.1f}px", (w - 180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 246, 252), 1, cv2.LINE_AA)

        # Bottom info
        cv2.putText(frame_bgr, f"PAN: {proc.ptu_pan:+.2f}*  TILT: {proc.ptu_tilt:+.2f}*  CONF: {proc.detection.confidence*100:.1f}%  MODE: {proc.control.mode}", (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 148, 158), 1, cv2.LINE_AA)

        return frame_bgr

    def write_frame(self, frame_gray_or_bgr: np.ndarray, proc: ProcessedFrame, trail: List[Tuple[float, float]]):
        if not self.writer or not self.writer.isOpened():
            return
        if len(frame_gray_or_bgr.shape) == 2:
            frame_bgr = cv2.cvtColor(frame_gray_or_bgr, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame_gray_or_bgr.copy()

        hud_frame = self.draw_hud(frame_bgr, proc, trail)
        self.writer.write(hud_frame)

    def close(self):
        if self.writer and self.writer.isOpened():
            self.writer.release()
        self.writer = None
