"""
Multi-stage classical computer vision detector for optical FSOC beacons.
Implements the 8-stage pipeline: grayscale -> denoising -> contrast enhancement ->
adaptive thresholding -> morphology -> connected components -> multi-parameter filtering -> centroid.
"""

import time
import math
from typing import List, Tuple, Optional
import cv2
import numpy as np

from astra_track.core.frame import DetectionCandidate, DetectionResult
from astra_track.perception.centroid import CentroidCalculator
from astra_track.perception.confidence import ConfidenceScorer


class ClassicalDetector:
    """Robust, sub-millisecond optical spot detector optimized for high noise and adverse weather."""

    def __init__(self, expected_size: int = 10, min_area: float = 4.0, max_area: float = 1200.0):
        self.expected_size = expected_size
        self.min_area = min_area
        self.max_area = max_area
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def detect(
        self,
        frame: np.ndarray,
        predicted_pos: Optional[Tuple[float, float]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> DetectionResult:
        """
        Runs the full detection pipeline.
        Args:
            frame: 2D or 3D image
            predicted_pos: Optional (x, y) prior from Kalman filter
            roi: Optional (x, y, w, h) bounding box to constrain search for speed & gating
        Returns:
            DetectionResult
        """
        start_time = time.perf_counter()

        if frame is None or frame.size == 0:
            return DetectionResult(detected=False, method="Classical CV")

        # 1. Grayscale Conversion
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        img_h, img_w = gray.shape

        # Extract ROI if specified
        roi_offset_x, roi_offset_y = 0, 0
        if roi is not None:
            rx, ry, rw, rh = roi
            rx1 = max(0, min(img_w - 1, rx))
            ry1 = max(0, min(img_h - 1, ry))
            rx2 = max(rx1 + 5, min(img_w, rx1 + rw))
            ry2 = max(ry1 + 5, min(img_h, ry1 + rh))
            search_region = gray[ry1:ry2, rx1:rx2]
            roi_offset_x, roi_offset_y = rx1, ry1
        else:
            search_region = gray

        sr_h, sr_w = search_region.shape
        if sr_h < 3 or sr_w < 3:
            return DetectionResult(detected=False, method="Classical CV")

        # Scene background statistics
        bg_mean = float(np.mean(search_region))
        bg_std = float(np.std(search_region))

        # 2. Denoising & Top-Hat Morphological Filter (Isolates optical spot from fog/cloud backgrounds)
        denoised = cv2.GaussianBlur(search_region, (3, 3), 0.8)
        
        # White Top-Hat: f - (f o b), extracts bright local features smaller than structuring element
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel_tophat)

        # 3. Contrast Enhancement via CLAHE on Top-Hat filtered image
        enhanced = self.clahe.apply(tophat)

        # 4. Dual Thresholding: Otsu on Top-Hat + High Intensity Slicing
        # Adaptive thresholding on enhanced contrast
        _, thresh_tophat = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Slicing threshold (4-sigma CFAR detection against Gaussian sensor noise)
        th_val = max(35.0, bg_mean + 4.0 * bg_std)
        _, thresh_direct = cv2.threshold(denoised, min(th_val, 220.0), 255, cv2.THRESH_BINARY)

        # Combined binary mask
        combined_bin = cv2.bitwise_or(thresh_tophat, thresh_direct)

        # 5. Morphological Processing
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(combined_bin, cv2.MORPH_OPEN, kernel_open)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morphed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 6. Connected Components & Contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[DetectionCandidate] = []

        # 7. Multi-Parameter Candidate Filtering
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / float(max(1, bh))
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4.0 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

            # Absolute sensor coordinates
            abs_x = bx + roi_offset_x
            abs_y = by + roi_offset_y

            # Extract local patch for photometry
            patch = search_region[by:by + bh, bx:bx + bw]
            peak_val = float(np.max(patch)) if patch.size > 0 else 0.0
            mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0

            if peak_val < 35.0:
                continue

            effective_std = max(1.5, bg_std)
            snr = (peak_val - bg_mean) / effective_std

            # Distance from prediction gating
            dist_to_pred = None
            if predicted_pos is not None:
                cand_center_x = abs_x + bw / 2.0
                cand_center_y = abs_y + bh / 2.0
                dist_to_pred = math.hypot(cand_center_x - predicted_pos[0], cand_center_y - predicted_pos[1])

            # Compute candidate confidence score
            conf = ConfidenceScorer.calculate_confidence(
                peak_intensity=peak_val,
                mean_intensity=mean_val,
                bg_mean=bg_mean,
                bg_std=bg_std,
                area=area,
                circularity=circularity,
                expected_size=float(self.expected_size),
                distance_from_prediction=dist_to_pred
            )

            cand = DetectionCandidate(
                x=float(abs_x),
                y=float(abs_y),
                width=float(bw),
                height=float(bh),
                area=float(area),
                peak_intensity=peak_val,
                mean_intensity=mean_val,
                snr=snr,
                circularity=circularity,
                score=conf
            )
            candidates.append(cand)

        # 8. Candidate Ranking & Centroid Extraction
        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not candidates:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                candidate_count=0,
                method="Classical CV",
                execution_time_ms=exec_time_ms,
                roi=roi
            )

        # Sort candidates descending by composite score
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        # Sub-pixel centroid using intensity weighting
        int_bbox = (int(best.x), int(best.y), int(best.width), int(best.height))
        patch_for_centroid = gray[int_bbox[1]:int_bbox[1] + int_bbox[3], int_bbox[0]:int_bbox[0] + int_bbox[2]]
        c_res = CentroidCalculator.compute(patch_for_centroid, int_bbox)

        return DetectionResult(
            detected=True,
            x=c_res.intensity_x,
            y=c_res.intensity_y,
            width=best.width,
            height=best.height,
            confidence=best.score,
            candidate_count=len(candidates),
            method="Classical CV",
            execution_time_ms=exec_time_ms,
            roi=roi,
            peak_intensity=best.peak_intensity
        )
