"""
Export subsystem for performance reports: CSV, JSON, and PDF (via fpdf2).
"""

import os
import csv
import json
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from fpdf import FPDF
from astra_track.app.constants import SIHBenchmarks


class PDFReport(FPDF):
    """Aerospace-grade styled PDF document generator."""

    def header(self):
        self.set_fill_color(13, 17, 23)
        self.rect(0, 0, 210, 32, 'F')
        self.set_text_color(240, 246, 252)
        self.set_font('Helvetica', 'B', 16)
        self.set_xy(10, 8)
        self.cell(0, 8, 'ASTRA-TRACK | PERFORMANCE REPORT', ln=True)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(121, 192, 255)
        self.set_xy(10, 18)
        self.cell(0, 6, 'SIH 2026 PS 26169: FSOC Virtual Camera Tracking & Coarse Alignment', ln=True)
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(110, 118, 129)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | ASTRA-Track Offline Simulation System', align='C')


class Exporter:
    """Multi-format report and data exporter."""

    @staticmethod
    def export_csv(file_path: str, frames_log: List[Dict[str, Any]]) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if not frames_log:
            return file_path

        fieldnames = list(frames_log[0].keys())
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(frames_log)
        return file_path

    @staticmethod
    def export_json(file_path: str, summary_data: Dict[str, Any]) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        return file_path

    @staticmethod
    def export_pdf(file_path: str, summary_data: Dict[str, Any], chart_image_path: Optional[str] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        pdf = PDFReport()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Overall Status Banner
        is_pass = summary_data.get("overall_pass", True)
        pdf.ln(5)
        if is_pass:
            pdf.set_fill_color(46, 160, 67) # Green
            verdict_text = "BENCHMARK RESULT: PASS (All SIH Targets Satisfied)"
        else:
            pdf.set_fill_color(248, 81, 73) # Red
            verdict_text = "BENCHMARK RESULT: FAIL (Target Thresholds Exceeded)"

        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, verdict_text, fill=True, align='C', ln=True)
        pdf.ln(5)

        # Meta info
        pdf.set_text_color(30, 30, 30)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 6, "1. Executive Summary & Session Details", ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(95, 6, f"Scenario: {summary_data.get('scenario', 'Default')}")
        pdf.cell(95, 6, f"Date / Time: {summary_data.get('timestamp', 'N/A')}", ln=True)
        pdf.cell(95, 6, f"Duration: {summary_data.get('duration_s', 0.0):.1f} s")
        pdf.cell(95, 6, f"Resolution: {summary_data.get('resolution', '640x480')} @ {summary_data.get('avg_fps', 30.0):.1f} FPS", ln=True)
        pdf.ln(4)

        # Metrics Table
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 6, "2. Operational Performance Scorecard", ln=True)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_fill_color(240, 243, 246)

        col_w = [65, 45, 45, 35]
        pdf.cell(col_w[0], 7, "Metric Parameter", border=1, fill=True)
        pdf.cell(col_w[1], 7, "Measured Value", border=1, fill=True)
        pdf.cell(col_w[2], 7, "SIH Benchmark", border=1, fill=True)
        pdf.cell(col_w[3], 7, "Status", border=1, fill=True, ln=True)

        pdf.set_font('Helvetica', '', 9)
        rows = [
            ("Acquisition Time", f"{summary_data.get('acquisition_time_s', 0.0):.2f} s", f"<= {SIHBenchmarks.MAX_ACQUISITION_TIME_SEC} s",
             "PASS" if summary_data.get('acquisition_time_s', 0.0) <= SIHBenchmarks.MAX_ACQUISITION_TIME_SEC else "FAIL"),
            ("Average Tracking Error", f"{summary_data.get('avg_error_px', 0.0):.2f} px", f"<= {SIHBenchmarks.MAX_TRACKING_ERROR_PX} px",
             "PASS" if summary_data.get('avg_error_px', 0.0) <= SIHBenchmarks.MAX_TRACKING_ERROR_PX else "FAIL"),
            ("Maximum Tracking Error", f"{summary_data.get('max_error_px', 0.0):.2f} px", "Diagnostic", "INFO"),
            ("Target Loss Rate", f"{summary_data.get('target_loss_pct', 0.0):.1f} %", f"<= {SIHBenchmarks.MAX_TARGET_LOSS_PCT} %",
             "PASS" if summary_data.get('target_loss_pct', 0.0) <= SIHBenchmarks.MAX_TARGET_LOSS_PCT else "FAIL"),
            ("Reacquisition Time", f"{summary_data.get('avg_reacquisition_time_s', 0.0):.2f} s", f"<= {SIHBenchmarks.MAX_REACQUISITION_TIME_SEC} s",
             "PASS" if summary_data.get('avg_reacquisition_time_s', 0.0) <= SIHBenchmarks.MAX_REACQUISITION_TIME_SEC else "FAIL"),
            ("Lock Retention Rate", f"{summary_data.get('lock_retention_pct', 100.0):.1f} %", f">= {SIHBenchmarks.MIN_LOCK_RETENTION_PCT} %",
             "PASS" if summary_data.get('lock_retention_pct', 100.0) >= SIHBenchmarks.MIN_LOCK_RETENTION_PCT else "FAIL"),
            ("Processing Throughput", f"{summary_data.get('avg_fps', 30.0):.1f} FPS", f">= {SIHBenchmarks.MIN_PROCESSING_FPS} FPS",
             "PASS" if summary_data.get('avg_fps', 30.0) >= SIHBenchmarks.MIN_PROCESSING_FPS else "FAIL"),
        ]

        for title, val, bench, status in rows:
            pdf.cell(col_w[0], 6, title, border=1)
            pdf.cell(col_w[1], 6, val, border=1)
            pdf.cell(col_w[2], 6, bench, border=1)
            if status == "PASS":
                pdf.set_text_color(46, 160, 67)
            elif status == "FAIL":
                pdf.set_text_color(248, 81, 73)
            else:
                pdf.set_text_color(30, 30, 30)
            pdf.cell(col_w[3], 6, status, border=1, ln=True)
            pdf.set_text_color(30, 30, 30)

        pdf.ln(5)

        # Embedded chart if provided
        if chart_image_path and os.path.exists(chart_image_path):
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(0, 6, "3. Trajectory & Tracking Telemetry Curves", ln=True)
            pdf.image(chart_image_path, x=15, y=pdf.get_y() + 2, w=180)

        pdf.output(file_path)
        return file_path
