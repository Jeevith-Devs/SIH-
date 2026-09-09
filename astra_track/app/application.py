"""
Unified application orchestrator for ASTRA-Track.
Handles CLI arguments, headless benchmarks, synthetic video generation, and GUI startup.
"""

import sys
import os
import argparse
import time

from astra_track.app.config import AstraConfig
from astra_track.simulation.synthetic_generator import SyntheticVideoGenerator
from astra_track.metrics.benchmark_runner import BenchmarkRunner
from astra_track.metrics.report import PerformanceReportGenerator
from astra_track.core.pipeline import TrackingPipeline
from astra_track.core.frame import ProcessedFrame
from astra_track.io.video_input import VideoInputReader
from astra_track.storage.session import SessionRecorder


class AstraApplication:
    """Master controller managing both CLI operational modes and Desktop GUI launch."""

    def __init__(self):
        self.parser = self._create_arg_parser()

    def _create_arg_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="ASTRA-Track: AI-Based Predictive Virtual Camera Tracking for Mobile FSOC Terminals (SIH 2026 PS 26169)"
        )
        parser.add_argument("--gui", action="store_true", help="Launch standalone desktop graphical interface (default)")
        parser.add_argument("--benchmark", action="store_true", help="Run automated benchmark battery (5 scenarios)")
        parser.add_argument("--count", type=int, default=5, help="Number of scenarios for benchmark mode (5, 10, 50, 100)")
        parser.add_argument("--stress-test", action="store_true", help="Run randomized multi-variable stress test battery")
        parser.add_argument("--generate-video", action="store_true", help="Generate synthetic benchmark MP4 video files")
        parser.add_argument("--preset", type=str, default="moderate", choices=["easy", "moderate", "hard", "extreme", "reacquisition", "stress"],
                            help="Preset to use for video generation or demo")
        parser.add_argument("--output", type=str, default="reports/videos/synthetic_benchmark.mp4", help="Output path for video generation")
        parser.add_argument("--video", type=str, default=None, help="Process recorded MP4/AVI video file in Mode B")
        parser.add_argument("--demo", action="store_true", help="Execute one-click SIH Judge Demonstration sequence")
        parser.add_argument("--gimbal", action="store_true", help="Launch standalone interactive 3D Gimbal Visualizer")
        parser.add_argument("--headless", action="store_true", help="Run demo or video testing headlessly without GUI popup")
        return parser

    def run(self, args=None) -> int:
        parsed = self.parser.parse_args(args)

        # 1. Benchmark Battery Mode
        if parsed.benchmark:
            print("\n" + "="*70)
            print(f"[*] ASTRA-TRACK: RUNNING AUTOMATED BENCHMARK BATTERY ({parsed.count} SCENARIOS)")
            print("="*70)
            summary = BenchmarkRunner.run_batch(
                count=parsed.count,
                progress_cb=lambda cur, tot, msg: print(f"  [{cur}/{tot}] {msg}")
            )
            print("\n" + "-"*70)
            print(f"Total Scenarios Evaluated: {summary['total_scenarios']}")
            print(f"Passed: {summary['passed']}  |  Failed: {summary['failed']}")
            print(f"Overall Pass Rate: {summary['pass_rate_pct']:.1f}%")
            print(f"Average Tracking Error: {summary['avg_tracking_error_px']:.2f} px")
            print(f"Maximum Tracking Error: {summary['max_tracking_error_px']:.2f} px")
            print(f"Average Target Loss Rate: {summary['avg_target_loss_pct']:.1f}%")
            print(f"Average Throughput: {summary['avg_fps']:.1f} FPS")
            print("="*70 + "\n")
            return 0 if summary['passed'] > 0 else 1

        # 2. Stress Test Mode
        if parsed.stress_test:
            print("\n" + "="*70)
            print(f"[*] ASTRA-TRACK: RUNNING MULTI-VARIABLE STRESS TEST BATTERY ({parsed.count} RUNS)")
            print("="*70)
            summary = BenchmarkRunner.run_stress_test(
                count=parsed.count,
                progress_cb=lambda cur, tot, msg: print(f"  [{cur}/{tot}] {msg}")
            )
            print("\n" + "-"*70)
            print(f"Runs: {summary['total_runs']} | Passed: {summary['passed']} | Pass Rate: {summary['pass_rate_pct']:.1f}%")
            print(f"Best Case Avg Error: {summary['best_case']['avg_error_px']:.2f} px")
            print(f"Worst Case Avg Error: {summary['worst_case']['avg_error_px']:.2f} px")
            print(f"Average Case Avg Error: {summary['average_case']['avg_error_px']:.2f} px")
            print("="*70 + "\n")
            return 0

        # 3. Synthetic Video Generator Mode
        if parsed.generate_video:
            print(f"[*] Generating synthetic test video with preset: '{parsed.preset}'...")
            if parsed.preset == "easy":
                cfg = AstraConfig.preset_easy()
            elif parsed.preset == "moderate":
                cfg = AstraConfig.preset_moderate()
            elif parsed.preset == "hard":
                cfg = AstraConfig.preset_hard()
            elif parsed.preset == "extreme":
                cfg = AstraConfig.preset_extreme()
            else:
                cfg = AstraConfig.preset_reacquisition()

            res = SyntheticVideoGenerator.generate_benchmark(
                config=cfg,
                output_mp4=parsed.output,
                duration_sec=8.0,
                progress_cb=lambda c, t, m: print(f"\r{m}", end="", flush=True)
            )
            print(f"\n[+] Video created: {res['video']}")
            print(f"[+] Ground truth CSV: {res['csv']}")
            print(f"[+] Configuration JSON: {res['json']}")
            return 0

        # 4. Mode B: Video File Tracking Mode
        if parsed.video:
            print(f"[*] Processing recorded video file: {parsed.video}")
            reader = VideoInputReader()
            if not reader.open(parsed.video):
                print(f"[-] Error: Could not open video file '{parsed.video}'")
                return 1

            print(f"[+] Loaded: {reader.total_frames} frames @ {reader.fps:.1f} FPS")
            pipe = TrackingPipeline(AstraConfig())
            session = SessionRecorder("Video Benchmark")

            while True:
                ret, frame, f_idx = reader.read_frame()
                if not ret or frame is None:
                    break
                det = pipe.detector.detect(frame)
                k_state, pred, _ = pipe.target_mgr.process_measurement(det, dt=1.0/reader.fps)
                st = pipe.fsm.update(det.detected, det.confidence, 1.0/reader.fps)
                sim_pan, sim_tilt = reader.update_simulated_ptu(det.x, det.y, 1.0/reader.fps) if det.detected else (0.0, 0.0)

                proc = ProcessedFrame(
                    frame_id=f_idx,
                    timestamp=f_idx / reader.fps,
                    raw_frame=frame,
                    detection=det,
                    kalman=k_state,
                    prediction=pred,
                    tracking_status=st,
                    fps=reader.fps,
                    ptu_pan=sim_pan,
                    ptu_tilt=sim_tilt
                )
                session.record_frame(proc, log_frame_detail=False)
                if f_idx % 30 == 0:
                    print(f"\r  Frame {f_idx}/{reader.total_frames} | State: {st.value} | Conf: {det.confidence*100:.0f}%", end="")

            reader.close()
            summary = session.finalize_summary(pipe.fsm)
            print("\n" + PerformanceReportGenerator.format_text_report(summary))
            return 0

        # 5. Demonstration Mode
        if parsed.demo:
            print("\n" + "="*70)
            print("[*] ASTRA-TRACK: EXECUTING SIH JUDGE DEMONSTRATION SEQUENCE")
            print("="*70)
            summary = BenchmarkRunner.run_scenario(
                AstraConfig.preset_extreme(),
                duration_s=10.0,
                scenario_name="SIH Demo Run",
                inject_loss=True
            )
            print(PerformanceReportGenerator.format_text_report(summary))
        # 6. Standalone 3D Gimbal Mode
        if parsed.gimbal:
            from astra_track.ui.widgets.gimbal_3d_widget import run_standalone_gimbal_window
            run_standalone_gimbal_window()
            return 0

        # Default: Launch GUI
        from astra_track.ui.main_window import launch_gui
        launch_gui()
        return 0
