"""
ASTRA-Track: Standalone Aerospace Engineering Desktop Application.
Features:
- Dark technical theme
- 8 Functional Pages: Home, Simulation Dashboard, Video Testing, Scenario Builder,
  Video Generator, Performance Analytics, Reports & History, Hardware Architecture.
- Interactive controls, real-time HUD telemetry, live Matplotlib plots, and Judge Demo Mode.
"""

import os
import sys
import time
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from astra_track.app.config import AstraConfig
from astra_track.app.constants import (
    TrackingStatus, ControlMode, TrajectoryType, AtmosphereMode,
    DisturbancePattern, SIHBenchmarks, PALETTE
)
from astra_track.core.pipeline import TrackingPipeline
from astra_track.core.frame import ProcessedFrame
from astra_track.storage.session import SessionRecorder
from astra_track.storage.database import DatabaseManager
from astra_track.storage.exports import Exporter
from astra_track.simulation.synthetic_generator import SyntheticVideoGenerator
from astra_track.metrics.benchmark_runner import BenchmarkRunner
from astra_track.metrics.report import PerformanceReportGenerator
from astra_track.io.video_input import VideoInputReader
from astra_track.io.video_output import AnnotatedVideoWriter
from astra_track.io.camera_interface import RealCameraAdapter
from astra_track.io.ptu_interface import SerialServoPTUAdapter
from astra_track.ui.hud_renderer import HUDVisualizer
from astra_track.ui.widgets.gimbal_3d_widget import Gimbal3DWidget
from astra_track.ui.audio_alerts import AudioAlertManager
from astra_track.storage.slide_exporter import SlideExporter
import webbrowser


class AstraTrackApp:
    """Master Desktop GUI Application for ASTRA-Track."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ASTRA-Track — AI-Based FSOC Coarse Alignment & Tracking Platform")
        self.root.geometry("1440x920")
        self.root.minsize(1200, 800)
        self.root.configure(bg=PALETTE["bg_dark"])

        # Core Configuration & State
        self.config = AstraConfig.preset_moderate()
        self.pipeline = TrackingPipeline(self.config)
        self.db_manager = DatabaseManager()
        self.session_recorder = SessionRecorder(scenario_name="Live Session", db_manager=self.db_manager)

        self.is_sim_running = False
        self.is_sim_paused = False
        self.sim_speed = 1.0

        # Hardware-in-the-Loop (HIL) & Physical Webcam State
        self.input_mode = "VIRTUAL" # "VIRTUAL", "WEBCAM"
        self.real_camera: Optional[RealCameraAdapter] = None
        self.real_ptu: Optional[SerialServoPTUAdapter] = None
        self.live_pan = 0.0
        self.live_tilt = 0.0

        # Video Testing State (Mode B)
        self.video_reader = VideoInputReader()
        self.is_video_running = False
        self.video_session = SessionRecorder(scenario_name="Video Test", db_manager=self.db_manager)

        # Demo Mode State
        self.is_demo_mode = False
        self.demo_step = 0
        self.demo_timer = 0.0

        # Audio PAT feedback and 3D Gimbal State
        self.audio_alerts = AudioAlertManager(enabled=True)
        self.last_status: Optional[TrackingStatus] = None

        # Setup Styles & UI Layout
        self._setup_styles()
        self._build_main_layout()

        # Telemetry refresh timer
        self.root.after(30, self._sim_tick_loop)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=PALETTE["bg_panel"], foreground=PALETTE["text_primary"])
        style.configure("TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["text_primary"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=PALETTE["text_primary"])
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=PALETTE["camera_fov"])
        style.configure("Badge.TLabel", font=("Segoe UI", 10, "bold"), padding=4)
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5, background=PALETTE["bg_card"], foreground=PALETTE["text_primary"])
        style.map("TButton", background=[("active", PALETTE["accent"])])
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground="#ffffff")
        style.configure("Danger.TButton", background=PALETTE["target_lost"], foreground="#ffffff")
        style.configure("Success.TButton", background=PALETTE["target_detected"], foreground="#ffffff")

    def _build_main_layout(self):
        # 1. Top Global Header Bar
        self.header_bar = tk.Frame(self.root, bg=PALETTE["bg_dark"], height=48, bd=0, highlightthickness=1, highlightbackground=PALETTE["border"])
        self.header_bar.pack(side=tk.TOP, fill=tk.X)

        lbl_logo = tk.Label(self.header_bar, text="ASTRA-TRACK", font=("Segoe UI", 13, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_dark"])
        lbl_logo.pack(side=tk.LEFT, padx=(15, 5), pady=8)

        lbl_sub = tk.Label(self.header_bar, text="| FSOC Optical Coarse Alignment & Tracking System [SIH 2026 PS 26169]",
                           font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_dark"])
        lbl_sub.pack(side=tk.LEFT, padx=5, pady=8)

        # Global Status Badge
        self.lbl_global_status = tk.Label(self.header_bar, text="SYSTEM READY", font=("Segoe UI", 9, "bold"),
                                          bg=PALETTE["bg_card"], fg=PALETTE["target_detected"], padx=10, pady=2)
        self.lbl_global_status.pack(side=tk.RIGHT, padx=15, pady=8)

        # 2. Main Body: Sidebar + Content Area
        self.body_frame = tk.Frame(self.root, bg=PALETTE["bg_dark"])
        self.body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_sidebar()

        self.content_frame = tk.Frame(self.body_frame, bg=PALETTE["bg_dark"])
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Pages Dictionary
        self.pages: Dict[str, tk.Frame] = {}
        self._init_pages()

        # Switch to Home Page initially
        self.show_page("home")

    def _build_sidebar(self):
        sidebar = tk.Frame(self.body_frame, bg=PALETTE["bg_panel"], width=220, bd=0, highlightthickness=1, highlightbackground=PALETTE["border"])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2))
        sidebar.pack_propagate(False)

        lbl_nav = tk.Label(sidebar, text="NAVIGATION", font=("Segoe UI", 8, "bold"), fg=PALETTE["text_muted"], bg=PALETTE["bg_panel"])
        lbl_nav.pack(anchor="w", padx=15, pady=(15, 8))

        nav_items = [
            ("Home Overview", "home"),
            ("Simulation Dashboard", "dashboard"),
            ("Video Testing (MP4)", "video"),
            ("Scenario Builder", "builder"),
            ("Synthetic Video Gen", "generator"),
            ("Performance Analytics", "analytics"),
            ("Reports & Database", "reports"),
            ("Hardware Architecture", "architecture")
        ]

        self.nav_buttons: Dict[str, tk.Button] = {}
        for title, key in nav_items:
            btn = tk.Button(
                sidebar, text=f"  {title}", anchor="w", font=("Segoe UI", 9),
                bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], activebackground=PALETTE["bg_card"],
                activeforeground=PALETTE["camera_fov"], bd=0, padx=12, pady=7, cursor="hand2",
                command=lambda k=key: self.show_page(k)
            )
            btn.pack(fill=tk.X, padx=6, pady=2)
            self.nav_buttons[key] = btn

        # Separator
        sep = tk.Frame(sidebar, height=1, bg=PALETTE["border"])
        sep.pack(fill=tk.X, padx=12, pady=15)

        # One-Click Judge Demo Button
        btn_demo = tk.Button(
            sidebar, text="⭐ ONE-CLICK DEMO", font=("Segoe UI", 9, "bold"),
            bg="#8957e5", fg="#ffffff", activebackground="#a371f7", activeforeground="#ffffff",
            bd=0, padx=10, pady=8, cursor="hand2", command=self.start_judge_demo
        )
        btn_demo.pack(fill=tk.X, padx=12, pady=5)

        # Quick Benchmark Button
        btn_bench = tk.Button(
            sidebar, text="⚡ RUN 5-SCENARIO BENCHMARK", font=("Segoe UI", 8, "bold"),
            bg=PALETTE["bg_card"], fg=PALETTE["text_primary"], activebackground=PALETTE["accent"],
            bd=0, padx=8, pady=6, cursor="hand2", command=self.run_quick_benchmark
        )
        btn_bench.pack(fill=tk.X, padx=12, pady=5)

        # SIH Presentation Pitch Slides Button
        btn_deck = tk.Button(
            sidebar, text="📊 SIH PITCH SLIDES", font=("Segoe UI", 8, "bold"),
            bg="#238636", fg="#ffffff", activebackground="#2ea043", activeforeground="#ffffff",
            bd=0, padx=8, pady=6, cursor="hand2", command=self.open_pitch_deck
        )
        btn_deck.pack(fill=tk.X, padx=12, pady=5)

        # Footer info
        lbl_ver = tk.Label(sidebar, text="ASTRA-Track v1.0\nReady for SIH Evaluation", font=("Segoe UI", 7),
                           fg=PALETTE["text_muted"], bg=PALETTE["bg_panel"], justify="left")
        lbl_ver.pack(side=tk.BOTTOM, anchor="w", padx=15, pady=12)

    def open_pitch_deck(self):
        """Generates and opens the executive SIH 2026 presentation slides in the default browser."""
        deck_path = SlideExporter.export_presentation_html()
        webbrowser.open(f"file:///{os.path.abspath(deck_path)}")

    def show_page(self, page_key: str):
        for k, page in self.pages.items():
            if k == page_key:
                page.pack(fill=tk.BOTH, expand=True)
                if k in self.nav_buttons:
                    self.nav_buttons[k].configure(bg=PALETTE["bg_card"], fg=PALETTE["camera_fov"], font=("Segoe UI", 9, "bold"))
            else:
                page.pack_forget()
                if k in self.nav_buttons:
                    self.nav_buttons[k].configure(bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], font=("Segoe UI", 9))

    def _init_pages(self):
        self._build_home_page()
        self._build_dashboard_page()
        self._build_video_page()
        self._build_builder_page()
        self._build_generator_page()
        self._build_analytics_page()
        self._build_reports_page()
        self._build_architecture_page()

    # =========================================================================
    # PAGE 1: HOME OVERVIEW
    # =========================================================================
    def _build_home_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["home"] = page

        # Hero Banner
        hero = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=25, pady=20)
        hero.pack(fill=tk.X, pady=(0, 15))

        tk.Label(hero, text="ASTRA-Track", font=("Segoe UI", 20, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")
        tk.Label(hero, text="AI-Based Predictive Coarse Alignment & Virtual Camera Tracking System for Mobile FSOC Terminals",
                 font=("Segoe UI", 12), fg=PALETTE["text_primary"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(2, 6))
        tk.Label(hero, text="Smart India Hackathon 2026 — Problem Statement 26169 | Hardware-Agnostic Simulation & Verification Laboratory",
                 font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"]).pack(anchor="w")

        # Feature Highlights Grid (4 Cards)
        cards_frame = tk.Frame(page, bg=PALETTE["bg_dark"])
        cards_frame.pack(fill=tk.X, pady=(0, 15))
        cards_frame.columnconfigure((0, 1, 2, 3), weight=1, uniform="c")

        cards_data = [
            ("⚡ Hybrid Perception", "Classical CV + AI Candidate Scoring",
             "Sub-millisecond dual-thresholding, morphology, sub-pixel intensity centroid, and AI confidence estimation."),
            ("🎯 6-State Kalman Filter", "Kinematic Constant-Acceleration",
             "Tracks [x, y, vx, vy, ax, ay]^T, estimates velocity/acceleration, computes covariance uncertainty ellipses."),
            ("🚀 Predictive Control", "Proactive Feedforward Pan/Tilt",
             "Eliminates dynamic tracking lag by steering toward future predicted coordinates subject to 5°/s limits."),
            ("🔄 Intelligent Reacquisition", "4-Tier Recovery Hierarchy",
             "Extrapolates beam trajectory during occultation, recovers coarse lock within <= 1.0s.")
        ]

        for idx, (title, sub, desc) in enumerate(cards_data):
            card = tk.Frame(cards_frame, bg=PALETTE["bg_card"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=14, pady=12)
            card.grid(row=0, column=idx, padx=6, sticky="nsew")
            tk.Label(card, text=title, font=("Segoe UI", 10, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_card"]).pack(anchor="w")
            tk.Label(card, text=sub, font=("Segoe UI", 8, "bold"), fg=PALETTE["text_secondary"], bg=PALETTE["bg_card"]).pack(anchor="w", pady=(2, 4))
            tk.Label(card, text=desc, font=("Segoe UI", 8), fg=PALETTE["text_primary"], bg=PALETTE["bg_card"], wraplength=220, justify="left").pack(anchor="w")

        # Action Panel
        actions_frame = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=20, pady=18)
        actions_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(actions_frame, text="QUICK ACTIONS & OPERATIONS", font=("Segoe UI", 11, "bold"), fg=PALETTE["text_primary"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(actions_frame, bg=PALETTE["bg_panel"])
        btn_row.pack(fill=tk.X, pady=5)

        tk.Button(btn_row, text="▶ Launch Real-Time Simulation", font=("Segoe UI", 9, "bold"), bg=PALETTE["accent"], fg="#fff",
                  padx=12, pady=8, bd=0, cursor="hand2", command=lambda: self.show_page("dashboard")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="🎬 Test Recorded MP4 Video", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                  padx=12, pady=8, bd=0, cursor="hand2", command=lambda: self.show_page("video")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="⚙️ Configure Scenario", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                  padx=12, pady=8, bd=0, cursor="hand2", command=lambda: self.show_page("builder")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_row, text="📊 Run Benchmark Battery", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                  padx=12, pady=8, bd=0, cursor="hand2", command=self.run_quick_benchmark).pack(side=tk.LEFT, padx=5)

        # System Benchmark Targets Box
        target_box = tk.Frame(actions_frame, bg=PALETTE["bg_card"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=15, pady=12)
        target_box.pack(fill=tk.X, pady=15)

        tk.Label(target_box, text="SMART INDIA HACKATHON 2026 (PS 26169) OFFICIAL TARGET CRITERIA", font=("Segoe UI", 9, "bold"),
                 fg=PALETTE["camera_fov"], bg=PALETTE["bg_card"]).pack(anchor="w", pady=(0, 6))

        criteria_txt = (
            "• Initial Acquisition Time: <= 2.0 seconds\n"
            "• Average Tracking Accuracy Error: <= 10.0 pixels (Euclidean)\n"
            "• Target Loss Rate: < 5.0% of frames\n"
            "• Predictive Reacquisition Time: <= 1.0 second\n"
            "• Processing Throughput: >= 20.0 FPS\n"
            "• Continuous Lock Retention: >= 95.0%"
        )
        tk.Label(target_box, text=criteria_txt, font=("Segoe UI", 9), fg=PALETTE["text_primary"], bg=PALETTE["bg_card"], justify="left").pack(anchor="w")

    # =========================================================================
    # PAGE 2: SIMULATION DASHBOARD (PRIMARY SCREEN)
    # =========================================================================
    def _build_dashboard_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["dashboard"] = page

        # Top Control & Status Toolbar
        toolbar = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=12, pady=8)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        # Buttons
        self.btn_sim_play = tk.Button(toolbar, text="▶ START", font=("Segoe UI", 9, "bold"), bg=PALETTE["target_detected"], fg="#fff",
                                      bd=0, padx=12, pady=4, cursor="hand2", command=self.toggle_simulation)
        self.btn_sim_play.pack(side=tk.LEFT, padx=4)

        self.btn_sim_pause = tk.Button(toolbar, text="⏸ PAUSE", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                                       bd=0, padx=10, pady=4, cursor="hand2", command=self.pause_simulation)
        self.btn_sim_pause.pack(side=tk.LEFT, padx=4)

        self.btn_sim_reset = tk.Button(toolbar, text="🔄 RESET", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                                       bd=0, padx=10, pady=4, cursor="hand2", command=self.reset_simulation)
        self.btn_sim_reset.pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar, text="| Speed:", font=("Segoe UI", 9), bg=PALETTE["bg_panel"]).pack(side=tk.LEFT, padx=(10, 4))
        self.speed_combo = ttk.Combobox(toolbar, values=["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"], width=6, state="readonly")
        self.speed_combo.set("1.0x")
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_changed)
        self.speed_combo.pack(side=tk.LEFT, padx=4)

        # Disturbance Injector Buttons
        tk.Label(toolbar, text="| Inject:", font=("Segoe UI", 9), bg=PALETTE["bg_panel"]).pack(side=tk.LEFT, padx=(10, 4))
        btn_loss = tk.Button(toolbar, text="⚠️ Target Loss", font=("Segoe UI", 8, "bold"), bg=PALETTE["target_lost"], fg="#fff",
                             bd=0, padx=8, pady=3, cursor="hand2", command=self.trigger_sim_target_loss)
        btn_loss.pack(side=tk.LEFT, padx=3)

        btn_fog = tk.Button(toolbar, text="☁ Fog", font=("Segoe UI", 8, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                            bd=0, padx=8, pady=3, cursor="hand2", command=self.toggle_fog)
        btn_fog.pack(side=tk.LEFT, padx=3)

        btn_jitter = tk.Button(toolbar, text="〰 Jitter", font=("Segoe UI", 8, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                               bd=0, padx=8, pady=3, cursor="hand2", command=self.toggle_jitter)
        btn_jitter.pack(side=tk.LEFT, padx=3)

        # Hardware Source Selector
        tk.Label(toolbar, text="| Source:", font=("Segoe UI", 9), bg=PALETTE["bg_panel"]).pack(side=tk.LEFT, padx=(8, 3))
        self.source_combo = ttk.Combobox(toolbar, values=["Virtual Sim", "Live Webcam"], width=11, state="readonly")
        self.source_combo.set("Virtual Sim")
        self.source_combo.bind("<<ComboboxSelected>>", self._on_source_changed)
        self.source_combo.pack(side=tk.LEFT, padx=3)

        # Hardware Gimbal Controls
        tk.Label(toolbar, text="| PTU:", font=("Segoe UI", 9), bg=PALETTE["bg_panel"]).pack(side=tk.LEFT, padx=(8, 3))
        available_ports = SerialServoPTUAdapter.list_available_ports()
        port_values = available_ports if available_ports else ["No COM Port"]
        self.gimbal_combo = ttk.Combobox(toolbar, values=port_values, width=10, state="readonly")
        self.gimbal_combo.set(port_values[0])
        self.gimbal_combo.pack(side=tk.LEFT, padx=3)

        self.btn_connect_gimbal = tk.Button(toolbar, text="Connect PTU", font=("Segoe UI", 8, "bold"),
                                           bg=PALETTE["bg_card"], fg="#fff", bd=0, padx=6, pady=2,
                                           cursor="hand2", command=self._toggle_gimbal_connection)
        self.btn_connect_gimbal.pack(side=tk.LEFT, padx=3)

        # Live Status Chip on Top Toolbar
        self.lbl_dash_status = tk.Label(toolbar, text="STATUS: INITIALIZING", font=("Segoe UI", 9, "bold"),
                                        bg=PALETTE["bg_card"], fg=PALETTE["camera_fov"], padx=10, pady=3)
        self.lbl_dash_status.pack(side=tk.RIGHT, padx=5)

        # Center Viewport Frame (Left: Camera Frame, Center: World Map, Right: Telemetry)
        viewport_frame = tk.Frame(page, bg=PALETTE["bg_dark"])
        viewport_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        viewport_frame.columnconfigure(0, weight=4) # Camera
        viewport_frame.columnconfigure(1, weight=4) # World Map
        viewport_frame.columnconfigure(2, weight=3) # Telemetry

        # 1. Camera Sensor View (Left)
        cam_card = tk.Frame(viewport_frame, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=8, pady=8)
        cam_card.grid(row=0, column=0, sticky="nsew", padx=3)

        tk.Label(cam_card, text="OPTICAL RECEIVER SENSOR (640x480 FOV)", font=("Segoe UI", 9, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 4))
        self.cam_canvas = tk.Label(cam_card, bg="#000000", width=460, height=345)
        self.cam_canvas.pack(fill=tk.BOTH, expand=True)

        # 2. Virtual Environment World Map (Center)
        world_card = tk.Frame(viewport_frame, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=8, pady=8)
        world_card.grid(row=0, column=1, sticky="nsew", padx=3)

        tk.Label(world_card, text="VIRTUAL ENVIRONMENT (2000x2000 px WORLD)", font=("Segoe UI", 9, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 4))
        self.world_canvas = tk.Label(world_card, bg="#000000", width=460, height=345)
        self.world_canvas.pack(fill=tk.BOTH, expand=True)

        # 3. Live Telemetry & Scorecard (Right)
        telemetry_card = tk.Frame(viewport_frame, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=10, pady=8)
        telemetry_card.grid(row=0, column=2, sticky="nsew", padx=3)

        tk.Label(telemetry_card, text="LIVE PAT TELEMETRY", font=("Segoe UI", 9, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 6))

        self.telemetry_labels: Dict[str, tk.Label] = {}
        fields = [
            ("Tracking State", "INITIALIZING"),
            ("Processing FPS", "30.0"),
            ("Tracking Error", "0.0 px"),
            ("Angular Error", "0.00°"),
            ("Detection Confidence", "0.0 %"),
            ("PTU Pan Angle", "+0.00°"),
            ("PTU Tilt Angle", "+0.00°"),
            ("Estimated Speed", "0.0 px/s"),
            ("Initial Acquisition", "0.00 s"),
            ("Reacquisition Time", "0.00 s"),
            ("Lock Retention Rate", "100.0 %"),
            ("Target Loss Rate", "0.0 %"),
            ("Control Strategy", "Predictive Feedforward")
        ]

        for name, init_val in fields:
            row = tk.Frame(telemetry_card, bg=PALETTE["bg_panel"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=name, font=("Segoe UI", 8), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"]).pack(side=tk.LEFT)
            lbl_val = tk.Label(row, text=init_val, font=("Consolas", 9, "bold"), fg=PALETTE["text_primary"], bg=PALETTE["bg_panel"])
            lbl_val.pack(side=tk.RIGHT)
            self.telemetry_labels[name] = lbl_val

        # 3D Gimbal Wireframe HUD
        self.gimbal_3d = Gimbal3DWidget(telemetry_card, width=240, height=140, bg=PALETTE["bg_panel"])
        self.gimbal_3d.pack(fill=tk.BOTH, expand=True, pady=(6, 2))

        # Bottom Real-time Graphs (Matplotlib Canvas)
        plots_card = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, height=220, highlightthickness=1, highlightbackground=PALETTE["border"], padx=6, pady=4)
        plots_card.pack(fill=tk.X)
        plots_card.pack_propagate(False)

        self._init_dashboard_plots(plots_card)

    def _init_dashboard_plots(self, parent_frame):
        self.dash_fig = Figure(figsize=(10, 2.2), dpi=100, facecolor=PALETTE["bg_panel"])
        self.dash_fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22, wspace=0.30)

        # Plot 1: Tracking Error vs Time
        self.ax_err = self.dash_fig.add_subplot(1, 3, 1)
        self.ax_err.set_facecolor(PALETTE["bg_dark"])
        self.ax_err.set_title("Tracking Error (px) [<=10px target]", fontsize=8, color=PALETTE["text_primary"])
        self.ax_err.tick_params(colors=PALETTE["text_secondary"], labelsize=7)
        self.ax_err.axhline(10.0, color="#f85149", linestyle="--", linewidth=1, label="SIH Limit")
        self.line_err, = self.ax_err.plot([], [], color="#3fb950", linewidth=1.5)

        # Plot 2: FPS vs Time
        self.ax_fps = self.dash_fig.add_subplot(1, 3, 2)
        self.ax_fps.set_facecolor(PALETTE["bg_dark"])
        self.ax_fps.set_title("Processing FPS [>=20 FPS target]", fontsize=8, color=PALETTE["text_primary"])
        self.ax_fps.tick_params(colors=PALETTE["text_secondary"], labelsize=7)
        self.ax_fps.axhline(20.0, color="#f85149", linestyle="--", linewidth=1)
        self.line_fps, = self.ax_fps.plot([], [], color="#58a6ff", linewidth=1.5)

        # Plot 3: Pan & Tilt Angles vs Time
        self.ax_angles = self.dash_fig.add_subplot(1, 3, 3)
        self.ax_angles.set_facecolor(PALETTE["bg_dark"])
        self.ax_angles.set_title("PTU Pan & Tilt Angles (deg)", fontsize=8, color=PALETTE["text_primary"])
        self.ax_angles.tick_params(colors=PALETTE["text_secondary"], labelsize=7)
        self.line_pan, = self.ax_angles.plot([], [], color="#f0883e", linewidth=1.5, label="Pan")
        self.line_tilt, = self.ax_angles.plot([], [], color="#a371f7", linewidth=1.5, label="Tilt")

        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=parent_frame)
        self.dash_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_source_changed(self, event=None):
        val = self.source_combo.get()
        if val == "Live Webcam":
            if self.real_camera is None:
                self.real_camera = RealCameraAdapter(camera_index_or_url=0, width=640, height=480)
            if not self.real_camera.is_connected:
                messagebox.showwarning(
                    "Webcam Not Detected",
                    "No optical camera detected at device index 0.\nFalling back to Virtual Simulation."
                )
                self.source_combo.set("Virtual Sim")
                self.input_mode = "VIRTUAL"
                return
            self.input_mode = "WEBCAM"
            self.lbl_dash_status.config(text="MODE: LIVE WEBCAM", fg=PALETTE["target_detected"])
            if not self.is_sim_running:
                self.toggle_simulation()
        else:
            self.input_mode = "VIRTUAL"
            self.lbl_dash_status.config(text="MODE: VIRTUAL SIM", fg=PALETTE["camera_fov"])

    def _toggle_gimbal_connection(self):
        if self.real_ptu and self.real_ptu.is_connected:
            self.real_ptu.release()
            self.real_ptu = None
            self.btn_connect_gimbal.config(text="Connect PTU", bg=PALETTE["bg_card"])
            messagebox.showinfo("Gimbal", "Disconnected from physical PTU.")
            return

        port = self.gimbal_combo.get()
        if port == "No COM Port" or not port:
            ports = SerialServoPTUAdapter.list_available_ports()
            if ports:
                self.gimbal_combo.config(values=ports)
                self.gimbal_combo.set(ports[0])
                port = ports[0]
            else:
                messagebox.showinfo(
                    "No COM Port Found",
                    "No physical Arduino/ESP32 gimbal detected.\nConnect hardware via USB and click again."
                )
                return

        self.real_ptu = SerialServoPTUAdapter(port=port, baudrate=115200)
        if self.real_ptu.is_connected:
            self.btn_connect_gimbal.config(text="Disconnect", bg=PALETTE["target_lost"])
            messagebox.showinfo("Gimbal Connected", f"Linked to PTU on {port} at 115200 baud.")
        else:
            self.btn_connect_gimbal.config(text="Connect PTU", bg=PALETTE["bg_card"])
            messagebox.showwarning("Connection Failed", f"Could not establish serial handshake on {port}.")

    def _render_hardware_status_card(self, proc: ProcessedFrame) -> np.ndarray:
        """Renders live hardware telemetry radar card for webcam tracking mode."""
        canvas = np.zeros((345, 460, 3), dtype=np.uint8)
        canvas[:, :] = (22, 27, 34) # Dark panel background

        # Header title
        cv2.putText(canvas, "LIVE HARDWARE-IN-THE-LOOP (HIL) RADAR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (88, 166, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (15, 33), (445, 33), (48, 54, 61), 1)

        # Crosshairs / Horizon Indicator
        center_x, center_y = 230, 150
        cv2.circle(canvas, (center_x, center_y), 70, (48, 54, 61), 1)
        cv2.circle(canvas, (center_x, center_y), 40, (48, 54, 61), 1)
        cv2.line(canvas, (center_x - 85, center_y), (center_x + 85, center_y), (48, 54, 61), 1)
        cv2.line(canvas, (center_x, center_y - 85), (center_x, center_y + 85), (48, 54, 61), 1)

        # Draw Gimbal Pan / Tilt Vector Indicator
        pan_px = int(center_x + self.live_pan * 3.0)
        tilt_px = int(center_y - self.live_tilt * 3.0)
        cv2.circle(canvas, (pan_px, tilt_px), 5, (240, 136, 62), -1)
        cv2.putText(canvas, f"GIMBAL ({self.live_pan:+.1f}deg, {self.live_tilt:+.1f}deg)", (pan_px + 8, tilt_px - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 136, 62), 1)

        # Draw Detected Optical Target Indicator
        if proc.detection.detected:
            # Map 640x480 camera to local radar
            scale_x = 140.0 / 640.0
            scale_y = 140.0 / 480.0
            t_x = int(center_x + (proc.detection.x - 320.0) * scale_x)
            t_y = int(center_y + (proc.detection.y - 240.0) * scale_y)
            cv2.circle(canvas, (t_x, t_y), 6, (63, 185, 80), 2)
            cv2.line(canvas, (pan_px, tilt_px), (t_x, t_y), (163, 113, 247), 1)
            cv2.putText(canvas, f"BEACON LOCK (Conf: {proc.detection.confidence*100:.0f}%)", (15, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (63, 185, 80), 1)
        else:
            cv2.putText(canvas, "OPTICAL SEARCH (Waiting for laser / beacon)", (15, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 153, 34), 1)

        # Hardware connection readouts
        cam_text = f"CAMERA: Active Webcam (640x480 @ {proc.fps:.1f} FPS)"
        cv2.putText(canvas, cam_text, (15, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 246, 252), 1)

        ptu_status = f"PTU: Linked ({self.real_ptu.port})" if (self.real_ptu and self.real_ptu.is_connected) else "PTU: Virtual Servo Mode (Ready for Arduino)"
        ptu_color = (63, 185, 80) if (self.real_ptu and self.real_ptu.is_connected) else (139, 148, 158)
        cv2.putText(canvas, ptu_status, (15, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ptu_color, 1)

        err_text = f"ALIGNMENT ERROR: {proc.tracking_error_px:.1f} px ({proc.angular_error_deg:.2f} deg)"
        cv2.putText(canvas, err_text, (15, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (88, 166, 255), 1)

        return canvas

    def _sim_tick_loop(self):
        """High-FPS periodic animation and telemetry update loop."""
        if self.is_sim_running and not self.is_sim_paused:
            if self.input_mode == "WEBCAM":
                # --- LIVE HARDWARE WEBCAM MODE ---
                if self.real_camera and self.real_camera.is_connected:
                    raw_bgr = self.real_camera.get_raw_bgr_frame()
                    if raw_bgr is not None:
                        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
                        proc = self.pipeline.process_live_frame(gray, self.live_pan, self.live_tilt)
                        self.session_recorder.record_frame(proc)

                        # Update physical or virtual gimbal
                        if self.real_ptu and self.real_ptu.is_connected:
                            self.live_pan, self.live_tilt = self.real_ptu.set_rates(
                                proc.control.pan_rate_cmd_deg_s, proc.control.tilt_rate_cmd_deg_s, self.pipeline.dt
                            )
                        else:
                            self.live_pan += proc.control.pan_rate_cmd_deg_s * self.pipeline.dt
                            self.live_tilt += proc.control.tilt_rate_cmd_deg_s * self.pipeline.dt

                        # 1. Camera Canvas: Render HUD on live webcam frame
                        hud_frame = HUDVisualizer.render_camera_hud(raw_bgr, proc, self.pipeline.frame_trail)
                        self._display_image_on_label(hud_frame, self.cam_canvas, (460, 345))

                        # 2. World Canvas: Render live HIL radar card
                        diag_card = self._render_hardware_status_card(proc)
                        self._display_image_on_label(diag_card, self.world_canvas, (460, 345))

                        # 3. Telemetry & Plots
                        self._update_telemetry_readouts(proc)
                        if proc.frame_id % 5 == 0:
                            self._update_dashboard_plots()
            else:
                # --- VIRTUAL SIMULATION MODE ---
                if self.is_demo_mode:
                    self._update_demo_mode()

                # Execute pipeline step
                proc = self.pipeline.process_simulation_tick()
                self.session_recorder.record_frame(proc)

                # 1. Update Camera View Canvas
                reacq_box = None
                if proc.tracking_status in (TrackingStatus.TARGET_LOST, TrackingStatus.REACQUIRING):
                    reacq_box = self.pipeline.reacq_mgr.update(self.pipeline.dt, proc.kalman)[0]

                hud_frame = HUDVisualizer.render_camera_hud(
                    proc.raw_frame, proc, self.pipeline.frame_trail, search_roi=reacq_box
                )
                self._display_image_on_label(hud_frame, self.cam_canvas, (460, 345))

                # 2. Update World Map Canvas
                fov_bounds = self.pipeline.sim_engine.camera.get_world_fov_bounds(proc.ptu_pan, proc.ptu_tilt)
                world_bgr = self.pipeline.sim_engine.env.render_world_map(
                    beacon_world_pos=(proc.ground_truth.world_x, proc.ground_truth.world_y),
                    trajectory_trail=self.pipeline.sim_engine.trajectory.trail,
                    fov_bounds=fov_bounds,
                    camera_origin=self.pipeline.sim_engine.origin,
                    output_size=(460, 345)
                )
                self._display_image_on_label(world_bgr, self.world_canvas, (460, 345))

                # 3. Update Telemetry Badges
                self._update_telemetry_readouts(proc)

                # 4. Refresh Plots periodically (every 5 frames)
                if proc.frame_id % 5 == 0:
                    self._update_dashboard_plots()

        # Schedule next tick (30 FPS default)
        interval_ms = max(10, int(33 / max(0.25, self.sim_speed)))
        self.root.after(interval_ms, self._sim_tick_loop)

    def _display_image_on_label(self, cv_img_bgr: np.ndarray, label_widget: tk.Label, target_size: Tuple[int, int]):
        """Converts OpenCV BGR image to PhotoImage and displays on Tkinter label."""
        if cv_img_bgr is None or cv_img_bgr.size == 0:
            return
        rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    def _update_telemetry_readouts(self, proc: ProcessedFrame):
        st = proc.tracking_status.value
        self.lbl_global_status.configure(text=st)
        self.lbl_dash_status.configure(text=f"STATUS: {st}")

        if "TRACK" in st or "LOCK" in st:
            col = PALETTE["target_detected"]
        elif "LOST" in st:
            col = PALETTE["target_lost"]
        elif "REACQ" in st:
            col = PALETTE["reacquiring"]
        else:
            col = PALETTE["camera_fov"]
        self.lbl_global_status.configure(fg=col)
        self.lbl_dash_status.configure(fg=col)

        self.telemetry_labels["Tracking State"].configure(text=st, fg=col)
        self.telemetry_labels["Processing FPS"].configure(text=f"{proc.fps:.1f}")
        
        err_col = PALETTE["target_detected"] if proc.tracking_error_px <= 10.0 else PALETTE["target_lost"]
        self.telemetry_labels["Tracking Error"].configure(text=f"{proc.tracking_error_px:.1f} px", fg=err_col)
        self.telemetry_labels["Angular Error"].configure(text=f"{proc.angular_error_deg*60.0:.1f}' ({proc.angular_error_deg:.3f}°)")
        self.telemetry_labels["Detection Confidence"].configure(text=f"{proc.detection.confidence*100:.1f} %")
        self.telemetry_labels["PTU Pan Angle"].configure(text=f"{proc.ptu_pan:+.2f}°")
        self.telemetry_labels["PTU Tilt Angle"].configure(text=f"{proc.ptu_tilt:+.2f}°")
        self.telemetry_labels["Estimated Speed"].configure(text=f"{proc.kalman.speed:.1f} px/s")

        acq_time = self.pipeline.fsm.initial_acquisition_duration or 0.0
        self.telemetry_labels["Initial Acquisition"].configure(text=f"{acq_time:.2f} s")
        reacq_time = self.pipeline.fsm.last_reacquisition_duration or 0.0
        self.telemetry_labels["Reacquisition Time"].configure(text=f"{reacq_time:.2f} s")

        self.telemetry_labels["Lock Retention Rate"].configure(text=f"{self.pipeline.fsm.lock_retention_pct:.1f} %")
        loss_pct = (self.pipeline.fsm.target_loss_count / max(1, self.pipeline.fsm.total_frames)) * 100.0
        self.telemetry_labels["Target Loss Rate"].configure(text=f"{loss_pct:.1f} %")
        self.telemetry_labels["Control Strategy"].configure(text=proc.control.mode)

        # Update 3D Gimbal Visualizer
        if hasattr(self, "gimbal_3d"):
            self.gimbal_3d.update_angles(proc.ptu_pan, proc.ptu_tilt, is_locked=self.pipeline.fsm.is_locked)

        # Audio PAT telemetry cues on status transition
        if hasattr(self, "audio_alerts") and self.last_status != proc.tracking_status:
            if proc.tracking_status in (TrackingStatus.TRACKING, TrackingStatus.LOCK_RESTORED):
                self.audio_alerts.play_lock_acquired()
            elif proc.tracking_status in (TrackingStatus.TARGET_LOST, TrackingStatus.REACQUIRING):
                self.audio_alerts.play_target_lost()
            self.last_status = proc.tracking_status

    def _update_dashboard_plots(self):
        times = self.session_recorder.timestamps[-60:]
        if not times:
            return

        # 1. Errors
        errs = self.session_recorder.errors_px[-60:]
        self.line_err.set_data(times, errs)
        self.ax_err.set_xlim(times[0], times[-1] + 0.1)
        self.ax_err.set_ylim(0, max(18.0, max(errs) + 2.0))

        # 2. FPS
        fps_vals = self.session_recorder.fps_series[-60:]
        self.line_fps.set_data(times, fps_vals)
        self.ax_fps.set_xlim(times[0], times[-1] + 0.1)
        self.ax_fps.set_ylim(0, max(45.0, max(fps_vals) + 5.0))

        # 3. Pan & Tilt
        pans = self.session_recorder.pan_series[-60:]
        tilts = self.session_recorder.tilt_series[-60:]
        self.line_pan.set_data(times, pans)
        self.line_tilt.set_data(times, tilts)
        self.ax_angles.set_xlim(times[0], times[-1] + 0.1)
        min_ang = min(min(pans), min(tilts)) - 1.0
        max_ang = max(max(pans), max(tilts)) + 1.0
        self.ax_angles.set_ylim(min_ang, max_ang)

        self.dash_canvas.draw_idle()

    def toggle_simulation(self):
        if not self.is_sim_running:
            self.is_sim_running = True
            self.is_sim_paused = False
            self.btn_sim_play.configure(text="⏹ STOP", bg=PALETTE["target_lost"])
        else:
            self.is_sim_running = False
            self.btn_sim_play.configure(text="▶ START", bg=PALETTE["target_detected"])
            # Finalize session and prompt summary
            summary = self.session_recorder.finalize_summary(self.pipeline.fsm, self.config.to_dict())
            self._refresh_reports_table()

    def pause_simulation(self):
        self.is_sim_paused = not self.is_sim_paused
        txt = "▶ RESUME" if self.is_sim_paused else "⏸ PAUSE"
        self.btn_sim_pause.configure(text=txt)

    def reset_simulation(self):
        self.pipeline.reset()
        self.session_recorder = SessionRecorder("Reset Session", db_manager=self.db_manager)
        self.lbl_global_status.configure(text="RESET")

    def _on_speed_changed(self, event=None):
        val = self.speed_combo.get().replace("x", "")
        try:
            self.sim_speed = float(val)
        except ValueError:
            self.sim_speed = 1.0

    def trigger_sim_target_loss(self):
        self.pipeline.sim_engine.trigger_target_loss(duration_frames=25)

    def toggle_fog(self):
        cur_mode = self.pipeline.config.atmosphere.mode
        if cur_mode == AtmosphereMode.FOG.value:
            self.pipeline.config.atmosphere.mode = AtmosphereMode.CLEAR.value
            self.pipeline.config.atmosphere.intensity = 0.0
        else:
            self.pipeline.config.atmosphere.mode = AtmosphereMode.FOG.value
            self.pipeline.config.atmosphere.intensity = 0.50

    def toggle_jitter(self):
        if self.pipeline.config.disturbance.camera_jitter_amp > 0:
            self.pipeline.config.disturbance.camera_jitter_amp = 0.0
        else:
            self.pipeline.config.disturbance.camera_jitter_amp = 12.0

    # =========================================================================
    # DEMONSTRATION MODE (FOR SIH JUDGES)
    # =========================================================================
    def start_judge_demo(self):
        """One-click automated sequence demonstrating all innovations to SIH evaluators."""
        self.show_page("dashboard")
        self.reset_simulation()
        self.is_demo_mode = True
        self.demo_step = 1
        self.demo_timer = time.time()
        self.is_sim_running = True
        self.is_sim_paused = False
        self.btn_sim_play.configure(text="⏹ STOP DEMO", bg=PALETTE["target_lost"])
        messagebox.showinfo("SIH Demonstration Mode",
                            "Initiating Judge Demonstration Sequence:\n\n"
                            "1. Baseline Optical Lock\n"
                            "2. High-Speed Evasive Maneuver\n"
                            "3. Heavy Fog & Sensor Noise\n"
                            "4. Mechanical Platform Jitter\n"
                            "5. Laser Beam Occultation (Target Loss)\n"
                            "6. Predictive Uncertainty Reacquisition\n"
                            "7. Lock Restoration & Official Scorecard")

    def _update_demo_mode(self):
        elapsed = time.time() - self.demo_timer

        if self.demo_step == 1:
            self.lbl_dash_status.configure(text="DEMO 1/7: BASELINE ACQUISITION & TRACKING")
            if elapsed > 4.0:
                self.demo_step = 2
                self.demo_timer = time.time()
                # Fast circular maneuver
                self.pipeline.config.trajectory.type = TrajectoryType.CIRCULAR.value
                self.pipeline.config.trajectory.speed = 95.0
                self.pipeline.sim_engine.trajectory = self.pipeline.sim_engine.trajectory

        elif self.demo_step == 2:
            self.lbl_dash_status.configure(text="DEMO 2/7: HIGH-SPEED MANEUVER (PREDICTIVE CONTROL)")
            if elapsed > 5.0:
                self.demo_step = 3
                self.demo_timer = time.time()
                # Fog & Noise
                self.pipeline.config.atmosphere.mode = AtmosphereMode.FOG.value
                self.pipeline.config.atmosphere.intensity = 0.45
                self.pipeline.config.noise.salt_pepper = 0.08

        elif self.demo_step == 3:
            self.lbl_dash_status.configure(text="DEMO 3/7: ATMOSPHERIC FOG & SENSOR NOISE DEGRADATION")
            if elapsed > 5.0:
                self.demo_step = 4
                self.demo_timer = time.time()
                # Jitter
                self.pipeline.config.disturbance.camera_jitter_amp = 10.0

        elif self.demo_step == 4:
            self.lbl_dash_status.configure(text="DEMO 4/7: PLATFORM MECHANICAL JITTER PULSE")
            if elapsed > 4.0:
                self.demo_step = 5
                self.demo_timer = time.time()
                # Trigger target loss
                self.trigger_sim_target_loss()

        elif self.demo_step == 5:
            self.lbl_dash_status.configure(text="DEMO 5/7: OPTICAL OCCULTATION (TARGET LOST)")
            if elapsed > 1.2:
                self.demo_step = 6
                self.demo_timer = time.time()

        elif self.demo_step == 6:
            self.lbl_dash_status.configure(text="DEMO 6/7: PREDICTIVE REACQUISITION ACTIVE")
            if elapsed > 4.0:
                self.demo_step = 7
                self.demo_timer = time.time()
                self.is_demo_mode = False
                self.toggle_simulation()
                summary = self.session_recorder.finalize_summary(self.pipeline.fsm, self.config.to_dict())
                rep_text = PerformanceReportGenerator.format_text_report(summary)
                messagebox.showinfo("Demonstration Completed!", rep_text)
                self.show_page("reports")

    # =========================================================================
    # PAGE 3: VIDEO TESTING (MODE B)
    # =========================================================================
    def _build_video_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["video"] = page

        # Top Video Controls
        v_top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=12, pady=8)
        v_top.pack(fill=tk.X, pady=(0, 6))

        btn_open = tk.Button(v_top, text="📂 Open MP4 Video", font=("Segoe UI", 9, "bold"), bg=PALETTE["accent"], fg="#fff",
                             bd=0, padx=10, pady=4, cursor="hand2", command=self._open_video_file)
        btn_open.pack(side=tk.LEFT, padx=4)

        self.btn_v_play = tk.Button(v_top, text="▶ PLAY", font=("Segoe UI", 9, "bold"), bg=PALETTE["target_detected"], fg="#fff",
                                    bd=0, padx=10, pady=4, cursor="hand2", command=self._toggle_video_play)
        self.btn_v_play.pack(side=tk.LEFT, padx=4)

        btn_step_f = tk.Button(v_top, text="⏭ STEP", font=("Segoe UI", 9), bg=PALETTE["bg_card"], fg="#fff",
                               bd=0, padx=8, pady=4, cursor="hand2", command=self._video_step_forward)
        btn_step_f.pack(side=tk.LEFT, padx=4)

        btn_v_restart = tk.Button(v_top, text="⏮ RESTART", font=("Segoe UI", 9), bg=PALETTE["bg_card"], fg="#fff",
                                  bd=0, padx=8, pady=4, cursor="hand2", command=self._video_restart)
        btn_v_restart.pack(side=tk.LEFT, padx=4)

        self.lbl_video_info = tk.Label(v_top, text="No video loaded", font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"])
        self.lbl_video_info.pack(side=tk.LEFT, padx=15)

        btn_export_ann = tk.Button(v_top, text="💾 Export Annotated MP4", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                                   bd=0, padx=10, pady=4, cursor="hand2", command=self._export_annotated_video)
        btn_export_ann.pack(side=tk.RIGHT, padx=4)

        # Video Viewport
        v_body = tk.Frame(page, bg=PALETTE["bg_dark"])
        v_body.pack(fill=tk.BOTH, expand=True)

        v_screen_frame = tk.Frame(v_body, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=8, pady=8)
        v_screen_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(v_screen_frame, text="VIDEO FRAME TRACKING DIAGNOSTICS", font=("Segoe UI", 9, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 4))
        self.video_canvas = tk.Label(v_screen_frame, bg="#000000")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Video Telemetry Panel
        v_telemetry = tk.Frame(v_body, bg=PALETTE["bg_panel"], width=280, bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=12, pady=10)
        v_telemetry.pack(side=tk.RIGHT, fill=tk.Y, padx=4)
        v_telemetry.pack_propagate(False)

        tk.Label(v_telemetry, text="MODE B TELEMETRY", font=("Segoe UI", 10, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 8))
        self.lbl_v_stat = tk.Label(v_telemetry, text="Status: IDLE\nFrame: 0/0\nBeacon Detected: NO\nCentroid: N/A\nConfidence: 0.0%\nSimulated Pan: 0.0°\nSimulated Tilt: 0.0°",
                                   font=("Consolas", 9), fg=PALETTE["text_primary"], bg=PALETTE["bg_panel"], justify="left")
        self.lbl_v_stat.pack(anchor="w", pady=5)

    def _open_video_file(self):
        path = filedialog.askopenfilename(title="Select Benchmark Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not path:
            return
        if self.video_reader.open(path):
            self.lbl_video_info.configure(text=f"{os.path.basename(path)} | {self.video_reader.total_frames} frames @ {self.video_reader.fps:.1f} FPS")
            self._video_step_forward()
        else:
            messagebox.showerror("Video Error", "Could not open selected video file.")

    def _toggle_video_play(self):
        if not self.video_reader.is_opened:
            return
        self.is_video_running = not self.is_video_running
        txt = "⏸ PAUSE" if self.is_video_running else "▶ PLAY"
        self.btn_v_play.configure(text=txt)
        if self.is_video_running:
            self._video_play_loop()

    def _video_play_loop(self):
        if not self.is_video_running:
            return
        ret = self._video_step_forward()
        if ret:
            self.root.after(int(1000 / self.video_reader.fps), self._video_play_loop)
        else:
            self.is_video_running = False
            self.btn_v_play.configure(text="▶ PLAY")

    def _video_step_forward(self) -> bool:
        if not self.video_reader.is_opened:
            return False
        ret, frame_gray, f_idx = self.video_reader.read_frame()
        if not ret or frame_gray is None:
            return False

        # Run Detection & Tracking on video frame
        det = self.pipeline.detector.detect(frame_gray)
        k_state, pred, _ = self.pipeline.target_mgr.process_measurement(det, dt=1.0/self.video_reader.fps)
        status = self.pipeline.fsm.update(det.detected, det.confidence, 1.0/self.video_reader.fps)

        sim_pan, sim_tilt = 0.0, 0.0
        if det.detected:
            sim_pan, sim_tilt = self.video_reader.update_simulated_ptu(det.x, det.y, 1.0/self.video_reader.fps)

        proc = ProcessedFrame(
            frame_id=f_idx,
            timestamp=f_idx / self.video_reader.fps,
            raw_frame=frame_gray,
            detection=det,
            kalman=k_state,
            prediction=pred,
            tracking_status=status,
            tracking_error_px=0.0,
            fps=self.video_reader.fps,
            ptu_pan=sim_pan,
            ptu_tilt=sim_tilt
        )

        hud_bgr = HUDVisualizer.render_camera_hud(frame_gray, proc, self.pipeline.frame_trail)
        self._display_image_on_label(hud_bgr, self.video_canvas, (640, 480))

        # Update telemetry
        txt = (
            f"Status: {status.value}\n"
            f"Frame: {f_idx}/{self.video_reader.total_frames}\n"
            f"Beacon Detected: {'YES' if det.detected else 'NO'}\n"
            f"Centroid: ({det.x:.1f}, {det.y:.1f})\n"
            f"Confidence: {det.confidence*100:.1f}%\n"
            f"Simulated Pan: {sim_pan:+.2f}°\n"
            f"Simulated Tilt: {sim_tilt:+.2f}°"
        )
        self.lbl_v_stat.configure(text=txt)
        return True

    def _video_restart(self):
        if self.video_reader.is_opened:
            self.video_reader.restart()
            self._video_step_forward()

    def _export_annotated_video(self):
        if not self.video_reader.is_opened:
            messagebox.showwarning("Warning", "Please open a video file first.")
            return

        out_path = filedialog.asksaveasfilename(title="Save Annotated Video", defaultextension=".mp4",
                                               filetypes=[("MP4 Video", "*.mp4")])
        if not out_path:
            return

        writer = AnnotatedVideoWriter(out_path, width=self.video_reader.width, height=self.video_reader.height, fps=self.video_reader.fps)
        self.video_reader.restart()

        while True:
            ret, frame, idx = self.video_reader.read_frame()
            if not ret or frame is None:
                break
            det = self.pipeline.detector.detect(frame)
            k_state, pred, _ = self.pipeline.target_mgr.process_measurement(det, dt=1.0/self.video_reader.fps)
            status = self.pipeline.fsm.update(det.detected, det.confidence, 1.0/self.video_reader.fps)
            sim_pan, sim_tilt = 0.0, 0.0
            if det.detected:
                sim_pan, sim_tilt = self.video_reader.update_simulated_ptu(det.x, det.y, 1.0/self.video_reader.fps)

            proc = ProcessedFrame(
                frame_id=idx,
                timestamp=idx / self.video_reader.fps,
                raw_frame=frame,
                detection=det,
                kalman=k_state,
                prediction=pred,
                tracking_status=status,
                fps=self.video_reader.fps,
                ptu_pan=sim_pan,
                ptu_tilt=sim_tilt
            )
            writer.write_frame(frame, proc, self.pipeline.frame_trail)

        writer.close()
        self.video_reader.restart()
        messagebox.showinfo("Export Complete", f"Annotated video saved successfully to:\n{out_path}")

    # =========================================================================
    # PAGE 4: SCENARIO BUILDER
    # =========================================================================
    def _build_builder_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["builder"] = page

        top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=15, pady=12)
        top.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top, text="SCENARIO & DISTURBANCE BUILDER", font=("Segoe UI", 12, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")
        tk.Label(top, text="Configure target kinematics, noise parameters, atmospheric conditions, and controller limits.",
                 font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"]).pack(anchor="w")

        # Preset Loader
        row_preset = tk.Frame(top, bg=PALETTE["bg_panel"])
        row_preset.pack(fill=tk.X, pady=(8, 0))
        tk.Label(row_preset, text="Select Preset:", font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_panel"]).pack(side=tk.LEFT, padx=(0, 8))
        self.preset_combo = ttk.Combobox(row_preset, values=["Easy (Straight, Clear)", "Moderate (Circular, Noise)",
                                                             "Hard (Figure-8, Fog, Jitter)", "Extreme (Stress Test)",
                                                             "Reacquisition Challenge"], width=28, state="readonly")
        self.preset_combo.set("Moderate (Circular, Noise)")
        self.preset_combo.pack(side=tk.LEFT, padx=5)
        btn_apply_preset = tk.Button(row_preset, text="Load Preset", font=("Segoe UI", 8, "bold"), bg=PALETTE["accent"], fg="#fff",
                                     bd=0, padx=10, pady=3, cursor="hand2", command=self._apply_preset_from_combo)
        btn_apply_preset.pack(side=tk.LEFT, padx=5)

        # Form Notebook
        form_frame = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=20, pady=15)
        form_frame.pack(fill=tk.BOTH, expand=True)

        # Grid of parameters
        form_frame.columnconfigure((0, 1), weight=1)

        # Column 1: Trajectory & Kinematics
        c1 = tk.Frame(form_frame, bg=PALETTE["bg_panel"])
        c1.grid(row=0, column=0, sticky="nsew", padx=10)
        tk.Label(c1, text="BEACON & TRAJECTORY", font=("Segoe UI", 10, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 6))

        tk.Label(c1, text="Trajectory Type:", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w")
        self.traj_combo = ttk.Combobox(c1, values=[TrajectoryType.STRAIGHT.value, TrajectoryType.CIRCULAR.value,
                                                   TrajectoryType.FIGURE8.value, TrajectoryType.SINUSOIDAL.value,
                                                   TrajectoryType.RANDOM.value], width=20, state="readonly")
        self.traj_combo.set(TrajectoryType.CIRCULAR.value)
        self.traj_combo.pack(anchor="w", pady=2)

        tk.Label(c1, text="Speed (pixels/s):", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(4, 0))
        self.scale_speed = tk.Scale(c1, from_=20, to=150, orient=tk.HORIZONTAL, bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], highlightthickness=0)
        self.scale_speed.set(60)
        self.scale_speed.pack(fill=tk.X)

        tk.Label(c1, text="Beacon Size (pixels):", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(4, 0))
        self.scale_beacon_sz = tk.Scale(c1, from_=5, to=20, orient=tk.HORIZONTAL, bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], highlightthickness=0)
        self.scale_beacon_sz.set(10)
        self.scale_beacon_sz.pack(fill=tk.X)

        # Column 2: Disturbances & Atmosphere
        c2 = tk.Frame(form_frame, bg=PALETTE["bg_panel"])
        c2.grid(row=0, column=1, sticky="nsew", padx=10)
        tk.Label(c2, text="DISTURBANCES & ATMOSPHERE", font=("Segoe UI", 10, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 6))

        tk.Label(c2, text="Atmospheric Mode:", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w")
        self.atmo_combo = ttk.Combobox(c2, values=[AtmosphereMode.CLEAR.value, AtmosphereMode.HAZE.value,
                                                   AtmosphereMode.FOG.value, AtmosphereMode.RAIN.value,
                                                   AtmosphereMode.LOW_LIGHT.value], width=20, state="readonly")
        self.atmo_combo.set(AtmosphereMode.CLEAR.value)
        self.atmo_combo.pack(anchor="w", pady=2)

        tk.Label(c2, text="Fog / Atmosphere Intensity (%):", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(4, 0))
        self.scale_atmo_int = tk.Scale(c2, from_=0, to=100, orient=tk.HORIZONTAL, bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], highlightthickness=0)
        self.scale_atmo_int.set(0)
        self.scale_atmo_int.pack(fill=tk.X)

        tk.Label(c2, text="Salt & Pepper Noise (%):", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(4, 0))
        self.scale_sp_noise = tk.Scale(c2, from_=0, to=20, orient=tk.HORIZONTAL, bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], highlightthickness=0)
        self.scale_sp_noise.set(0)
        self.scale_sp_noise.pack(fill=tk.X)

        tk.Label(c2, text="Camera Mounting Jitter (± pixels):", font=("Segoe UI", 8), bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(4, 0))
        self.scale_jitter = tk.Scale(c2, from_=0, to=20, orient=tk.HORIZONTAL, bg=PALETTE["bg_panel"], fg=PALETTE["text_primary"], highlightthickness=0)
        self.scale_jitter.set(0)
        self.scale_jitter.pack(fill=tk.X)

        btn_save_apply = tk.Button(form_frame, text="✔ APPLY SETTINGS TO SIMULATION", font=("Segoe UI", 10, "bold"),
                                   bg=PALETTE["target_detected"], fg="#fff", bd=0, padx=15, pady=8, cursor="hand2", command=self._apply_builder_settings)
        btn_save_apply.grid(row=1, column=0, columnspan=2, pady=20)

    def _apply_preset_from_combo(self):
        val = self.preset_combo.get()
        if "Easy" in val:
            self.config = AstraConfig.preset_easy()
        elif "Moderate" in val:
            self.config = AstraConfig.preset_moderate()
        elif "Hard" in val:
            self.config = AstraConfig.preset_hard()
        elif "Extreme" in val:
            self.config = AstraConfig.preset_extreme()
        elif "Reacquisition" in val:
            self.config = AstraConfig.preset_reacquisition()

        self.pipeline = TrackingPipeline(self.config)
        self.reset_simulation()
        messagebox.showinfo("Preset Loaded", f"Loaded preset: {val}")

    def _apply_builder_settings(self):
        self.config.trajectory.type = self.traj_combo.get()
        self.config.trajectory.speed = float(self.scale_speed.get())
        self.config.beacon.size = int(self.scale_beacon_sz.get())
        self.config.atmosphere.mode = self.atmo_combo.get()
        self.config.atmosphere.intensity = float(self.scale_atmo_int.get()) / 100.0
        self.config.noise.salt_pepper = float(self.scale_sp_noise.get()) / 100.0
        self.config.disturbance.camera_jitter_amp = float(self.scale_jitter.get())

        self.pipeline = TrackingPipeline(self.config)
        self.reset_simulation()
        messagebox.showinfo("Config Applied", "Scenario settings updated and loaded into simulation engine!")

    # =========================================================================
    # PAGE 5: SYNTHETIC VIDEO GENERATOR
    # =========================================================================
    def _build_generator_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["generator"] = page

        top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=15, pady=12)
        top.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top, text="SYNTHETIC BENCHMARK VIDEO GENERATOR", font=("Segoe UI", 12, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")
        tk.Label(top, text="Synthesize test MP4 videos alongside ground truth CSV files for repeatable benchmarking.",
                 font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"]).pack(anchor="w")

        # 9 Preset Generation Buttons
        grid_frame = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=20, pady=20)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(grid_frame, text="ONE-CLICK BENCHMARK SCENARIO GENERATORS", font=("Segoe UI", 10, "bold"), fg=PALETTE["text_primary"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(0, 12))

        btn_grid = tk.Frame(grid_frame, bg=PALETTE["bg_panel"])
        btn_grid.pack(fill=tk.X, pady=5)
        btn_grid.columnconfigure((0, 1, 2), weight=1, uniform="b")

        presets = [
            ("1. Straight Test", SyntheticVideoGenerator.generate_straight_test),
            ("2. Circular Test", SyntheticVideoGenerator.generate_circular_test),
            ("3. Figure-8 Test", SyntheticVideoGenerator.generate_figure8_test),
            ("4. Random Test", SyntheticVideoGenerator.generate_random_test),
            ("5. Noise Surge Test", SyntheticVideoGenerator.generate_noise_test),
            ("6. Fog Obscuration Test", SyntheticVideoGenerator.generate_fog_test),
            ("7. Camera Jitter Test", SyntheticVideoGenerator.generate_jitter_test),
            ("8. Target Loss & Reacq Test", SyntheticVideoGenerator.generate_target_loss_test),
            ("9. Full Stress Test", SyntheticVideoGenerator.generate_full_stress_test)
        ]

        for i, (name, fn) in enumerate(presets):
            row = i // 3
            col = i % 3
            btn = tk.Button(btn_grid, text=name, font=("Segoe UI", 9, "bold"), bg=PALETTE["bg_card"], fg="#fff",
                            bd=0, padx=12, pady=10, cursor="hand2", command=lambda f=fn, n=name: self._run_generator_preset(f, n))
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")

        # Progress Status Box
        self.lbl_gen_status = tk.Label(grid_frame, text="Ready to generate benchmark videos in reports/videos/ directory.",
                                       font=("Segoe UI", 9), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"])
        self.lbl_gen_status.pack(anchor="w", pady=(20, 5))

    def _run_generator_preset(self, gen_fn, name: str):
        self.lbl_gen_status.configure(text=f"Generating '{name}'... please wait...")
        self.root.update()

        def worker():
            try:
                res = gen_fn()
                msg = f"Generated successfully:\n\nVideo: {res['video']}\nGround Truth CSV: {res['csv']}\nConfig: {res['json']}"
                self.root.after(0, lambda: messagebox.showinfo("Generation Complete", msg))
                self.root.after(0, lambda: self.lbl_gen_status.configure(text=f"Completed '{name}'. Output saved to {res['video']}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Generation Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # =========================================================================
    # PAGE 6: PERFORMANCE ANALYTICS
    # =========================================================================
    def _build_analytics_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["analytics"] = page

        top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=15, pady=10)
        top.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top, text="PERFORMANCE ANALYTICS & DIAGNOSTICS", font=("Segoe UI", 12, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")

        # 4 In-Depth Analytical Charts
        chart_frame = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=10, pady=10)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        self.analytics_fig = Figure(figsize=(10, 6), dpi=100, facecolor=PALETTE["bg_panel"])
        self.analytics_fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, hspace=0.38, wspace=0.25)

        # Plot A: Error vs Time
        self.ax_an_err = self.analytics_fig.add_subplot(2, 2, 1)
        self.ax_an_err.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_err.set_title("Tracking Error Over Time (Target <= 10 px)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_err.tick_params(colors=PALETTE["text_secondary"], labelsize=8)

        # Plot B: Actual vs Filtered vs Predicted Trajectory
        self.ax_an_traj = self.analytics_fig.add_subplot(2, 2, 2)
        self.ax_an_traj.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_traj.set_title("Target Positions: Actual vs Filtered vs Predicted", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_traj.tick_params(colors=PALETTE["text_secondary"], labelsize=8)

        # Plot C: Pan and Tilt Actuator Slew Curves
        self.ax_an_ptu = self.analytics_fig.add_subplot(2, 2, 3)
        self.ax_an_ptu.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_ptu.set_title("PTU Angular Orientation (Pan & Tilt)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_ptu.tick_params(colors=PALETTE["text_secondary"], labelsize=8)

        # Plot D: Detection Confidence & SNR
        self.ax_an_conf = self.analytics_fig.add_subplot(2, 2, 4)
        self.ax_an_conf.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_conf.set_title("Optical Detection Confidence (%)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_conf.tick_params(colors=PALETTE["text_secondary"], labelsize=8)

        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_fig, master=chart_frame)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_refresh = tk.Button(top, text="🔄 Refresh Charts", font=("Segoe UI", 9, "bold"), bg=PALETTE["accent"], fg="#fff",
                                bd=0, padx=10, pady=4, cursor="hand2", command=self._refresh_analytics_charts)
        btn_refresh.pack(side=tk.RIGHT, padx=5)

    def _refresh_analytics_charts(self):
        times = self.session_recorder.timestamps
        if len(times) < 2:
            return

        # A. Error
        self.ax_an_err.clear()
        self.ax_an_err.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_err.set_title("Tracking Error Over Time (Target <= 10 px)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_err.tick_params(colors=PALETTE["text_secondary"], labelsize=8)
        self.ax_an_err.axhline(10.0, color="#f85149", linestyle="--", linewidth=1, label="SIH Limit")
        self.ax_an_err.plot(times, self.session_recorder.errors_px, color="#3fb950", linewidth=1.5)

        # B. Trajectory Comparison
        self.ax_an_traj.clear()
        self.ax_an_traj.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_traj.set_title("Sensor Space: Tracked vs Predicted Coordinates", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_traj.tick_params(colors=PALETTE["text_secondary"], labelsize=8)
        if self.session_recorder.tracked_points:
            tx = [p["x"] for p in self.session_recorder.tracked_points]
            ty = [p["y"] for p in self.session_recorder.tracked_points]
            self.ax_an_traj.plot(tx, ty, color="#2ea043", label="Tracked", linewidth=1.5)
        if self.session_recorder.predicted_points:
            px = [p["x"] for p in self.session_recorder.predicted_points]
            py = [p["y"] for p in self.session_recorder.predicted_points]
            self.ax_an_traj.plot(px, py, color="#58a6ff", linestyle=":", label="Predicted", linewidth=1.2)
        self.ax_an_traj.legend(loc="upper right", fontsize=7)

        # C. PTU
        self.ax_an_ptu.clear()
        self.ax_an_ptu.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_ptu.set_title("PTU Angular Orientation (Pan & Tilt)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_ptu.tick_params(colors=PALETTE["text_secondary"], labelsize=8)
        self.ax_an_ptu.plot(times, self.session_recorder.pan_series, color="#f0883e", label="Pan (°)")
        self.ax_an_ptu.plot(times, self.session_recorder.tilt_series, color="#a371f7", label="Tilt (°)")
        self.ax_an_ptu.legend(loc="upper right", fontsize=7)

        # D. Confidence
        self.ax_an_conf.clear()
        self.ax_an_conf.set_facecolor(PALETTE["bg_dark"])
        self.ax_an_conf.set_title("Optical Detection Confidence (%)", fontsize=9, color=PALETTE["text_primary"])
        self.ax_an_conf.tick_params(colors=PALETTE["text_secondary"], labelsize=8)
        conf_pct = [c * 100.0 for c in self.session_recorder.confidence_series]
        self.ax_an_conf.plot(times, conf_pct, color="#e3b341", linewidth=1.5)

        self.analytics_canvas.draw()

    # =========================================================================
    # PAGE 7: REPORTS & DATABASE
    # =========================================================================
    def _build_reports_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["reports"] = page

        top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=15, pady=12)
        top.pack(fill=tk.X, pady=(0, 10))
        tk.Label(top, text="HISTORICAL SIMULATION REPORTS & SQLITE DATABASE", font=("Segoe UI", 12, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")

        btn_box = tk.Frame(top, bg=PALETTE["bg_panel"])
        btn_box.pack(fill=tk.X, pady=(8, 0))

        tk.Button(btn_box, text="📄 Export PDF Report", font=("Segoe UI", 9, "bold"), bg=PALETTE["accent"], fg="#fff",
                  bd=0, padx=12, pady=4, cursor="hand2", command=self._export_pdf_report).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_box, text="📊 Export CSV Telemetry", font=("Segoe UI", 9), bg=PALETTE["bg_card"], fg="#fff",
                  bd=0, padx=10, pady=4, cursor="hand2", command=self._export_csv_report).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_box, text="📋 Export JSON Metadata", font=("Segoe UI", 9), bg=PALETTE["bg_card"], fg="#fff",
                  bd=0, padx=10, pady=4, cursor="hand2", command=self._export_json_report).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_box, text="🔄 Refresh Database", font=("Segoe UI", 9), bg=PALETTE["bg_card"], fg="#fff",
                  bd=0, padx=10, pady=4, cursor="hand2", command=self._refresh_reports_table).pack(side=tk.RIGHT, padx=4)

        # Treeview Table for SQLite records
        tbl_frame = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"])
        tbl_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("ID", "Timestamp", "Scenario", "Duration", "Avg Error", "Loss %", "Reacq Time", "Avg FPS", "Result")
        self.reports_tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=15)
        for c in cols:
            self.reports_tree.heading(c, text=c)
            self.reports_tree.column(c, width=100, anchor="center")
        self.reports_tree.column("Scenario", width=180, anchor="w")
        self.reports_tree.column("Timestamp", width=140, anchor="center")

        scroll = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL, command=self.reports_tree.yview)
        self.reports_tree.configure(yscrollcommand=scroll.set)
        self.reports_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_reports_table()

    def _refresh_reports_table(self):
        for item in self.reports_tree.get_children():
            self.reports_tree.delete(item)
        sims = self.db_manager.get_all_simulations()
        for s in sims:
            self.reports_tree.insert("", tk.END, values=(
                s.get("id"),
                s.get("timestamp"),
                s.get("scenario"),
                f"{s.get('duration', 0.0):.1f}s",
                f"{s.get('avg_error', 0.0):.2f} px",
                f"{s.get('loss_percentage', 0.0):.1f}%",
                f"{s.get('reacquisition_time', 0.0):.2f}s",
                f"{s.get('avg_fps', 30.0):.1f}",
                s.get("result", "PASS")
            ))

    def _export_pdf_report(self):
        summary = self.session_recorder.finalize_summary(self.pipeline.fsm, self.config.to_dict())
        out_path = filedialog.asksaveasfilename(title="Save PDF Performance Report", defaultextension=".pdf",
                                               filetypes=[("PDF Document", "*.pdf")])
        if not out_path:
            return

        # Save current analytics chart to temp image to embed in PDF
        temp_chart = "temp_chart.png"
        self._refresh_analytics_charts()
        self.analytics_fig.savefig(temp_chart, dpi=120)

        try:
            Exporter.export_pdf(out_path, summary, chart_image_path=temp_chart)
            messagebox.showinfo("Report Exported", f"Professional PDF Report successfully generated:\n{out_path}")
        finally:
            if os.path.exists(temp_chart):
                os.remove(temp_chart)

    def _export_csv_report(self):
        out_path = filedialog.asksaveasfilename(title="Save Telemetry CSV", defaultextension=".csv",
                                               filetypes=[("CSV File", "*.csv")])
        if not out_path:
            return
        Exporter.export_csv(out_path, self.session_recorder.frames_log)
        messagebox.showinfo("Export Complete", f"CSV telemetry saved to:\n{out_path}")

    def _export_json_report(self):
        summary = self.session_recorder.finalize_summary(self.pipeline.fsm, self.config.to_dict())
        out_path = filedialog.asksaveasfilename(title="Save JSON Metadata", defaultextension=".json",
                                               filetypes=[("JSON File", "*.json")])
        if not out_path:
            return
        Exporter.export_json(out_path, summary)
        messagebox.showinfo("Export Complete", f"JSON metadata saved to:\n{out_path}")

    # =========================================================================
    # PAGE 8: HARDWARE ARCHITECTURE (EXTENSION SPECIFICATION)
    # =========================================================================
    def _build_architecture_page(self):
        page = tk.Frame(self.content_frame, bg=PALETTE["bg_dark"])
        self.pages["architecture"] = page

        top = tk.Frame(page, bg=PALETTE["bg_panel"], bd=0, highlightthickness=1, highlightbackground=PALETTE["border"], padx=20, pady=15)
        top.pack(fill=tk.BOTH, expand=True)

        tk.Label(top, text="HARDWARE-AGNOSTIC ARCHITECTURE & PHYSICAL EXTENSION", font=("Segoe UI", 12, "bold"), fg=PALETTE["camera_fov"], bg=PALETTE["bg_panel"]).pack(anchor="w")
        tk.Label(top, text="Section 63 Compliance: Universal Camera & Gimbal Abstraction Interfaces for Physical Deployment",
                 font=("Segoe UI", 9), fg=PALETTE["text_secondary"], bg=PALETTE["bg_panel"]).pack(anchor="w", pady=(2, 15))

        schematic_txt = """
+-----------------------------------------------------------------------------------------+
|                               ASTRA-TRACK SYSTEM ARCHITECTURE                           |
+-----------------------------------------------------------------------------------------+

 CURRENT: VIRTUAL TESTBED MODE                       FUTURE: PHYSICAL HARDWARE DEPLOYMENT
 -----------------------------                       ------------------------------------
 
    +-------------------------+                         +-------------------------------+
    |  Scenario Engine / Sim  |                         | Physical Optical Sensor / Cam |
    |  (Beacon, World, Noise) |                         | (GigE, USB3, NIR/SWIR Sensor) |
    +------------+------------+                         +---------------+---------------+
                 |                                                      |
                 v                                                      v
    +-------------------------+                         +-------------------------------+
    |   CameraInterface       | <==== UNCHANGED CORE ===> |   CameraInterface             |
    |  (VirtualCameraAdapter) |                         |  (RealCameraAdapter / Video)  |
    +------------+------------+                         +---------------+---------------+
                 |                                                      |
                 v                                                      v
  =========================================================================================
  ||                             CORE TRACKING & PREDICTION PIPELINE                     ||
  ||   - Grayscale -> Denoise -> Adaptive Threshold -> Sub-Pixel Intensity Centroid      ||
  ||   - AI Candidate Verification -> 6-State Kinematic Discrete Kalman Filter           ||
  ||   - Forward Trajectory Predictor -> Multi-Stage Intelligent Reacquisition           ||
  ||   - Predictive Slew-Rate-Limited Feedforward Controller                             ||
  =========================================================================================
                 |                                                      |
                 v                                                      v
    +-------------------------+                         +-------------------------------+
    |    PTUInterface         | <==== UNCHANGED CORE ===> |    PTUInterface               |
    |   (VirtualPTUAdapter)   |                         |  (RealPTUAdapter - VISCA/COM) |
    +------------+------------+                         +---------------+---------------+
                 |                                                      |
                 v                                                      v
    +-------------------------+                         +-------------------------------+
    |  Virtual Pan-Tilt Unit  |                         |  Physical Precision Gimbal    |
    |  (Kinematic Motor Model)|                         |  (DC Servo / Stepper PTU)     |
    +-------------------------+                         +-------------------------------+

KEY ARCHITECTURAL GUARANTEE:
Zero lines of detection, tracking, prediction, or control code need to be modified when
transitioning from synthetic simulation to physical ISRO/DRDO mobile FSOC optical terminals!
"""
        lbl_schematic = tk.Label(top, text=schematic_txt, font=("Consolas", 9), fg=PALETTE["text_primary"],
                                 bg=PALETTE["bg_card"], justify="left", padx=15, pady=15, relief=tk.SOLID, bd=1)
        lbl_schematic.pack(fill=tk.BOTH, expand=True)

    # =========================================================================
    # QUICK BATCH BENCHMARK RUNNER
    # =========================================================================
    def run_quick_benchmark(self):
        self.show_page("reports")
        self.lbl_global_status.configure(text="RUNNING BENCHMARK BATCH...")
        self.root.update()

        def worker():
            summary = BenchmarkRunner.run_batch(count=5, duration_per_scenario_s=3.0)
            rep_text = (
                f"ASTRA-Track Batch Benchmark Finished!\n\n"
                f"Total Scenarios: {summary['total_scenarios']}\n"
                f"Passed: {summary['passed']}\n"
                f"Failed: {summary['failed']}\n"
                f"Pass Rate: {summary['pass_rate_pct']:.1f}%\n"
                f"Average Tracking Error: {summary['avg_tracking_error_px']:.2f} px\n"
                f"Average Throughput: {summary['avg_fps']:.1f} FPS\n"
            )
            self.root.after(0, lambda: messagebox.showinfo("Benchmark Completed", rep_text))
            self.root.after(0, self._refresh_reports_table)
            self.root.after(0, lambda: self.lbl_global_status.configure(text="BENCHMARK DONE"))

        threading.Thread(target=worker, daemon=True).start()


def launch_desktop_gui():
    """Entry point for launching Desktop GUI."""
    root = tk.Tk()
    app = AstraTrackApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_desktop_gui()
