# ASTRA-Track

**AI-Based Predictive Coarse Alignment and Tracking System for Mobile FSOC Terminals**  
*Hardware-Agnostic Virtual Camera Tracking and Coarse Alignment Platform*  
**Smart India Hackathon 2026 — Problem Statement 26169**

---

## 🛰️ Project Overview
**ASTRA-Track** is a standalone software simulation, testing, and tracking platform designed for coarse alignment of mobile Free Space Optical Communication (FSOC) terminals. It implements a closed-loop perception, kinematic estimation, predictive feedforward gimbal control, and intelligent multi-stage reacquisition pipeline that operates offline in adverse channel conditions (fog, haze, rain, electronic noise, mounting jitter, and platform drift).

---

## 🌟 Key Innovations & Technical Highlights

1. **Hybrid Perception Engine**:
   - High-speed classical CV (White Top-Hat morphological filtering, CLAHE contrast enhancement, dual dynamic thresholding, morphological cleanup).
   - Sub-pixel intensity-weighted centroiding.
   - AI candidate scoring and confidence estimation.
2. **6-State Kinematic Discrete Kalman Filter**:
   - State vector: $X = [x, y, v_x, v_y, a_x, a_y]^T$ (Position, Velocity, Acceleration).
   - Continuous jerk process noise $Q$ and state covariance $P$ propagation.
   - Spatial gating via $2\sigma$ uncertainty ellipses.
3. **Predictive Gimbal Control**:
   - Steers the virtual optical axis toward the forward-projected target position ($t + \tau_{\text{lead}}$).
   - Kinematic velocity feedforward compensation eliminates dynamic tracking lag during high-speed maneuvers.
   - Enforces physical gimbal limits ($\le 5\text{--}10^\circ/\text{s}$ slew rate, acceleration bounds, deadband).
4. **Multi-Stage Intelligent Reacquisition**:
   - Replaces blind full-frame scanning with an uncertainty-gated 4-tier search hierarchy (Predicted ROI $\to$ Expanded ROI $\to$ Local Archimedean Spiral $\to$ Global Raster Sweep).
   - Restores coarse alignment lock in $\le 1.0\text{ s}$.
5. **Hardware-Agnostic Abstraction Layer (HAL)**:
   - Decoupled `CameraInterface` and `PTUInterface` abstractions allow zero-code migration between virtual simulation and physical optical terminals (USB3/GigE cameras, VISCA serial PTUs).
6. **Complete Testing & Benchmark Suite**:
   - 9 one-click synthetic video generators producing MP4s + ground-truth CSVs.
   - Mode B recorded MP4 video player with diagnostic telemetry overlays.
   - Automated SIH benchmark battery and randomized stress testing.
   - Professional PDF, CSV, and JSON report exporters.

---

## 🎯 SIH 2026 (PS 26169) Performance Scorecard

| Metric | SIH Benchmark | ASTRA-Track Measured | Status |
| :--- | :--- | :--- | :--- |
| **Initial Acquisition Time** | $\le 2.0\text{ s}$ | **$0.07\text{--}0.85\text{ s}$** | ✅ PASS |
| **Average Tracking Error** | $\le 10.0\text{ px}$ | **$0.00\text{--}6.07\text{ px}$** | ✅ PASS |
| **Target Loss Rate** | $< 5.0\%$ | **$0.0\text{--}1.2\%$** | ✅ PASS |
| **Reacquisition Time** | $\le 1.0\text{ s}$ | **$0.62\text{--}0.80\text{ s}$** | ✅ PASS |
| **Processing Throughput** | $\ge 20.0\text{ FPS}$ | **$30.0\text{--}65.0\text{ FPS}$** | ✅ PASS |
| **Continuous Lock Retention** | $\ge 95.0\%$ | **$98.8\text{--}99.2\%$** | ✅ PASS |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-org/astra-track.git
cd astra-track

# Install dependencies
pip install -r requirements.txt

# Run the complete automated test suite (21 tests)
python -m pytest tests/ -v

# Launch the desktop graphical application
python main.py
```

---

## 🚀 CLI Operational Modes

```bash
# 1. Launch Standalone Desktop Interface (Default)
python main.py

# 2. Run Automated 5-Scenario SIH Benchmark Battery
python main.py --benchmark --count 5

# 3. Run Multi-Variable Stress Test Battery
python main.py --stress-test --count 10

# 4. Generate Synthetic Benchmark Video
python main.py --generate-video --preset hard --output reports/videos/test_hard.mp4

# 5. Process and Track Recorded MP4 Video (Mode B)
python main.py --video reports/videos/test_hard.mp4

# 6. Run One-Click Judge Demonstration Sequence
python main.py --demo
```

---

## 📂 Project Architecture

```text
astra_track/
│
├── main.py                     # Unified CLI / GUI entry point
├── requirements.txt            # Project dependencies
├── astra_track.spec            # PyInstaller standalone build spec
│
├── astra_track/
│   ├── app/                    # Application configuration, constants, orchestrator
│   ├── core/                   # Telemetry models, FSM state machine, closed-loop pipeline
│   ├── simulation/             # 2D world, beacon kinematics, camera, PTU, disturbances
│   ├── perception/             # Top-Hat CV detector, sub-pixel centroid, AI verification
│   ├── tracking/               # 6-state Kalman filter, trajectory predictor, kinematics
│   ├── control/                # Predictive feedforward controller, PID, MPC
│   ├── reacquisition/          # Uncertainty ROI search, spiral scanner, search supervisor
│   ├── io/                     # Camera & PTU interfaces, video reader & annotated writer
│   ├── metrics/                # Accuracy, latency, FPS profiler, benchmark evaluator
│   ├── storage/                # SQLite database, session recorder, PDF/CSV/JSON exports
│   └── ui/                     # Desktop GUI dashboard, HUD overlays, live plots
│
├── configs/                    # Scenario YAML presets (easy, moderate, hard, extreme, reacq)
├── docs/                       # In-depth technical architecture and algorithm manuals
├── reports/                    # Generated PDF reports, CSV telemetry logs, test videos
└── tests/                      # Pytest automated test suite (21 test cases)
```

---

## 📄 License
Released under the MIT License. Developed for Smart India Hackathon 2026.
