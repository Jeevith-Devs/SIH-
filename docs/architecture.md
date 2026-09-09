# ASTRA-Track System Architecture

## 1. High-Level Architecture
**ASTRA-Track** is engineered as a modular, hardware-agnostic simulation, coarse alignment, and predictive tracking platform for mobile Free Space Optical Communication (FSOC) optical terminals (Smart India Hackathon 2026 Problem Statement 26169).

```text
                  +-----------------------------------+
                  |          Scenario Engine          |
                  |  (2D World, Beacon, Disturbances) |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |      Camera Sensor Frame          |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |         Perception Engine         |
                  |  - Top-Hat Morphological Filter   |
                  |  - CLAHE Contrast Enhancement     |
                  |  - Dual Adaptive Thresholding     |
                  |  - Sub-Pixel Intensity Centroid   |
                  |  - AI Candidate Verification      |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |       6-State Kalman Filter       |
                  |  State: [x, y, vx, vy, ax, ay]^T  |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |    Trajectory Lead Predictor      |
                  |     (N-Step Forward Horizon)      |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |    Predictive Slew Controller     |
                  |  (Feedforward + Gated Feedback)   |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |    Pan-Tilt Mechanism (PTU)       |
                  | (Speed Limit: 5-10 deg/s, Accel)  |
                  +-----------------+-----------------+
                                    |
                                    +---> Closed-Loop Feedback to Camera
```

## 2. Hardware Abstraction Layer (HAL)
The system is decoupled from virtual simulation primitives through universal interfaces:
- **`CameraInterface`**: Abstract receiver interface implemented by `VirtualCameraAdapter` (synthetic simulation) and `RealCameraAdapter` (USB3, GigE Vision, SWIR/NIR optical camera).
- **`PTUInterface`**: Abstract gimbal actuator interface implemented by `VirtualPTUAdapter` and `RealPTUAdapter` (VISCA / RS-422 / Pelco-D serial PTU).

Zero core code modifications are required when migrating from simulation to physical optical terminals.
