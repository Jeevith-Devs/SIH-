#!/usr/bin/env python
"""
ASTRA-Track: AI-Based Predictive Coarse Alignment & Tracking System for Mobile FSOC Terminals.
Smart India Hackathon 2026 Problem Statement 26169.

Usage:
    python main.py                  # Launches Desktop GUI
    python main.py --benchmark      # Runs 5-scenario automated benchmark
    python main.py --stress-test    # Runs multi-variable stress test battery
    python main.py --generate-video # Generates synthetic MP4 benchmark video
    python main.py --video <path>   # Tracks target in recorded video file
    python main.py --demo           # Executes SIH Judge Demonstration sequence
    python main.py --gimbal         # Launches standalone 3D Gimbal Visualizer
"""

import sys
import os

# Ensure repository root is on Python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astra_track.app.application import AstraApplication


def main():
    app = AstraApplication()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
