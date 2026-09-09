# ASTRA-Track User Manual

## 1. Launching the Application

### Graphical User Interface (GUI)
```bash
python main.py
```

### CLI Command Options
- **Run Automated SIH Benchmark Battery**:
  ```bash
  python main.py --benchmark --count 5
  ```
- **Run Multi-Variable Stress Test**:
  ```bash
  python main.py --stress-test --count 10
  ```
- **Generate Synthetic Benchmark Video**:
  ```bash
  python main.py --generate-video --preset hard --output reports/videos/figure8_fog.mp4
  ```
- **Track Recorded Video (Mode B)**:
  ```bash
  python main.py --video reports/videos/figure8_fog.mp4
  ```
- **Execute One-Click Judge Demonstration Sequence**:
  ```bash
  python main.py --demo
  ```

## 2. Navigating the Desktop Dashboard
1. **Home Overview**: System summary and quick-launch cards.
2. **Simulation Dashboard**:
   - **Start / Stop**: Controls real-time simulation loop.
   - **Playback Speed**: $0.25\times$, $0.5\times$, $1\times$, $2\times$, $4\times$.
   - **Disturbance Injection**: Click *Target Loss*, *Fog*, or *Jitter* in real time to observe adaptive response.
3. **Video Testing (Mode B)**: Load `.mp4`, `.avi`, or `.mov` files to execute frame-by-frame diagnostic tracking and export annotated video.
4. **Scenario Builder**: Customize world dimensions, target speeds, trajectory shapes, atmospheric fog, and noise.
5. **Synthetic Video Generator**: 9 one-click preset benchmark video generators.
6. **Performance Analytics**: Real-time multi-variable curves (Tracking Error, Trajectories, PTU Pan/Tilt, Confidence).
7. **Reports & Database**: View SQLite history, generate official PDF reports with graphs and export CSV/JSON.
8. **Hardware Architecture**: Technical schematics of universal hardware abstraction layer.
