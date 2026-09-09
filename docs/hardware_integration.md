# Hardware Integration & Extension Guide

## 1. Physical Hardware Migration Path (Section 63)
ASTRA-Track is designed from the ground up to support transition to real physical FSOC mobile optical terminals without modifying any detection, Kalman filtering, prediction, or control algorithms.

```text
                      Universal Abstraction Layer
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
    CameraInterface                                PTUInterface
    - get_frame()                                  - set_pan_tilt(pan, tilt)
    - get_resolution()                             - set_rates(pan_rate, tilt_rate)
            |                                           |
     +------+------+                             +------+------+
     |             |                             |             |
     v             v                             v             v
VirtualCamera  RealCameraAdapter            VirtualPTU     RealPTUAdapter
(Simulation)   (GigE / USB3 / SWIR)         (Kinematic)    (VISCA / RS-422)
```

## 2. Connecting Real Optical Sensors
To connect an optical camera:
```python
from astra_track.io.camera_interface import RealCameraAdapter

# Initialize USB3 or GigE Camera stream
real_cam = RealCameraAdapter(camera_index_or_url=0, width=640, height=480)
frame = real_cam.get_frame()
```

## 3. Connecting Physical Gimbal / Pan-Tilt Units
To command a physical serial PTU (VISCA / Pelco-D):
```python
from astra_track.io.ptu_interface import RealPTUAdapter

# Initialize serial port driver
real_ptu = RealPTUAdapter(port="COM3", baudrate=9600)
real_ptu.set_rates(pan_rate_deg_s=2.5, tilt_rate_deg_s=-1.2, dt=0.033)
```
