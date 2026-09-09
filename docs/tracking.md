# ASTRA-Track Tracking & Reacquisition Subsystem

## 1. Tracking Finite State Machine (FSM)
The tracking lifecycle is governed by a deterministic 9-state state machine:
- `INITIALIZING`: System boot and sensor calibration.
- `SEARCHING`: Scanning optical field of view for beacon candidates.
- `ACQUIRING`: Candidate spot detected; verifying temporal consistency ($N \ge 3$ hits).
- `TRACKING`: Coarse alignment lock confirmed; closed-loop predictive gimbal tracking active.
- `PREDICTING`: Short-duration missing frame ($< 5$ misses); Kalman extrapolation active.
- `TARGET_LOST`: Miss threshold exceeded ($N \ge 5$ misses); declaring lock lost.
- `REACQUIRING`: Initiating multi-stage uncertainty search.
- `LOCK_RESTORED`: Optical spot redetected; measuring recovery duration ($\le 1.0\text{ s}$ target).
- `ERROR`: Sensor or actuator fault condition.

## 2. Multi-Stage Intelligent Reacquisition
When occultation or cloud obscuration breaks optical line of sight, ASTRA-Track does not execute a blind full-frame sweep. Instead, it follows a 4-tier recovery hierarchy:
1. **Level 1 (0 to 0.25s) — Predicted ROI**: Tightly windowed search ROI centered at the Kalman extrapolated trajectory:
   $$\text{ROI} = [x_{pred} \pm 2.5\sigma_x, \; y_{pred} \pm 2.5\sigma_y]$$
2. **Level 2 (0.25 to 0.60s) — Expanded Uncertainty ROI**: Expands window to $5\sigma$ uncertainty boundary.
3. **Level 3 (0.60 to 1.20s) — Local Archimedean Spiral**: Steers gimbal in an expanding spiral pattern centered on the extrapolated line of sight.
4. **Level 4 (> 1.20s) — Global Raster Sweep**: Broad fallback raster sweep across the search sector.
