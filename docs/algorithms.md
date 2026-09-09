# ASTRA-Track Algorithm Specifications

## 1. Classical Computer Vision Detection Pipeline
The perception engine implements an 8-stage deterministic optical spot extraction pipeline:
1. **Grayscale Conversion**: Eliminates color chromaticity and prepares raw sensor DN matrix.
2. **Top-Hat Morphological Filtering**: Applies White Top-Hat transform $T(f) = f - (f \circ b)$ using an elliptical structuring element to isolate narrow laser bright spots from low-frequency fog, clouds, and background irradiance.
3. **Contrast Enhancement**: CLAHE (Contrast-Limited Adaptive Histogram Equalization) boosts local SNR without saturating background noise.
4. **Dual Dynamic Thresholding**: Fuses Otsu thresholding on Top-Hat filtered space with adaptive local Gaussian thresholding ($I > \mu_{bg} + 1.5\sigma_{bg}$).
5. **Morphological Opening & Closing**: Suppresses isolated single-pixel salt-and-pepper impulses while bridging fragmented beacon cores.
6. **Connected Components & Contour Geometry**: Extracts spatial contours and computes area, bounding box, aspect ratio, and circularity.
7. **Multi-Parameter Candidate Filtering**: Rejects candidates violating optical beacon constraints ($4 \le \text{Area} \le 1200$, $0.25 \le \text{AR} \le 4.0$, $\text{Circularity} > 0.15$).
8. **Sub-Pixel Intensity Centroiding**:
   $$cx = \frac{\sum (I(x, y) - I_{bg}) \cdot x}{\sum (I(x, y) - I_{bg})}, \quad cy = \frac{\sum (I(x, y) - I_{bg}) \cdot y}{\sum (I(x, y) - I_{bg})}$$

## 2. 6-State Kinematic Discrete Kalman Filter
The tracking filter maintains state $X = [x, y, v_x, v_y, a_x, a_y]^T$:
- **State Transition Matrix $F$**:
  $$F = \begin{bmatrix}
  1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 & 0 \\
  0 & 1 & 0 & \Delta t & 0 & \frac{1}{2}\Delta t^2 \\
  0 & 0 & 1 & 0 & \Delta t & 0 \\
  0 & 0 & 0 & 1 & 0 & \Delta t \\
  0 & 0 & 0 & 0 & 1 & 0 \\
  0 & 0 & 0 & 0 & 0 & 1
  \end{bmatrix}$$
- **Covariance Propagation & Gating**:
  $$P_{pred} = F P F^T + Q$$
  $$K = P_{pred} H^T (H P_{pred} H^T + R)^{-1}$$
  $$X = X_{pred} + K (z - H X_{pred})$$

## 3. Predictive Lead Feedforward Control
To compensate for motor rate slewing lag during dynamic target maneuvers:
$$\dot{\theta}_{pan} = 0.6 \cdot v_x \frac{\text{HFOV}}{W} + K_p \cdot \Delta \theta_{lead, pan}$$
$$\dot{\theta}_{tilt} = 0.6 \cdot v_y \frac{\text{VFOV}}{H} + K_p \cdot \Delta \theta_{lead, tilt}$$
Clamped strictly to physical gimbal velocity bounds ($\le 5\text{--}10^\circ/\text{s}$) and acceleration slew limits.
