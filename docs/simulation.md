# ASTRA-Track Simulation Subsystem

## 1. 2D World Coordinate Space
The virtual environment simulates a configurable 2D optical propagation theater (default: $2000 \times 2000$ px, configurable from $500 \times 500$ to $4000 \times 4000$ px). Background celestial starfields are pre-generated with reproducible seeds.

## 2. Beacon Target Models
Optical bright spots are generated with configurable dimensions ($5 \times 5$ to $20 \times 20$ px, default $10 \times 10$ px), peak irradiance ($0\text{--}255$ DN), and intensity profiles (Gaussian laser divergence or square beam).

## 3. Mathematical Trajectory Generators
- **Straight Line**: Linear movement with symmetric boundary reversal.
- **Circular**: $x(t) = x_c + R\cos(\omega t), \; y(t) = y_c + R\sin(\omega t)$.
- **Figure-8**: Lissajous / Lemniscate curve $x(t) = x_c + A\sin(\omega t), \; y(t) = y_c + \frac{A}{2}\sin(2\omega t)$.
- **Sinusoidal**: Continuous cross-track wave motion $y(t) = y_c + A\sin(\omega t)$.
- **Spiral**: Expanding/contracting Archimedean spiral.
- **Random Walk**: 2D momentum-damped Ornstein-Uhlenbeck stochastic drift.

## 4. Atmospheric Degradation & Noise Injection
- **Clear**: Baseline propagation.
- **Haze**: Contrast compression and airlight radiance addition.
- **Fog**: Beer-Lambert optical attenuation + Mie scattering Gaussian blur + backscatter.
- **Rain**: Directional precipitation streaks.
- **Low Light**: Scene irradiance dimming.
- **Electronic Noise**: Salt-and-pepper impulse noise, Gaussian thermal/readout noise, and Poisson photon shot noise.
- **Platform Disturbances**: High-frequency mounting jitter ($\pm 20$ px max) and low-frequency platform drift ($\pm 20$ px/frame max).
