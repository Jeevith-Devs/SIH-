"""
Sensor noise simulation: Salt-and-pepper, Gaussian additive noise, and Poisson shot noise.
"""

import numpy as np
from astra_track.app.config import NoiseConfig


class NoiseSimulator:
    """Applies realistic optical detector and electronic noise models."""

    def __init__(self, config: NoiseConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply configured noise models to the grayscale sensor frame.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.astype(np.float32)

        # 1. Poisson shot noise (quantum photon arrival noise)
        if self.config.poisson:
            # Scale frame to expected photon count and draw Poisson samples
            scale = 0.5
            scaled = np.clip(out * scale, 0, None)
            noisy_photons = self.rng.poisson(scaled).astype(np.float32)
            out = np.clip(noisy_photons / scale, 0, 255)

        # 2. Gaussian thermal / readout noise
        if self.config.gaussian_std > 0.0:
            gauss = self.rng.normal(0.0, self.config.gaussian_std, out.shape).astype(np.float32)
            out += gauss

        # 3. Salt & Pepper impulse noise
        sp_prob = self.config.salt_pepper
        if sp_prob > 0.0:
            # Salt (hot pixels) and Pepper (dead pixels)
            rand_mask = self.rng.random(out.shape, dtype=np.float32)
            half_prob = sp_prob / 2.0
            out[rand_mask < half_prob] = 0.0             # Pepper
            out[(rand_mask >= half_prob) & (rand_mask < sp_prob)] = 255.0  # Salt

        return np.clip(out, 0, 255).astype(np.uint8)
