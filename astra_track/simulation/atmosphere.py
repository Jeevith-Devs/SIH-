"""
Atmospheric degradation simulation for FSOC optical propagation channels.
Models attenuation, fog scattering (Mie blur), haze contrast reduction, rain streaks, and low-light dimming.
"""

import cv2
import numpy as np
from astra_track.app.constants import AtmosphereMode
from astra_track.app.config import AtmosphereConfig


class AtmosphereSimulator:
    """Simulates atmospheric channel degradation on camera frames."""

    def __init__(self, config: AtmosphereConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0 or self.config.mode == AtmosphereMode.CLEAR.value:
            return frame

        mode = self.config.mode
        intensity = max(0.0, min(1.0, self.config.intensity))
        if intensity <= 0.001 and mode != AtmosphereMode.CLEAR.value:
            # Default intensity if unspecified
            intensity = 0.35

        h, w = frame.shape[:2]
        out = frame.astype(np.float32)

        if mode == AtmosphereMode.HAZE.value:
            # Contrast compression and airlight addition
            # I_out = I_in * (1 - 0.4 * intensity) + 40 * intensity
            out = out * (1.0 - 0.45 * intensity) + (50.0 * intensity)

        elif mode == AtmosphereMode.FOG.value:
            # Beer-Lambert optical attenuation + scattering blur + backscatter radiance
            # 1. Attenuation
            out = out * (1.0 - 0.70 * intensity) + (75.0 * intensity)
            # 2. Scattering blur (Gaussian kernel size proportional to fog density)
            k_size = int(5 + 16 * intensity)
            if k_size % 2 == 0:
                k_size += 1
            sigma = 1.0 + 4.0 * intensity
            out = cv2.GaussianBlur(out, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)

        elif mode == AtmosphereMode.RAIN.value:
            # Directional rain streaks
            num_streaks = int(80 + 350 * intensity)
            streak_len = int(10 + 25 * intensity)
            for _ in range(num_streaks):
                x1 = self.rng.integers(0, w)
                y1 = self.rng.integers(0, h)
                x2 = int(x1 + streak_len * 0.3)
                y2 = int(y1 + streak_len)
                brightness = float(self.rng.integers(120, 230))
                cv2.line(out, (x1, y1), (x2, y2), brightness, thickness=1)

        elif mode == AtmosphereMode.LOW_LIGHT.value:
            # Attenuation of overall scene irradiance
            factor = max(0.15, 1.0 - 0.85 * intensity)
            out = out * factor

        return np.clip(out, 0, 255).astype(np.uint8)
