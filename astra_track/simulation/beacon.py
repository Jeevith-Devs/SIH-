"""
Optical beacon model for FSOC terminal tracking simulation.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Beacon:
    """Optical transmitter beacon spot."""
    id: int = 1
    x: float = 1000.0            # World coordinates
    y: float = 1000.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    size: int = 10               # Pixel width/height (5 to 20 px)
    brightness: int = 255        # Peak intensity (0 to 255)
    shape: str = "square"        # "square" or "gaussian"
    visible: bool = True         # Toggle for occultation / target loss simulation
    is_designated: bool = True   # Primary target for multi-beacon scenarios
    modulation_frequency: float = 0.0  # Hz (0 = CW / unmodulated)
    modulation_depth: float = 1.0      # 0.0 to 1.0

    def render_patch(self, sim_time: float = 0.0) -> np.ndarray:
        """Render optical intensity profile for the beacon with optional carrier modulation."""
        sz = max(3, int(self.size))
        if sz % 2 == 0:
            sz += 1  # ensure odd size for symmetric center
        
        effective_brightness = float(self.brightness)
        if self.modulation_frequency > 0.0:
            # Sinusoidal carrier pulse modulation
            mod_factor = 1.0 - (self.modulation_depth / 2.0) * (1.0 - np.cos(2.0 * np.pi * self.modulation_frequency * sim_time))
            effective_brightness *= float(mod_factor)

        patch = np.zeros((sz, sz), dtype=np.float32)
        center = sz // 2
        
        if self.shape == "square":
            patch[:, :] = effective_brightness
        else:
            # Gaussian laser spot profile
            sigma = sz / 3.0
            y_idx, x_idx = np.ogrid[:sz, :sz]
            dist_sq = (x_idx - center)**2 + (y_idx - center)**2
            gaussian = np.exp(-dist_sq / (2.0 * sigma**2))
            patch = gaussian * effective_brightness
            
        return np.clip(patch, 0, 255).astype(np.uint8)
