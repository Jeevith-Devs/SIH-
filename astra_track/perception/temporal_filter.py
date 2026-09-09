"""
Temporal Carrier Frequency Discrimination and Decoy/Glint Rejection.
In Free-Space Optical Communications (FSOC), beacon lasers are modulated at a designated
carrier frequency (e.g., 10-50 Hz) to allow the coarse tracking receiver to cleanly
distinguish the cooperative beacon from ambient solar glints, city lights, and optical decoys.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


@dataclass
class TemporalAnalysisResult:
    """Spectral analysis result for a tracked candidate spot."""
    dominant_frequency_hz: float = 0.0
    spectral_purity: float = 0.0       # [0.0, 1.0] peak power ratio over AC spectrum
    modulation_depth: float = 0.0      # [0.0, 1.0] (I_max - I_min) / (I_max + I_min)
    target_frequency_power: float = 0.0 # Power fraction in expected carrier bin
    is_carrier_matched: bool = False
    is_static_glint: bool = False
    confidence_factor: float = 1.0     # Multiplier to scale candidate confidence


class TemporalBeaconFilter:
    """
    Rolling-window temporal spectral analyzer.
    Tracks candidate intensity profiles over time and evaluates modulation characteristics via FFT.
    """

    def __init__(
        self,
        fps: float = 30.0,
        buffer_size: int = 24,
        target_carrier_hz: float = 0.0,
        frequency_tolerance_hz: float = 2.5
    ):
        self.fps = float(fps)
        self.buffer_size = max(8, buffer_size)
        self.target_carrier_hz = float(target_carrier_hz)
        self.frequency_tolerance_hz = float(frequency_tolerance_hz)

        # Rolling intensity history per track ID: {track_id: list of intensities}
        self.history: Dict[int, List[float]] = {}
        # Rolling position history: {track_id: list of (x, y)}
        self.positions: Dict[int, List[Tuple[float, float]]] = {}

    def reset(self):
        """Clears all temporal track histories."""
        self.history.clear()
        self.positions.clear()

    def update_sample(self, track_id: int, intensity: float, position: Optional[Tuple[float, float]] = None):
        """Appends a new intensity sample for a given track identifier."""
        if track_id not in self.history:
            self.history[track_id] = []
            self.positions[track_id] = []

        self.history[track_id].append(float(intensity))
        if position is not None:
            self.positions[track_id].append(position)

        # Maintain rolling window buffer size
        if len(self.history[track_id]) > self.buffer_size:
            self.history[track_id].pop(0)
            if self.positions[track_id]:
                self.positions[track_id].pop(0)

    def analyze_track(self, track_id: int) -> TemporalAnalysisResult:
        """
        Performs spectral analysis on the rolling intensity window for a track.
        Returns TemporalAnalysisResult with frequency, modulation depth, and confidence factor.
        """
        samples = self.history.get(track_id, [])
        n_samples = len(samples)

        # Default result when insufficient temporal history is available (< 8 frames)
        if n_samples < 8:
            return TemporalAnalysisResult(
                dominant_frequency_hz=0.0,
                spectral_purity=0.0,
                modulation_depth=0.0,
                target_frequency_power=0.0,
                is_carrier_matched=(self.target_carrier_hz <= 0.0),
                is_static_glint=False,
                confidence_factor=1.0
            )

        signal = np.array(samples, dtype=np.float64)
        mean_intensity = float(np.mean(signal))
        min_intensity = float(np.min(signal))
        max_intensity = float(np.max(signal))

        # Modulation depth: (max - min) / (max + min + eps)
        eps = 1e-6
        modulation_depth = (max_intensity - min_intensity) / (max_intensity + min_intensity + eps)
        modulation_depth = float(np.clip(modulation_depth, 0.0, 1.0))

        # Check for static DC glint (negligible intensity fluctuation)
        intensity_std = float(np.std(signal))
        if intensity_std < 3.0 or modulation_depth < 0.08:
            # Static unmodulated light source (sun glint, streetlight, DC decoy)
            conf_factor = 0.35 if self.target_carrier_hz > 0.0 else 1.0
            return TemporalAnalysisResult(
                dominant_frequency_hz=0.0,
                spectral_purity=0.0,
                modulation_depth=modulation_depth,
                target_frequency_power=0.0,
                is_carrier_matched=False,
                is_static_glint=True,
                confidence_factor=conf_factor
            )

        # 1. AC Signal decomposition (remove DC component)
        ac_signal = signal - mean_intensity

        # Apply Hanning window to mitigate spectral leakage
        window = np.hanning(n_samples)
        windowed_ac = ac_signal * window

        # 2. Real FFT
        fft_vals = np.fft.rfft(windowed_ac)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.fps)
        power_spectrum = np.abs(fft_vals) ** 2

        # Ignore zero-frequency bin (DC)
        if len(power_spectrum) > 1:
            ac_powers = power_spectrum[1:]
            ac_freqs = freqs[1:]
        else:
            ac_powers = np.array([0.0])
            ac_freqs = np.array([0.0])

        total_ac_power = float(np.sum(ac_powers))
        if total_ac_power < eps:
            return TemporalAnalysisResult(
                dominant_frequency_hz=0.0,
                spectral_purity=0.0,
                modulation_depth=modulation_depth,
                confidence_factor=1.0
            )

        # Peak AC frequency
        max_idx = int(np.argmax(ac_powers))
        dominant_freq = float(ac_freqs[max_idx])
        peak_power = float(ac_powers[max_idx])
        spectral_purity = float(np.clip(peak_power / total_ac_power, 0.0, 1.0))

        # 3. Evaluate Target Carrier Frequency Match
        is_matched = False
        target_power_ratio = 0.0

        if self.target_carrier_hz > 0.0:
            # Find bins close to target carrier frequency
            target_mask = np.abs(ac_freqs - self.target_carrier_hz) <= self.frequency_tolerance_hz
            if np.any(target_mask):
                target_power = float(np.sum(ac_powers[target_mask]))
                target_power_ratio = float(target_power / total_ac_power)
                if target_power_ratio >= 0.35 and abs(dominant_freq - self.target_carrier_hz) <= self.frequency_tolerance_hz:
                    is_matched = True

            # Calculate confidence factor
            if is_matched:
                # Strong periodic match at expected carrier -> boost confidence (+30% to +40%)
                conf_factor = 1.0 + 0.35 * spectral_purity
            else:
                # Carrier mismatch (e.g. 50/60Hz mains flicker or arbitrary glint)
                conf_factor = max(0.20, 1.0 - 0.70 * (1.0 - target_power_ratio))
        else:
            # No specific carrier required; general periodic beacon check
            is_matched = (spectral_purity > 0.40 and modulation_depth > 0.15)
            conf_factor = 1.0 + 0.15 * spectral_purity

        return TemporalAnalysisResult(
            dominant_frequency_hz=round(dominant_freq, 2),
            spectral_purity=round(spectral_purity, 3),
            modulation_depth=round(modulation_depth, 3),
            target_frequency_power=round(target_power_ratio, 3),
            is_carrier_matched=is_matched,
            is_static_glint=False,
            confidence_factor=float(np.clip(conf_factor, 0.1, 1.5))
        )
