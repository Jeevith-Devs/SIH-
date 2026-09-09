"""
Aerospace acoustic telemetry cues for PAT (Pointing, Acquisition, Tracking) status changes.
Runs non-blocking audio cues using native Windows audio APIs (winsound).
"""

import threading
import sys
from typing import Optional


class AudioAlertManager:
    """Non-blocking audio chime synthesizer for coarse alignment lock & loss events."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()

    def _play_tones_thread(self, tone_list):
        """Worker thread to play a sequence of (freq_hz, duration_ms) tones."""
        if not self.enabled or sys.platform != "win32":
            return
        try:
            import winsound
            for freq, dur in tone_list:
                winsound.Beep(int(freq), int(dur))
        except Exception:
            pass  # Non-critical: suppress any OS audio driver issues silently

    def _async_play(self, tone_list):
        if not self.enabled or sys.platform != "win32":
            return
        thread = threading.Thread(target=self._play_tones_thread, args=(tone_list,), daemon=True)
        thread.start()

    def play_lock_acquired(self):
        """Crisp high dual-tone chime: Target lock established."""
        # High C6 (1046 Hz) -> High E6 (1318 Hz)
        self._async_play([(1046, 75), (1318, 90)])

    def play_target_lost(self):
        """Subtle warning descent: Beam occultation / target lost."""
        # A4 (440 Hz) -> E4 (330 Hz)
        self._async_play([(440, 90), (330, 110)])

    def play_reacquired(self):
        """Rising triad confirmation chime: Reacquisition successful."""
        # A5 (880 Hz) -> D6 (1175 Hz) -> G6 (1568 Hz)
        self._async_play([(880, 50), (1175, 50), (1568, 80)])

    def play_calibration_beep(self):
        """Short neutral blip for UI feedback."""
        self._async_play([(1200, 40)])
