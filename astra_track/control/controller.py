"""
Base controller interface for PTU closed-loop alignment.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from astra_track.core.frame import ControlCommand


class BaseController(ABC):
    """Abstract controller interface."""

    @abstractmethod
    def compute_command(
        self,
        target_x: float,
        target_y: float,
        center_x: float,
        center_y: float,
        dt: float
    ) -> ControlCommand:
        pass

    @abstractmethod
    def reset(self):
        pass
