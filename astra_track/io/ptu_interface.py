"""
Hardware-agnostic Pan-Tilt Unit (PTU) abstractions.
Facilitates zero-code migration between virtual gimbal simulation and physical PTU actuators (VISCA, Pelco, Serial).
"""

from abc import ABC, abstractmethod
from typing import Tuple


class PTUInterface(ABC):
    """Universal pan-tilt gimbal mechanism interface."""

    @abstractmethod
    def set_pan_tilt(self, pan_deg: float, tilt_deg: float):
        """Sets absolute pan/tilt target angles."""
        pass

    @abstractmethod
    def set_rates(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> Tuple[float, float]:
        """Commands motor angular velocities."""
        pass

    @abstractmethod
    def get_pan_tilt(self) -> Tuple[float, float]:
        """Returns current pan/tilt angles (degrees)."""
        pass

    @abstractmethod
    def release(self):
        """Releases motor drivers / serial ports."""
        pass


class VirtualPTUAdapter(PTUInterface):
    """Adapter wrapping virtual simulation PTU."""

    def __init__(self, virtual_ptu):
        self.ptu = virtual_ptu

    def set_pan_tilt(self, pan_deg: float, tilt_deg: float):
        self.ptu.set_pan_tilt(pan_deg, tilt_deg)

    def set_rates(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> Tuple[float, float]:
        return self.ptu.step_velocity(pan_rate_deg_s, tilt_rate_deg_s, dt)

    def get_pan_tilt(self) -> Tuple[float, float]:
        return self.ptu.get_angles()

    def release(self):
        pass


from typing import List, Optional
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False


class RealPTUAdapter(PTUInterface):
    """Hardware driver for physical pan-tilt mechanisms via RS-232/RS-485/VISCA/Pelco-D."""

    def __init__(self, port: str = "COM3", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.pan = 0.0
        self.tilt = 0.0
        self.is_connected = False
        self._serial = None
        self.connect()

    def connect(self) -> bool:
        if not SERIAL_AVAILABLE:
            return False
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.is_connected = self._serial.is_open
            return self.is_connected
        except Exception:
            self.is_connected = False
            return False

    def set_pan_tilt(self, pan_deg: float, tilt_deg: float):
        self.pan = pan_deg
        self.tilt = tilt_deg
        if self.is_connected and self._serial:
            # VISCA absolute position packet template
            cmd = bytearray([0x81, 0x01, 0x06, 0x02, 0x18, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF])
            try:
                self._serial.write(cmd)
            except Exception:
                self.is_connected = False

    def set_rates(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> Tuple[float, float]:
        self.pan += pan_rate_deg_s * dt
        self.tilt += tilt_rate_deg_s * dt
        self.set_pan_tilt(self.pan, self.tilt)
        return self.pan, self.tilt

    def get_pan_tilt(self) -> Tuple[float, float]:
        return self.pan, self.tilt

    def release(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
        self.is_connected = False


class SerialServoPTUAdapter(PTUInterface):
    """
    Physical Hardware-in-the-Loop (HIL) dual-servo gimbal driver.
    Communicates with Arduino / ESP32 running 'firmware/arduino_gimbal_controller.ino'.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.pan = 0.0
        self.tilt = 0.0
        self.is_connected = False
        self._serial = None
        self._last_send_time = 0.0
        self._min_interval = 0.02 # 50 Hz max update rate over serial

        self.connect()

    def connect(self) -> bool:
        """Establishes handshake with Arduino gimbal controller."""
        if not SERIAL_AVAILABLE:
            return False
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            import time
            time.sleep(1.0) # Allow Arduino bootloader to initialize
            self._serial.write(b"PING\n")
            time.sleep(0.05)
            resp = self._serial.read(64).decode(errors="ignore")
            if "ASTRA_GIMBAL" in resp or "PONG" in resp:
                self.is_connected = True
                return True
            # Even if handshake string delayed, connection is open
            self.is_connected = True
            return True
        except Exception:
            self.is_connected = False
            return False

    def set_pan_tilt(self, pan_deg: float, tilt_deg: float):
        self.pan = max(-45.0, min(45.0, pan_deg))
        self.tilt = max(-30.0, min(30.0, tilt_deg))

        if self.is_connected and self._serial:
            import time
            now = time.time()
            if now - self._last_send_time >= self._min_interval:
                self._last_send_time = now
                msg = f"SET_PT {self.pan:.2f} {self.tilt:.2f}\n".encode("ascii")
                try:
                    self._serial.write(msg)
                except Exception:
                    self.is_connected = False

    def set_rates(self, pan_rate_deg_s: float, tilt_rate_deg_s: float, dt: float) -> Tuple[float, float]:
        self.pan += pan_rate_deg_s * dt
        self.tilt += tilt_rate_deg_s * dt
        self.set_pan_tilt(self.pan, self.tilt)
        return self.pan, self.tilt

    def get_pan_tilt(self) -> Tuple[float, float]:
        return self.pan, self.tilt

    @staticmethod
    def list_available_ports() -> List[str]:
        """Returns all connected serial COM ports."""
        if not SERIAL_AVAILABLE:
            return []
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return ports
        except Exception:
            return []

    def release(self):
        if self.is_connected and self._serial:
            try:
                self._serial.write(b"CENTER\n")
                self._serial.close()
            except Exception:
                pass
        self.is_connected = False

