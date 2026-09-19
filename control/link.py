"""
link.py — the Pi<->STM32 wire.

Two binary frames, both framed the way openRound.py already proved out:
a 2-byte sync word, a fixed payload, and an xor8 over the payload so a
partial or corrupt frame is dropped rather than acted on.

    Pi  -> STM32   DRIVE  11 bytes, sync AA 55   (what to do)
    STM32 -> Pi    TELEM  22 bytes, sync 55 AA   (what is happening)

The sync words are byte-reversed between directions on purpose: on a link
where both sides can babble, a frame can never be mistaken for one going the
other way.

Full field-by-field spec: docs/PI_STM32_PROTOCOL.md

Transport is the Blackpill's native USB CDC, which enumerates as
/dev/ttyACM0. That is a different namespace from the lidar's /dev/ttyUSB0,
so the two can never collide.
"""

import math
import struct
import threading
import time
from dataclasses import dataclass, field

import serial

# ---------------------------------------------------------------- wire ----

DRIVE_SYNC = b"\xAA\x55"
TELEM_SYNC = b"\x55\xAA"
DRIVE_LEN = 11
TELEM_LEN = 22

TOF_INVALID = 0xFFFF

# flags byte, Pi -> STM32
F_ENABLE = 0x01        # 0 = coast/stop regardless of the rest
F_CLOSED_LOOP = 0x02   # 0 = open-loop duty, 1 = encoder PID on speed
F_MODE_SHIFT = 2       # bits 2-3 carry SteerMode

# status byte, STM32 -> Pi
S_ENABLED = 0x01
S_WATCHDOG = 0x02      # link went silent; motor was cut
S_BUTTON = 0x04        # start button has been pressed since boot
S_CLOSED_LOOP = 0x08
S_IMU_OK = 0x10
S_TOF_OK = 0x20
S_COLOUR_OK = 0x40

FLOOR_NONE, FLOOR_ORANGE, FLOOR_BLUE = 0, 1, 2


def _xor8(payload: bytes) -> int:
    x = 0
    for b in payload:
        x ^= b
    return x


def _ddeg(v: float) -> int:
    """degrees -> decidegrees, clamped to int16."""
    return max(-32768, min(32767, int(round(v * 10.0))))


def _i16(v: float) -> int:
    return max(-32768, min(32767, int(round(v))))


# ----------------------------------------------------------- telemetry ----

@dataclass
class Telemetry:
    """
    Everything the STM32 knows, decoded. This is the OTHER half of the FSM's
    world — worldstate.py gives you what the camera and lidar see, this gives
    you heading, odometry and the down-facing sensors.

    Fields are None / inf when the sensor reported invalid, never a fake zero.
    """
    stamp: float = 0.0            # Pi wall-clock when the frame was decoded
    seq_ack: int = 0              # last DriveCommand seq the STM32 saw
    enabled: bool = False
    watchdog_tripped: bool = False
    button_pressed: bool = False  # latched: start button pressed since boot
    closed_loop: bool = False
    imu_ok: bool = False
    tof_ok: bool = False
    colour_ok: bool = False

    distance_mm: float = 0.0      # cumulative signed odometry since boot
    speed_mmps: float = 0.0       # measured ground speed
    heading_deg: float = 0.0      # + = left of the boot heading, wrapped +-180
    yaw_rate_dps: float = 0.0

    tof_front_mm: float = float("inf")
    tof_left_mm: float = float("inf")
    tof_right_mm: float = float("inf")

    floor_colour: int = FLOOR_NONE

    def fresh(self, max_age_s: float = 0.25) -> bool:
        return (time.time() - self.stamp) < max_age_s


def parse_telemetry(payload: bytes) -> Telemetry:
    """payload = the 19 bytes between sync and checksum (bytes 2..20)."""
    (seq_ack, status, distance_mm, speed_mmps, heading_dd, yaw_dd,
     tof_f, tof_l, tof_r, floor) = struct.unpack("<BBihhhHHHB", payload)

    def tof(v):
        return float("inf") if v == TOF_INVALID else float(v)

    return Telemetry(
        stamp=time.time(),
        seq_ack=seq_ack,
        enabled=bool(status & S_ENABLED),
        watchdog_tripped=bool(status & S_WATCHDOG),
        button_pressed=bool(status & S_BUTTON),
        closed_loop=bool(status & S_CLOSED_LOOP),
        imu_ok=bool(status & S_IMU_OK),
        tof_ok=bool(status & S_TOF_OK),
        colour_ok=bool(status & S_COLOUR_OK),
        distance_mm=float(distance_mm),
        speed_mmps=float(speed_mmps),
        heading_deg=heading_dd / 10.0,
        yaw_rate_dps=yaw_dd / 10.0,
        tof_front_mm=tof(tof_f),
        tof_left_mm=tof(tof_l),
        tof_right_mm=tof(tof_r),
        floor_colour=floor,
    )


# ------------------------------------------------------------ the link ----

class Link(threading.Thread):
    """
    Owns the serial port. Sends DRIVE frames on demand, continuously reads and
    decodes TELEM frames in this thread, and exposes the newest one.

    The STM32 runs its own watchdog: if DRIVE frames stop arriving it cuts the
    motor by itself. This class never needs to guarantee delivery — it only
    needs to keep a steady cadence, which main.py does by calling send() once
    per control tick.
    """

    def __init__(self, port="/dev/ttyACM0", baud=115200):
        super().__init__(name="LinkThread", daemon=True)
        self.port_name = port
        self.baud = baud
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._telem = Telemetry()
        self._seq = 0
        self.rx_frames = 0
        self.rx_bad = 0
        self.error = None

    # ---- lifecycle ----

    def open(self):
        """Open the port. Raises serial.SerialException if the STM32 is absent."""
        self._ser = serial.Serial(self.port_name, self.baud, timeout=0.05)
        return self._ser

    def run(self):
        buf = bytearray()
        while not self._stop.is_set():
            if self._ser is None:
                time.sleep(0.05)
                continue
            try:
                chunk = self._ser.read(64)
            except Exception as e:                     # port yanked mid-run
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
                continue
            if chunk:
                buf.extend(chunk)
            # resync on the sync word, drop anything that fails the checksum
            while len(buf) >= TELEM_LEN:
                i = buf.find(TELEM_SYNC)
                if i < 0:
                    del buf[:-1]                       # keep a possible split sync
                    break
                if i > 0:
                    del buf[:i]
                if len(buf) < TELEM_LEN:
                    break
                frame = bytes(buf[:TELEM_LEN])
                payload, chk = frame[2:TELEM_LEN - 1], frame[TELEM_LEN - 1]
                if _xor8(payload) == chk:
                    try:
                        t = parse_telemetry(payload)
                        with self._lock:
                            self._telem = t
                        self.rx_frames += 1
                    except struct.error:
                        self.rx_bad += 1
                    del buf[:TELEM_LEN]
                else:
                    self.rx_bad += 1
                    del buf[:2]                        # drop sync, hunt again

    def stop(self):
        self._stop.set()

    def close(self):
        try:
            if self._ser is not None:
                self.send_stop()
                self._ser.close()
        except Exception:
            pass

    # ---- tx ----

    def send_intent(self, intent, steer_deg: float, speed_mmps: float):
        """
        Send one DRIVE frame. steer_deg / speed_mmps come from the mapper,
        already clamped; intent supplies the mode and heading.
        """
        self._seq = (self._seq + 1) & 0xFF
        flags = F_ENABLE
        if intent.closed_loop:
            flags |= F_CLOSED_LOOP
        flags |= (intent.steer_mode.value & 0x03) << F_MODE_SHIFT

        payload = struct.pack(
            "<BBhhh",
            self._seq, flags,
            _ddeg(steer_deg),
            _i16(speed_mmps),
            _ddeg(intent.target_heading_deg),
        )
        self._write(DRIVE_SYNC + payload + bytes([_xor8(payload)]))

    def send_stop(self):
        """Explicit all-stop: enable bit clear. Sent on shutdown."""
        self._seq = (self._seq + 1) & 0xFF
        payload = struct.pack("<BBhhh", self._seq, 0x00, 0, 0, 0)
        self._write(DRIVE_SYNC + payload + bytes([_xor8(payload)]))

    def _write(self, frame: bytes):
        if self._ser is None:
            return
        try:
            self._ser.write(frame)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    # ---- rx ----

    def telemetry(self) -> Telemetry:
        with self._lock:
            return self._telem
