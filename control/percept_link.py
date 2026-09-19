"""
percept_link.py — the Pi <-> STM32 wire for the obstacle lap.

Two binary frames, framed the way link.py already proved out: a 2-byte sync
word, a fixed payload, xor8 over the payload. The sync words are byte-reversed
between directions so a frame can never be mistaken for one going the other way.

    Pi  -> STM32   PERCEPT  16 bytes, sync AA 55   what we see, what to do
    STM32 -> Pi    TELEM    22 bytes, sync 55 AA   what is happening

DIFFERENT FROM link.py ON PURPOSE. In the obstacle lap the STM32 owns the state
machine (it is OpenRound.cpp with one state added), so the Pi does not send
steering angles or speeds — it sends PERCEPTION plus one solved manoeuvre. The
Pi is a supervisor, not a driver.

The STM32 keeps printing its '#' log lines on the same port. 0xAA is not a valid
ASCII byte, so a log line can never contain the TELEM sync word: everything the
frame hunter skips over is text, and this module hands it back as log lines.

Transport: Blackpill native USB CDC -> /dev/ttyACM0. A different namespace from
the lidar's /dev/ttyUSB0, so they can never collide or swap order at boot.
"""

import struct
import threading
import time
from dataclasses import dataclass

try:
    import serial
except ImportError:          # the codec and the constants are useful off-robot
    serial = None            # PerceptLink.open() is the only thing that needs it

PERCEPT_SYNC = b"\xAA\x55"
TELEM_SYNC = b"\x55\xAA"
PERCEPT_LEN = 16
TELEM_LEN = 22

# ---- PERCEPT flags (Pi -> STM32) ----
P_LIDAR_OK = 0x01
P_CAM_OK = 0x02
P_AVOID_MASK = 0x0C       # bits 2-3 carry the solver's action code
P_AVOID_SHIFT = 2
P_GREEN = 0x10            # 0 = RED, 1 = GREEN; only meaningful when avoiding
P_HELLO = 0x20            # first frame of the run: sensors are up

# ---- TELEM status bits (STM32 -> Pi) ----
S_RUNNING = 0x01
S_LIDAR_STALE = 0x02
S_LIDAR_DEAD = 0x04
S_DIR_LOCKED = 0x08
S_CLOCKWISE = 0x10
S_IMU_OK = 0x20
S_COLOUR_OK = 0x40
S_RECOVERING = 0x80

# ---- STM32 states. Must match the enum in ObstacleLap.cpp. ----
ST_BOOT, ST_HEADING, ST_AVOID, ST_TURN90, ST_FINISH, ST_RECOVER = range(6)
STATE_NAMES = ["BOOT", "HEADING", "AVOID", "TURN90", "FINISH", "RECOVER"]

INVALID_MM = 0xFFFF


def _xor8(payload: bytes) -> int:
    x = 0
    for b in payload:
        x ^= b
    return x


def _u16(mm) -> int:
    """mm -> uint16, with inf/None/out-of-range collapsing to the invalid code."""
    if mm is None or mm != mm or mm == float("inf"):
        return INVALID_MM
    v = int(round(mm))
    return INVALID_MM if not (0 <= v < INVALID_MM) else v


def _ddeg(v: float) -> int:
    return max(-32768, min(32767, int(round(v * 10.0))))


# ------------------------------------------------------------- telemetry ----

@dataclass
class Telemetry:
    stamp: float = 0.0
    seq_ack: int = 0
    running: bool = False
    lidar_stale: bool = True
    lidar_dead: bool = True
    dir_locked: bool = False
    clockwise: bool = True
    imu_ok: bool = False
    colour_ok: bool = False
    recovering: bool = False

    odo_mm: float = 0.0          # cumulative, never reset. Take differences.
    speed_mmps: float = 0.0
    heading_deg: float = 0.0     # + = left of the heading zeroed at boot
    yaw_rate_dps: float = 0.0
    front_mm: float = float("inf")
    state: int = ST_BOOT
    corner_count: int = 0
    avoid_remaining_mm: float = 0.0
    floor_colour: int = 0

    def fresh(self, max_age_s: float = 0.25) -> bool:
        return (time.time() - self.stamp) < max_age_s

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state] if self.state < len(STATE_NAMES) else "?"


def parse_telemetry(payload: bytes) -> Telemetry:
    """payload = the 19 bytes between the sync word and the checksum."""
    (seq, status, odo, speed, head_dd, yaw_dd, front,
     state, corners, avoid_rem, floor) = struct.unpack("<BBihhhHBBHB", payload)
    return Telemetry(
        stamp=time.time(), seq_ack=seq,
        running=bool(status & S_RUNNING),
        lidar_stale=bool(status & S_LIDAR_STALE),
        lidar_dead=bool(status & S_LIDAR_DEAD),
        dir_locked=bool(status & S_DIR_LOCKED),
        clockwise=bool(status & S_CLOCKWISE),
        imu_ok=bool(status & S_IMU_OK),
        colour_ok=bool(status & S_COLOUR_OK),
        recovering=bool(status & S_RECOVERING),
        odo_mm=float(odo), speed_mmps=float(speed),
        heading_deg=head_dd / 10.0, yaw_rate_dps=yaw_dd / 10.0,
        front_mm=float("inf") if front == INVALID_MM else float(front),
        state=state, corner_count=corners,
        avoid_remaining_mm=float(avoid_rem), floor_colour=floor,
    )


def pack_percept(seq: int, left_mm, front_mm, right_mm, rev: int,
                 lidar_ok: bool, cam_ok: bool, hello: bool,
                 action: int, green: bool,
                 target_heading_deg: float, leg_mm: float) -> bytes:
    flags = 0
    if lidar_ok:
        flags |= P_LIDAR_OK
    if cam_ok:
        flags |= P_CAM_OK
    if hello:
        flags |= P_HELLO
    if green:
        flags |= P_GREEN
    flags |= (action & 0x03) << P_AVOID_SHIFT
    payload = struct.pack("<BBHHHBhH", seq & 0xFF, flags,
                          _u16(left_mm), _u16(front_mm), _u16(right_mm),
                          rev & 0xFF, _ddeg(target_heading_deg),
                          max(0, min(0xFFFF, int(round(leg_mm)))))
    return PERCEPT_SYNC + payload + bytes([_xor8(payload)])


# -------------------------------------------------------------- the link ----

class PerceptLink(threading.Thread):
    """
    Owns the serial port. send() writes one PERCEPT frame; this thread reads and
    decodes TELEM continuously and exposes the newest one, plus any '#' log
    lines the firmware printed.

    Delivery is never guaranteed and never needs to be: the STM32 runs its own
    watchdog and stops by itself if PERCEPT frames stop arriving. All this class
    owes it is a steady cadence.
    """

    def __init__(self, port="/dev/ttyACM0", baud=115200, on_log=None):
        super().__init__(name="PerceptLink", daemon=True)
        self.port_name = port
        self.baud = baud
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._telem = Telemetry()
        self._seq = 0
        self._on_log = on_log or (lambda line: print("[stm32] " + line))
        self.rx_frames = 0
        self.rx_bad = 0
        self.tx_frames = 0
        self.error = None

    # ---- lifecycle ----

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed (pip install pyserial)")
        self._ser = serial.Serial(self.port_name, self.baud, timeout=0.02)
        return self._ser

    def stop(self):
        self._stop.set()

    def close(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass

    # ---- rx ----

    def run(self):
        buf = bytearray()
        text = bytearray()
        while not self._stop.is_set():
            if self._ser is None:
                time.sleep(0.02)
                continue
            try:
                chunk = self._ser.read(256)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
                continue
            if chunk:
                buf.extend(chunk)

            while True:
                i = buf.find(TELEM_SYNC)
                if i < 0:
                    # No frame start in sight. Everything but a possible split
                    # sync byte is firmware log text.
                    if len(buf) > 1:
                        text.extend(buf[:-1])
                        del buf[:-1]
                    break
                if i > 0:
                    text.extend(buf[:i])
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
                    del buf[:2]          # false sync; hunt again

            self._drain_text(text)

    def _drain_text(self, text: bytearray):
        while b"\n" in text:
            line, _, rest = bytes(text).partition(b"\n")
            del text[:len(line) + 1]
            s = line.decode("ascii", "replace").strip()
            if s:
                self._on_log(s)
        if len(text) > 512:             # no newline for ages: something is off
            del text[:]

    def telemetry(self) -> Telemetry:
        with self._lock:
            return self._telem

    # ---- tx ----

    def send(self, **kw) -> bytes:
        """Build and write one PERCEPT frame. kwargs go straight to pack_percept."""
        self._seq = (self._seq + 1) & 0xFF
        frame = pack_percept(self._seq, **kw)
        if self._ser is not None:
            try:
                self._ser.write(frame)
                self.tx_frames += 1
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        return frame
