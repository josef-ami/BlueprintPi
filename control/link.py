"""
link.py - the Pi <-> STM32 wire.

Two binary frames and one ASCII protocol, all on /dev/ttyACM0:

    Pi  -> STM32   DRIVE  19 bytes, sync AA 55, 50 Hz   what to do
    STM32 -> Pi    TELEM  21 bytes, sync 55 AA, 50 Hz   what is happening
    both           parameter lines, ASCII               N / ?P / ?V / C
    STM32 -> Pi    '#' log lines, ASCII

Full field-by-field spec: docs/PI_STM32_PROTOCOL.md. If you change a layout
here, change it there and in firmware/ObstacleRound.cpp.

The sync words are byte-reversed between directions so a frame can never be
mistaken for one going the other way, and 0xAA is not a valid ASCII byte, so
a log line or a parameter line can never contain either pair. That is what
lets one port carry all four things: the reader hunts for 55 AA, and
everything it steps over on the way is text.

Transport is the Blackpill's native USB CDC, which enumerates as
/dev/ttyACM0 - a different namespace from the LiDAR's /dev/ttyUSB0, so the
two can never collide or swap order at boot.

Nothing here blocks on the port. Writes are non-blocking and a refused write
is retried on the next tick rather than stalling the 50 Hz feed.
"""

import struct
import threading
import time
from dataclasses import dataclass, field

try:
    import serial
except ImportError:                       # tests run with no pyserial
    serial = None

from .intent import ActionIntent, SteerMode

# ---------------------------------------------------------------- wire ----

DRIVE_SYNC = b"\xAA\x55"
TELEM_SYNC = b"\x55\xAA"
DRIVE_LEN = 19
TELEM_LEN = 21

# payload = everything between the sync word and the checksum
_DRIVE_FMT = "<BBhhBBHHHBB"      # seq flags heading steer speed lock l f r rev cmd
_TELEM_FMT = "<BBhhihBBI"        # seq status heading yaw odo servo floor tries boot
DRIVE_PAYLOAD = struct.calcsize(_DRIVE_FMT)     # 16
TELEM_PAYLOAD = struct.calcsize(_TELEM_FMT)     # 18

RANGE_NONE = 0xFFFF

# DRIVE flags
F_ENABLE = 0x01
F_LIDAR_OK = 0x02
F_CAM_OK = 0x04
F_PILLAR_SEEN = 0x08
F_MODE_MASK = 0x30
F_MODE_SHIFT = 4
F_REVERSE = 0x40

# TELEM status
S_ENABLED = 0x01
S_IMU_OK = 0x02
S_COLOUR_OK = 0x04
S_ARC_DONE = 0x08
S_RECOVERING = 0x10
S_RECOVER_CAPPED = 0x20
S_LINK_STALE = 0x40
S_PARAMS_PUSHED = 0x80

# CMD byte
CMD_NONE, CMD_REBOOT = 0, 1

# floor colour
FLOOR_NONE, FLOOR_ORANGE, FLOOR_BLUE = 0, 1, 2
FLOOR_NAMES = {FLOOR_NONE: "none", FLOOR_ORANGE: "orange", FLOOR_BLUE: "blue"}


def xor8(payload: bytes) -> int:
    x = 0
    for b in payload:
        x ^= b
    return x


def _ddeg(v):
    """degrees -> decidegrees, clamped to int16."""
    return max(-32768, min(32767, int(round(v * 10.0))))


def _range(mm):
    if mm is None:
        return RANGE_NONE
    try:
        v = int(round(mm))
    except (TypeError, ValueError, OverflowError):
        return RANGE_NONE
    return v if 0 <= v < RANGE_NONE else RANGE_NONE


# --------------------------------------------------------------- telemetry

@dataclass
class Telemetry:
    """The last TELEM frame, decoded. `stamp` is when it arrived."""
    stamp: float = 0.0
    seq: int = 0
    status: int = 0
    heading_deg: float = 0.0
    yaw_rate_dps: float = 0.0
    odo_ticks: int = 0
    servo_deg: float = 0.0
    floor: int = FLOOR_NONE
    recover_tries: int = 0
    boot_id: int = 0

    def fresh(self, now=None, stale_s=0.3):
        return (now or time.time()) - self.stamp < stale_s

    @property
    def enabled(self):
        return bool(self.status & S_ENABLED)

    @property
    def imu_ok(self):
        return bool(self.status & S_IMU_OK)

    @property
    def arc_done(self):
        return bool(self.status & S_ARC_DONE)

    @property
    def recovering(self):
        return bool(self.status & S_RECOVERING)

    @property
    def recover_capped(self):
        return bool(self.status & S_RECOVER_CAPPED)

    @property
    def link_stale(self):
        return bool(self.status & S_LINK_STALE)

    @property
    def floor_name(self):
        return FLOOR_NAMES.get(self.floor, "?")


def pack_drive(seq, intent: ActionIntent, left_mm, front_mm, right_mm, rev,
               lidar_ok, cam_ok, pillar_seen, cmd=CMD_NONE):
    """One DRIVE frame. `intent` should already have been through the mapper."""
    flags = (int(intent.mode) << F_MODE_SHIFT) & F_MODE_MASK
    if intent.mode != SteerMode.STOP and intent.speed_pwm > 0:
        flags |= F_ENABLE
    if lidar_ok:
        flags |= F_LIDAR_OK
    if cam_ok:
        flags |= F_CAM_OK
    if pillar_seen:
        flags |= F_PILLAR_SEEN
    if intent.reverse:
        flags |= F_REVERSE

    payload = struct.pack(
        _DRIVE_FMT, seq & 0xFF, flags,
        _ddeg(intent.target_heading_deg), _ddeg(intent.steer_deg),
        max(0, min(255, int(intent.speed_pwm))),
        max(0, min(255, int(round(intent.arc_lock * 100)))),
        _range(left_mm), _range(front_mm), _range(right_mm),
        rev & 0xFF, cmd & 0xFF)
    return DRIVE_SYNC + payload + bytes([xor8(payload)])


def unpack_telem(payload: bytes, stamp=None):
    """Decode a verified TELEM payload into a Telemetry."""
    (seq, status, heading, yaw, odo, servo, floor, tries,
     boot) = struct.unpack(_TELEM_FMT, payload)
    return Telemetry(stamp=stamp if stamp is not None else time.time(),
                     seq=seq, status=status, heading_deg=heading / 10.0,
                     yaw_rate_dps=yaw / 10.0, odo_ticks=odo,
                     servo_deg=servo / 10.0, floor=floor,
                     recover_tries=tries, boot_id=boot)


# -------------------------------------------------------------------- link

class Link:
    """Owns the serial port. One reader thread; writes happen on the caller's
    thread and never block.

    on_log     called with each '#' line the firmware prints
    on_param   called with each '!' line (params.StmParams.on_line)
    """

    def __init__(self, port, baud=115200, on_log=None, on_param=None):
        self.port, self.baud = port, baud
        self.on_log = on_log or (lambda line: None)
        self.on_param = on_param or (lambda line: None)
        self.ser = None
        self._halt = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._telem = Telemetry()
        self._retry = None            # a write the port would not take

        self.error = None
        self.rx_frames = 0
        self.rx_bad = 0
        self.rx_dropped = 0           # frames missing, by seq
        self.tx_frames = 0
        self._last_rx_seq = None
        self._tx_seq = 0

    # ---- lifecycle ----

    def open(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        self.ser = serial.Serial(self.port, self.baud, timeout=0,
                                 write_timeout=0)
        return self

    def start(self):
        self._thread = threading.Thread(target=self._read_loop, name="LinkRx",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass

    # ---- reading ----

    def telemetry(self):
        with self._lock:
            return self._telem

    def _read_loop(self):
        buf = bytearray()
        text = bytearray()
        while not self._halt.is_set():
            try:
                n = self.ser.in_waiting
                chunk = self.ser.read(n) if n else b""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                return
            if not chunk:
                time.sleep(0.002)
                continue
            buf += chunk

            # Hunt for the sync word. Everything stepped over is text, and is
            # accumulated until a newline so a log line split across two
            # reads still arrives whole.
            i = 0
            while i < len(buf):
                if buf[i] == 0x55 and i + 1 < len(buf) and buf[i + 1] == 0xAA:
                    if len(buf) - i < TELEM_LEN:
                        break                       # wait for the rest
                    frame = bytes(buf[i:i + TELEM_LEN])
                    payload = frame[2:2 + TELEM_PAYLOAD]
                    if xor8(payload) == frame[-1]:
                        self._accept(payload)
                        i += TELEM_LEN
                        continue
                    self.rx_bad += 1
                    # bad checksum: this was not a frame after all, so step
                    # one byte and keep hunting
                if buf[i] == 0x55 and i + 1 >= len(buf):
                    break                           # might be a sync, wait
                if buf[i] == 0x0A:                  # newline: a line is done
                    self._line(bytes(text))
                    text.clear()
                elif buf[i] != 0x0D:
                    if len(text) < 512:
                        text.append(buf[i])
                    else:
                        text.clear()                # no newline for ages
                i += 1
            del buf[:i]

    def _accept(self, payload):
        t = unpack_telem(payload)
        with self._lock:
            prev = self._last_rx_seq
            self._telem = t
        self.rx_frames += 1
        if prev is not None:
            gap = (t.seq - prev) & 0xFF
            if gap > 1:
                self.rx_dropped += gap - 1
        self._last_rx_seq = t.seq

    def _line(self, raw):
        try:
            s = raw.decode("ascii", "replace").strip()
        except Exception:
            return
        if not s:
            return
        if s.startswith("!"):
            self.on_param(s)
        else:
            self.on_log(s)

    # ---- writing ----

    def _write(self, data):
        """Non-blocking. False means the port was full; the caller retries."""
        try:
            self.ser.write(data)
            return True
        except Exception as e:
            if serial is not None and isinstance(e, serial.SerialTimeoutException):
                return False
            self.error = f"{type(e).__name__}: {e}"
            raise

    def send(self, intent, left_mm=None, front_mm=None, right_mm=None, rev=0,
             lidar_ok=False, cam_ok=False, pillar_seen=False, cmd=CMD_NONE):
        """One DRIVE frame. Returns True if it reached the port."""
        self._tx_seq = (self._tx_seq + 1) & 0xFF
        frame = pack_drive(self._tx_seq, intent, left_mm, front_mm, right_mm,
                           rev, lidar_ok, cam_ok, pillar_seen, cmd)
        if self._write(frame):
            self.tx_frames += 1
            return True
        return False

    def push_params(self, lines):
        """Write queued parameter lines, rate-limited by the caller. A line
        the port refuses is held and retried before the next batch, so the
        table can never be pushed out of order."""
        if self._retry is not None:
            if not self._write(self._retry):
                return 0
            self._retry = None
        sent = 0
        for line in lines:
            data = (line + "\n").encode("ascii")
            if not self._write(data):
                self._retry = data
                break
            sent += 1
        return sent

    def stats(self):
        return {"rx_frames": self.rx_frames, "rx_bad": self.rx_bad,
                "rx_dropped": self.rx_dropped, "tx_frames": self.tx_frames,
                "error": self.error}


def send_command_burst(port, baud, intent, frames=6, gap_s=0.02):
    """Open the port just long enough to repeat one command frame - for a
    REBOOT when no Link is running. Never used while a session holds the
    port."""
    if serial is None:
        raise RuntimeError("pyserial is not installed")
    with serial.Serial(port, baud, timeout=0, write_timeout=0.2) as ser:
        for i in range(frames):
            ser.write(pack_drive(i, intent, None, None, None, 0,
                                 False, False, False, CMD_REBOOT))
            time.sleep(gap_s)
