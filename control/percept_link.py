"""
percept_link.py — the Pi <-> STM32 wire for the obstacle lap.

Binary frames, framed the way link.py already proved out: a 2-byte sync word,
a fixed payload, xor8 over the payload. The sync words are byte-reversed between
directions so a frame can never be mistaken for one going the other way.

    Pi  -> STM32   PERCEPT  17 bytes, sync AA 55   what we see, what to do,
                                                   plus a CMD byte (byte 15)
    STM32 -> Pi    TELEM    22 bytes, sync 55 AA   what is happening
    STM32 -> Pi    STATUS   61 bytes, sync 55 A5   FSM internals, 10 Hz,
                                                   reporting only (dashboard)

CMD (PERCEPT byte 15): 0 NONE, 1 RERUN, 2 STOP, 3 REBOOT.
    RERUN   taken on a 0 -> 1 edge while the car is FINISHED or STOPPED. Hold it
            until TELEM shows BOOT, then drop back to 0.
    STOP    taken after CMD_CONFIRM_FRAMES identical frames, from any state but
            FINISH/STOPPED. Hold it until TELEM shows STOPPED (or FINISH).
    REBOOT  taken after CMD_CONFIRM_FRAMES identical frames: motor off, then an
            MCU reset. USB drops and re-enumerates.
A command frame need not carry ranges: sent with LIDAR_OK clear, it refreshes
the STM32's link clock but never its lidar clock.

DIFFERENT FROM link.py ON PURPOSE. In the obstacle lap the STM32 owns the state
machine (it is OpenRound.cpp with one state added), so the Pi does not send
steering angles or speeds — it sends PERCEPTION plus one solved manoeuvre. The
Pi is a supervisor, not a driver.

The STM32 keeps printing its '#' log lines on the same port. 0xAA and 0xA5 are
not valid ASCII bytes, so a log line can never contain the TELEM or STATUS sync
word: everything the frame hunter skips over is text, and this module hands it
back as log lines. Firmware that predates STATUS simply never sends it;
status() then stays None and everything else works as before.

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
PERCEPT_LEN = 17
TELEM_LEN = 22
STATUS_SYNC = b"\x55\xA5"
STATUS_LEN = 61
STATUS_FMT = "<8B4H4h6B12HI"      # the 58 bytes between the sync word and the xor

# ---- PERCEPT byte 15 (Pi -> STM32). Must match ObstacleLap.cpp. ----
CMD_NONE, CMD_RERUN, CMD_STOP, CMD_REBOOT = 0, 1, 2, 3
CMD_NAMES = {CMD_NONE: "NONE", CMD_RERUN: "RERUN", CMD_STOP: "STOP",
             CMD_REBOOT: "REBOOT"}
CMD_CONFIRM_FRAMES = 3             # STOP / REBOOT need this many in a row

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
S_COLOUR_OK = 0x40          # never set any more: the colour sensor is not read
S_RECOVERING = 0x80

# ---- STATUS flags (STM32 -> Pi). Must match ObstacleLap.cpp. ----
SF_FSM_STARTED = 0x01
SF_BOOT_READY = 0x02
SF_LIDAR_HOLD = 0x04
SF_LINK_STALE = 0x08
SF_BLIND = 0x10
SF_RERUN_ARMED = 0x20
SF_WALL_SEEN_L = 0x40
SF_WALL_SEEN_R = 0x80
SF2_REAL_OPEN_L = 0x01
SF2_REAL_OPEN_R = 0x02
SF2_LOCK_NEEDS_REAL = 0x04
PF_HELLO = 0x01
PF_LIDAR_OK = 0x02
PF_CAM_OK = 0x04
PF_GREEN = 0x08
PF_ACTION_SHIFT = 4

# ---- STM32 states. Must match the enum in ObstacleLap.cpp. ----
ST_BOOT, ST_HEADING, ST_AVOID, ST_TURN90, ST_FINISH, ST_RECOVER, ST_STOPPED = range(7)
STATE_NAMES = ["BOOT", "HEADING", "AVOID", "TURN90", "FINISH", "RECOVER", "STOPPED"]

# why the car is standing still (FinishReason in the .cpp)
FINISH_NONE, FINISH_BY_WALL, FINISH_BY_ODO, FINISH_BY_STOP_CMD = range(4)
FINISH_NAMES = ["", "wall", "odo", "STOP command"]

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


# ---------------------------------------------------------------- status ----

def _mm_or_inf(v: int) -> float:
    return float("inf") if v == INVALID_MM else float(v)


def _age_or_none(v: int):
    return None if v == INVALID_MM else v


@dataclass
class FsmStatus:
    """
    The STM32 FSM's internals, from the 10 Hz STATUS frame. Reporting only:
    nothing on the Pi decides anything from this; the dashboard shows it.
    Per-state fields read 0 outside the state they belong to (sendStatus()).
    """
    stamp: float = 0.0
    version: int = 0
    state: int = ST_BOOT
    fsm_started: bool = False
    boot_ready: bool = False         # BOOT: Pi ready, 5 s countdown running
    lidar_hold: bool = False         # HEADING parked the car: lidar dead
    link_stale: bool = True
    blind: bool = False              # this run's GO had the camera down
    rerun_armed: bool = False        # FINISH/STOPPED: a RERUN edge would be taken
    wall_seen_l: bool = False
    wall_seen_r: bool = False
    real_open_l: bool = False
    real_open_r: bool = False
    lock_needs_real: bool = False    # the LOCK_NEEDS_REAL_RETURN constant
    p_hello: bool = False            # the last PERCEPT, as the STM32 decoded it
    p_lidar_ok: bool = False
    p_cam_ok: bool = False
    p_green: bool = False
    p_action: int = 0
    p_cmd: int = CMD_NONE
    p_cmd_run: int = 0               # consecutive frames carrying p_cmd
    run_number: int = 0              # GOs since power-up
    state_ms: int = 0
    countdown_ms: int = 0            # BOOT
    grace_ms: int = 0                # BOOT
    avoid_ms: int = 0                # AVOID, vs the 5 s timeout
    lane_heading_deg: float = 0.0
    target_heading_deg: float = 0.0
    servo_deg: float = 76.5          # servo command; < straight = left
    motor_pwm: int = 0
    open_revs_l: int = 0
    open_revs_r: int = 0
    recover_tries: int = 0
    recover_return_state: int = ST_HEADING
    finish_reason: int = FINISH_NONE
    lidar_l_mm: float = float("inf")
    lidar_r_mm: float = float("inf")
    straight_mm: int = 0
    corner_lockout_mm: int = 0
    avoid_lockout_mm: int = 0
    segment_mm: int = 0
    finish_target_mm: int = 0
    front_at_start_mm: float = float("inf")
    phase_mm: int = 0                # AVOID travelled / TURN90 arc / RECOVER backed
    avoid_leg_mm: int = 0
    link_age_ms: object = None       # None = never received
    lidar_age_ms: object = None
    percept_frames: int = 0

    def fresh(self, max_age_s: float = 0.5) -> bool:
        return (time.time() - self.stamp) < max_age_s

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state] if self.state < len(STATE_NAMES) else "?"


def parse_status(payload: bytes) -> FsmStatus:
    """payload = the 58 bytes between the STATUS sync word and the checksum."""
    (ver, state, fl, fl2, pf, pcmd, pcmd_run, run_no,
     state_ms, countdown, grace, avoid_ms,
     lane_dd, tgt_dd, servo_dd, pwm,
     revs_l, revs_r, rec_tries, rec_ret, fin, _reserved,
     lid_l, lid_r, straight, corner_lock, avoid_lock, segment, fin_tgt,
     front_start, phase, leg, link_age, lidar_age,
     frames) = struct.unpack(STATUS_FMT, payload)
    return FsmStatus(
        stamp=time.time(), version=ver, state=state,
        fsm_started=bool(fl & SF_FSM_STARTED), boot_ready=bool(fl & SF_BOOT_READY),
        lidar_hold=bool(fl & SF_LIDAR_HOLD), link_stale=bool(fl & SF_LINK_STALE),
        blind=bool(fl & SF_BLIND), rerun_armed=bool(fl & SF_RERUN_ARMED),
        wall_seen_l=bool(fl & SF_WALL_SEEN_L), wall_seen_r=bool(fl & SF_WALL_SEEN_R),
        real_open_l=bool(fl2 & SF2_REAL_OPEN_L), real_open_r=bool(fl2 & SF2_REAL_OPEN_R),
        lock_needs_real=bool(fl2 & SF2_LOCK_NEEDS_REAL),
        p_hello=bool(pf & PF_HELLO), p_lidar_ok=bool(pf & PF_LIDAR_OK),
        p_cam_ok=bool(pf & PF_CAM_OK), p_green=bool(pf & PF_GREEN),
        p_action=(pf >> PF_ACTION_SHIFT) & 0x03,
        p_cmd=pcmd, p_cmd_run=pcmd_run, run_number=run_no,
        state_ms=state_ms, countdown_ms=countdown, grace_ms=grace, avoid_ms=avoid_ms,
        lane_heading_deg=lane_dd / 10.0, target_heading_deg=tgt_dd / 10.0,
        servo_deg=servo_dd / 10.0, motor_pwm=pwm,
        open_revs_l=revs_l, open_revs_r=revs_r, recover_tries=rec_tries,
        recover_return_state=rec_ret, finish_reason=fin,
        lidar_l_mm=_mm_or_inf(lid_l), lidar_r_mm=_mm_or_inf(lid_r),
        straight_mm=straight, corner_lockout_mm=corner_lock,
        avoid_lockout_mm=avoid_lock, segment_mm=segment, finish_target_mm=fin_tgt,
        front_at_start_mm=_mm_or_inf(front_start), phase_mm=phase, avoid_leg_mm=leg,
        link_age_ms=_age_or_none(link_age), lidar_age_ms=_age_or_none(lidar_age),
        percept_frames=frames,
    )


def pack_percept(seq: int, left_mm, front_mm, right_mm, rev: int,
                 lidar_ok: bool, cam_ok: bool, hello: bool,
                 action: int, green: bool,
                 target_heading_deg: float, leg_mm: float,
                 cmd: int = CMD_NONE) -> bytes:
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
    payload = struct.pack("<BBHHHBhHB", seq & 0xFF, flags,
                          _u16(left_mm), _u16(front_mm), _u16(right_mm),
                          rev & 0xFF, _ddeg(target_heading_deg),
                          max(0, min(0xFFFF, int(round(leg_mm)))),
                          cmd & 0xFF)
    return PERCEPT_SYNC + payload + bytes([_xor8(payload)])


def pack_command_frame(seq: int, cmd: int) -> bytes:
    """A PERCEPT frame that carries only a command: no ranges, LIDAR_OK and
    CAM_OK clear, no manoeuvre. Used when there is no live perception to send."""
    return pack_percept(seq, None, None, None, 0, False, False, False,
                        0, False, 0.0, 0, cmd=cmd)


def send_command_burst(port: str, cmd: int, frames: int = 2 * CMD_CONFIRM_FRAMES,
                       period_s: float = 0.02, baud: int = 115200) -> None:
    """
    Open the port, send `frames` command-only PERCEPT frames `period_s` apart,
    flush, close. For when no PerceptLink holds the port (e.g. REBOOT with no
    session running). Raises on open failure. Closing promptly matters for
    REBOOT: if the old handle is still open when the board re-enumerates,
    Linux hands out /dev/ttyACM1 instead of ACM0.
    """
    if serial is None:
        raise RuntimeError("pyserial is not installed (pip install pyserial)")
    ser = serial.Serial(port, baud, timeout=0.1, write_timeout=0.5)
    try:
        for i in range(frames):
            ser.write(pack_command_frame(i, cmd))
            ser.flush()
            time.sleep(period_s)
    finally:
        ser.close()


# -------------------------------------------------------------- the link ----

class PerceptLink(threading.Thread):
    """
    Owns the serial port. send() writes one PERCEPT frame; this thread reads and
    decodes TELEM (and STATUS, when the firmware sends it) continuously and
    exposes the newest of each, plus any '#' log lines the firmware printed.

    Delivery is never guaranteed and never needs to be; all this class owes the
    STM32 is a steady cadence. NOTE: the firmware has no link-loss motor cut —
    if frames stop, the car carries on under its own logic. Halting it takes
    CMD_STOP (see the module docstring).
    """

    def __init__(self, port="/dev/ttyACM0", baud=115200, on_log=None):
        super().__init__(name="PerceptLink", daemon=True)
        self.port_name = port
        self.baud = baud
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._telem = Telemetry()
        self._status = None          # FsmStatus once the firmware sends one
        self._seq = 0
        self._wlock = threading.Lock()
        self._on_log = on_log or (lambda line: print("[stm32] " + line))
        self.rx_frames = 0
        self.rx_status = 0
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
            self._parse(buf, text)
            self._drain_text(text)

    def _parse(self, buf: bytearray, text: bytearray):
        """
        Pull every complete TELEM / STATUS frame out of buf (in place) and move
        the bytes between frames into text. Leaves a partial frame, or one
        trailing byte that could be half a sync word, in buf for next time.
        """
        while True:
            i_t = buf.find(TELEM_SYNC)
            i_s = buf.find(STATUS_SYNC)
            hits = [i for i in (i_t, i_s) if i >= 0]
            if not hits:
                # No frame start in sight. Everything but a possible split
                # sync byte is firmware log text.
                if len(buf) > 1:
                    text.extend(buf[:-1])
                    del buf[:-1]
                return
            i = min(hits)
            is_status = (i == i_s)
            if i > 0:
                text.extend(buf[:i])
                del buf[:i]
            n = STATUS_LEN if is_status else TELEM_LEN
            if len(buf) < n:
                return
            frame = bytes(buf[:n])
            payload, chk = frame[2:n - 1], frame[n - 1]
            if _xor8(payload) == chk:
                try:
                    if is_status:
                        st = parse_status(payload)
                        with self._lock:
                            self._status = st
                        self.rx_status += 1
                    else:
                        t = parse_telemetry(payload)
                        with self._lock:
                            self._telem = t
                        self.rx_frames += 1
                except struct.error:
                    self.rx_bad += 1
                del buf[:n]
            else:
                self.rx_bad += 1
                del buf[:2]          # false sync; hunt again

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

    def status(self):
        """Newest FsmStatus, or None if the firmware has not sent one."""
        with self._lock:
            return self._status

    # ---- tx ----

    def send(self, **kw) -> bytes:
        """Build and write one PERCEPT frame. kwargs go straight to pack_percept."""
        self._seq = (self._seq + 1) & 0xFF
        frame = pack_percept(self._seq, **kw)
        if self._ser is not None:
            try:
                with self._wlock:
                    self._ser.write(frame)
                self.tx_frames += 1
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        return frame
