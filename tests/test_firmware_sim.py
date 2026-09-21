"""
The REAL firmware, compiled for the host and driven over its real wire.

firmware/sim/build.sh compiles firmware/ObstacleRound.cpp unmodified against
stub Arduino headers. These tests load the result with ctypes, feed it DRIVE
frames built by control.link.pack_drive - the same function the run loop
uses - and decode what it sends back with control.link's own TELEM decoder
and params.StmParams' own line parser. So both halves of the protocol are the
production code, and a disagreement between them fails here rather than on
the mat.

What this cannot tell you: anything about the hardware. The IMU, the encoder,
the colour sensor, the servo and the motor driver are stubs that do exactly
what they are told. It proves the logic and the wire, not the car.

Skipped when there is no g++ (e.g. a Pi without build-essential).
"""

import ctypes
import os
import shutil
import subprocess
import tempfile

import pytest

import params as prm
from control.intent import ActionIntent, SteerMode
from control.link import (CMD_NONE, CMD_REBOOT, DRIVE_SYNC, S_ARC_DONE,
                          S_ENABLED, S_IMU_OK, S_LINK_STALE, S_RECOVERING,
                          S_RECOVER_CAPPED, TELEM_LEN, TELEM_SYNC, pack_drive,
                          unpack_telem, xor8)

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(HERE), "firmware", "sim")

# The firmware's compiled-in defaults, restated so a test says what it means.
STRAIGHT = 76.5
MAX_LEFT, MAX_RIGHT = 20.0, 140.0
STEER_LOCK = 35.0
PANIC_MM, CLEAR_MM = 200, 350
LINK_STALE_MS = 200
TICKS_PER_CM = 14.853

_BUILT = None
_N = 0


def _build():
    """Compile once per process, into a temp dir, so a stale .so in the tree
    can never be what gets tested."""
    global _BUILT
    if _BUILT is None:
        out = os.path.join(tempfile.mkdtemp(prefix="fwsim-"), "libexec.so")
        r = subprocess.run(["sh", os.path.join(SIM, "build.sh"), out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("firmware did not compile:\n" + r.stderr)
        assert "warning" not in r.stderr, \
            "firmware compiles with warnings:\n" + r.stderr
        _BUILT = out
    return _BUILT


def _lib():
    """A FRESH copy of the firmware for every board.

    dlopen() refcounts by path, so loading the same .so twice hands back the
    same globals - a parameter one test set would still be set in the next,
    and a board would inherit the last one's timers. A real power cycle
    re-initialises RAM, so each board gets its own copy of the library, which
    re-runs every static initialiser exactly as a reset would.
    """
    global _N
    src = _build()
    _N += 1
    dst = os.path.join(os.path.dirname(src), f"libexec-{_N}.so")
    shutil.copyfile(src, dst)
    lib = ctypes.CDLL(dst)
    lib.fw_step.argtypes = [ctypes.c_ulong, ctypes.c_float, ctypes.c_long,
                            ctypes.c_int, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_int]
    lib.fw_tx.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.fw_tx.restype = ctypes.POINTER(ctypes.c_ubyte)
    lib.fw_servo.restype = ctypes.c_float
    return lib


needs_gcc = pytest.mark.skipif(shutil.which("g++") is None,
                               reason="no g++ to build the firmware with")


class Board:
    """One freshly powered board: its own copy of the firmware, then setup()."""

    def __init__(self):
        self.lib = _lib()
        self.t = 0
        self.heading = 0.0
        self.floor = 0
        self.seq = 0
        self.buf = b""
        self.lines = []
        self.telem = []
        self.lib.fw_setup()
        self.lib.fw_set_tx_room(4096)
        self._drain()

    # ---- the port ----
    def _drain(self):
        n = ctypes.c_int(0)
        p = self.lib.fw_tx(ctypes.byref(n))
        self.buf += bytes(p[:n.value])
        # Split the byte stream the way control.link's reader does: a 55 AA
        # starts a TELEM frame, anything else accumulates into text lines.
        while self.buf:
            i = self.buf.find(TELEM_SYNC)
            text = self.buf if i < 0 else self.buf[:i]
            if i < 0:
                *done, self.buf = self.buf.split(b"\n")
                self.lines += [d.decode() for d in done if d]
                return
            if text:
                *done, rest = text.split(b"\n")
                self.lines += [d.decode() for d in done if d]
                assert not rest, f"a TELEM frame split a text line: {rest!r}"
            if len(self.buf) - i < TELEM_LEN:
                self.buf = self.buf[i:]
                return
            frame = self.buf[i:i + TELEM_LEN]
            payload = frame[2:-1]
            assert xor8(payload) == frame[-1], "bad TELEM checksum"
            self.telem.append(unpack_telem(payload, stamp=self.t))
            self.buf = self.buf[i + TELEM_LEN:]

    def step(self, n=1, dt=10, rx=b"", advance_ticks=0, imu=True):
        for k in range(n):
            self.t += dt
            data = rx if k == 0 else b""
            self.lib.fw_step(self.t, self.heading, advance_ticks, self.floor,
                             1 if imu else 0, data, len(data))
        self._drain()

    def send(self, intent, front=2000, left=500, right=500, lidar_ok=True,
             pillar=False, cmd=CMD_NONE, n=2, **kw):
        """One DRIVE frame, then n loop() passes of 10 ms. The default of two
        is one 50 Hz frame period - the Pi's cadence - which is also what it
        takes for a TELEM frame reflecting this DRIVE frame to go out."""
        frame = pack_drive(self.seq, intent, left, front, right, 0,
                           lidar_ok, True, pillar, cmd)
        self.seq += 1
        self.step(n=n, rx=frame, **kw)

    def drive(self, intent, ms, **kw):
        """Keep sending the same intent at 50 Hz for `ms`."""
        for _ in range(max(1, ms // 20)):
            self.send(intent, n=2, **kw)

    def line(self, text):
        self.step(rx=(text + "\n").encode())

    # ---- what we can see ----
    @property
    def motor(self):
        return self.lib.fw_motor()

    @property
    def servo(self):
        return self.lib.fw_servo()

    @property
    def last(self):
        return self.telem[-1]


def hold(heading=0.0, speed=60):
    return ActionIntent(mode=SteerMode.HEADING_HOLD, target_heading_deg=heading,
                        speed_pwm=speed)


# ------------------------------------------------------------------- boot

@needs_gcc
def test_it_boots_and_announces_itself():
    b = Board()
    v = [l for l in b.lines if l.startswith("!V ")]
    assert v, f"no !V at boot: {b.lines}"
    _, ver, count, boot = v[0].split()
    assert int(count) == 20
    assert any("executor ready" in l for l in b.lines)
    assert not any("ERROR" in l for l in b.lines), b.lines


@needs_gcc
def test_it_sits_still_until_the_pi_speaks():
    """No frame has ever arrived: motor off, servo straight."""
    b = Board()
    b.step(n=50)
    assert b.motor == 0
    assert b.servo == pytest.approx(STRAIGHT, abs=0.2)


@needs_gcc
def test_telem_flows_at_50hz_and_decodes():
    b = Board()
    b.step(n=100, dt=10)                      # 1 s
    assert 45 <= len(b.telem) <= 55
    assert b.last.imu_ok
    seqs = [t.seq for t in b.telem]
    assert all((b2 - a) % 256 == 1 for a, b2 in zip(seqs, seqs[1:]))


@needs_gcc
def test_boot_id_is_in_every_frame_and_matches_the_announcement():
    b = Board()
    b.step(n=20)
    boot = int([l for l in b.lines if l.startswith("!V ")][0].split()[3])
    assert {t.boot_id for t in b.telem} == {boot}


# ------------------------------------------------------------ the heading loop

@needs_gcc
def test_heading_hold_drives_forward():
    b = Board()
    b.drive(hold(0.0, 60), 200)
    assert b.motor == 60
    assert b.last.enabled


@needs_gcc
def test_a_heading_to_the_left_steers_left():
    """+ heading = left, the Pi's convention, and left is BELOW straight on
    this servo. A sign error here would turn every correction into a crash."""
    b = Board()
    b.drive(hold(20.0), 300)
    assert b.servo < STRAIGHT - 5
    b2 = Board()
    b2.drive(hold(-20.0), 300)
    assert b2.servo > STRAIGHT + 5


@needs_gcc
def test_the_servo_is_slew_limited():
    """2.5 deg per IMU event, however big the error - that is what stops the
    tyres scrubbing. It is the reason this loop stayed on the STM32."""
    # The servo is written in whole microseconds, ~0.09 deg a step, hence
    # the tolerance.
    b = Board()
    b.send(hold(90.0), n=1)
    first = abs(b.servo - STRAIGHT)
    assert 2.4 <= first <= 2.6
    b.send(hold(90.0), n=1)
    assert 4.9 <= abs(b.servo - STRAIGHT) <= 5.1


@needs_gcc
def test_the_heading_is_reported_back():
    b = Board()
    b.heading = 12.5
    b.drive(hold(0.0), 100)
    assert b.last.heading_deg == pytest.approx(12.5, abs=0.2)


@needs_gcc
def test_odometry_is_reported_and_never_zeroed():
    b = Board()
    b.drive(hold(0.0), 200, advance_ticks=15)
    a = b.last.odo_ticks
    assert a > 0
    b.drive(hold(0.0), 200, advance_ticks=15)
    assert b.last.odo_ticks > a


@needs_gcc
def test_the_floor_colour_is_reported():
    b = Board()
    b.floor = 1
    b.step(n=5)
    assert b.last.floor == 1               # orange
    b.floor = 2
    b.step(n=5)
    assert b.last.floor == 2               # blue
    b.floor = 0
    b.step(n=5)
    assert b.last.floor == 0


# ------------------------------------------------------------- DIRECT / ARC

@needs_gcc
def test_direct_mode_maps_road_wheel_degrees_onto_the_servo():
    """The Pi sends physical road-wheel degrees; only the firmware knows the
    trim and the stops. Full lock either way must land on the stop."""
    b = Board()
    d = lambda s: ActionIntent(mode=SteerMode.DIRECT, steer_deg=s, speed_pwm=50)
    b.send(d(STEER_LOCK))
    assert b.servo == pytest.approx(MAX_LEFT, abs=0.2)
    b.send(d(-STEER_LOCK))
    assert b.servo == pytest.approx(MAX_RIGHT, abs=0.2)
    b.send(d(0.0))
    assert b.servo == pytest.approx(STRAIGHT, abs=0.2)
    b.send(d(STEER_LOCK / 2))
    assert b.servo == pytest.approx(STRAIGHT - (STRAIGHT - MAX_LEFT) / 2, abs=0.3)


@needs_gcc
def test_direct_mode_reverses():
    b = Board()
    b.send(ActionIntent(mode=SteerMode.DIRECT, steer_deg=0, speed_pwm=40,
                        reverse=True))
    assert b.motor == -40


@needs_gcc
def test_the_arc_turns_toward_the_target_and_reports_done():
    b = Board()
    arc = ActionIntent(mode=SteerMode.ARC, target_heading_deg=-90.0,
                       speed_pwm=50, arc_lock=1.0)
    b.send(arc)
    assert b.servo > STRAIGHT + 10         # right = above straight
    assert not b.last.arc_done
    b.heading = -80.0                       # within TURN_STOP_DEG (15)
    b.send(arc, n=3)
    assert b.last.arc_done


@needs_gcc
def test_the_arc_lock_fraction_limits_the_steer():
    """arc_lock is how the Pi's corner-exit shaping reaches the arc: the
    three shapes differ in it. 0.5 must steer about half as hard as 1.0."""
    full, half = Board(), Board()
    mk = lambda f: ActionIntent(mode=SteerMode.ARC, target_heading_deg=-90.0,
                                speed_pwm=50, arc_lock=f)
    full.send(mk(1.0))
    half.send(mk(0.5))
    assert abs(half.servo - STRAIGHT) == pytest.approx(
        0.5 * abs(full.servo - STRAIGHT), rel=0.05)


# ---------------------------------------------------------- stopping it

@needs_gcc
def test_a_stop_stops():
    b = Board()
    b.drive(hold(), 100)
    assert b.motor == 60
    b.send(ActionIntent.stop())
    assert b.motor == 0
    assert b.servo == pytest.approx(STRAIGHT, abs=0.2)
    assert not b.last.enabled


@needs_gcc
def test_silence_stops_the_car():
    """New since the FSM moved to the Pi: the firmware has no state machine
    to carry on with, so a link that goes quiet must mean stop."""
    b = Board()
    b.drive(hold(), 200)
    assert b.motor == 60
    b.step(n=(LINK_STALE_MS // 10) + 5)    # no frames for > LINK_STALE_MS
    assert b.motor == 0
    assert b.last.link_stale
    # and it picks straight back up when the Pi does
    b.drive(hold(), 60)
    assert b.motor == 60
    assert not b.last.link_stale


@needs_gcc
def test_a_corrupt_frame_is_ignored():
    b = Board()
    b.drive(hold(0.0, 60), 100)
    frame = bytearray(pack_drive(99, hold(0.0, 200), 500, 2000, 500, 0,
                                 True, True, False))
    frame[-1] ^= 0xFF                       # break the checksum
    b.step(rx=bytes(frame))
    assert b.motor == 60, "a frame with a bad checksum was acted on"


@needs_gcc
def test_ascii_between_frames_does_not_desync_the_parser():
    """One port carries frames and parameter lines. 0xAA is not ASCII, so a
    line can never contain a sync word - which is the whole design."""
    b = Board()
    f = pack_drive(0, hold(0.0, 70), 500, 2000, 500, 0, True, True, False)
    b.step(rx=b"?V\n" + f + b"C\n" + f)
    assert b.motor == 70
    assert "!C" in b.lines


# ------------------------------------------------------------ the reflex

@needs_gcc
def test_a_wall_close_ahead_triggers_the_reverse():
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50)
    assert b.motor < 0, "did not back off the wall"
    assert b.last.recovering


@needs_gcc
def test_the_reflex_stands_down_for_a_pillar():
    """A short front reading with the camera on a pillar IS that pillar, and
    the planner is already steering round it."""
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50, pillar=True)
    assert b.motor > 0
    assert not b.last.recovering


@needs_gcc
def test_the_reflex_needs_a_trusted_lidar():
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50, lidar_ok=False)
    assert b.motor > 0


@needs_gcc
def test_recovery_ends_when_the_wall_is_clear():
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50)
    assert b.last.recovering
    b.send(hold(), front=CLEAR_MM + 50, n=3)
    assert not b.last.recovering
    assert not b.last.recover_capped


@needs_gcc
def test_recovery_is_capped_by_distance():
    """Backing off the wall that never clears - front between the panic and
    the clear distance - stops at RECOVER_MAX_CM (30) and says so."""
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50)
    assert b.last.recovering
    for _ in range(40):
        b.send(hold(), front=PANIC_MM + 50, advance_ticks=-15)
        if not b.last.recovering:
            break
    assert not b.last.recovering
    assert b.last.recover_capped
    assert b.last.recover_tries == 1
    backed_cm = -b.last.odo_ticks / TICKS_PER_CM
    assert 30 <= backed_cm < 33


@needs_gcc
def test_a_stop_during_recovery_ends_it():
    """The deadlock this used to have: a Stop mid-recovery handed a speed of
    zero to the recovery step, which then 'reversed' at a standstill forever,
    never reached the clear distance or the cap, and ignored every later
    command. A Stop is now checked first."""
    b = Board()
    b.drive(hold(), 100)
    b.send(hold(), front=PANIC_MM - 50)
    assert b.last.recovering
    b.send(ActionIntent.stop(), front=PANIC_MM - 50, n=3)
    assert b.motor == 0
    assert not b.last.recovering
    assert any("recovery abandoned" in l for l in b.lines)
    # and the car is still commandable afterwards
    b.drive(hold(), 60, front=2000)
    assert b.motor == 60


# ------------------------------------------------------------ reboot

@needs_gcc
def test_one_reboot_frame_is_not_enough():
    """A single corrupt or mis-synced frame must never reset the car."""
    b = Board()
    b.send(ActionIntent.stop(), cmd=CMD_REBOOT)
    b.send(ActionIntent.stop(), cmd=CMD_NONE)
    b.send(ActionIntent.stop(), cmd=CMD_REBOOT)
    assert b.lib.fw_resets() == 0


@needs_gcc
def test_a_run_of_reboot_frames_resets():
    b = Board()
    for _ in range(4):
        b.send(ActionIntent.stop(), cmd=CMD_REBOOT)
    assert b.lib.fw_resets() == 1
    assert b.motor == 0


# --------------------------------------------------- the parameter protocol

@needs_gcc
def test_the_real_dump_parses_into_the_real_mirror():
    """The two halves of the parameter protocol, both production code."""
    b = Board()
    stm = prm.StmParams()
    b.line("?P")
    b.step(n=30)
    for l in b.lines:
        if l.startswith("!"):
            stm.on_line(l)
    assert len(stm.table) == 20
    assert stm.synced
    assert stm.live["HEAD_KP"] == pytest.approx(2.0)
    assert stm.live["STEER_LOCK_DEG"] == pytest.approx(STEER_LOCK)
    assert stm.ticks_per_mm() == pytest.approx(TICKS_PER_CM / 10.0, rel=1e-3)
    assert set(stm.table) == {
        "TICKS_PER_CM", "SERVO_TRUE_STRAIGHT", "SERVO_MAX_LEFT",
        "SERVO_MAX_RIGHT", "STEER_LOCK_DEG", "IMU_YAW_SIGN", "HEAD_KP",
        "HEAD_KI", "HEAD_KD", "YAW_FILT_ALPHA", "SERVO_SLEW", "INTEGRAL_CLAMP",
        "TURN_KP", "TURN_MIN_STEER", "TURN_STOP_DEG", "WALL_PANIC_MM",
        "WALL_CLEAR_MM", "RECOVER_MAX_CM", "RECOVER_MAX_TRIES",
        "LINK_STALE_MS"}
    groups = {v["group"] for v in stm.table.values()}
    assert groups == set(range(len(prm.STM_GROUPS)))


@needs_gcc
def test_the_dump_waits_for_room_in_the_usb_buffer():
    """Streamed, and only while the CDC buffer has room - dumping 20 lines in
    one go would stall the control loop for as long as the host took."""
    b = Board()
    b.lib.fw_set_tx_room(10)
    b.line("?P")
    b.step(n=10)
    assert not any(l.startswith("!P ") for l in b.lines)
    b.lib.fw_set_tx_room(4096)
    b.step(n=20)
    assert sum(l.startswith("!P ") for l in b.lines) == 20


@needs_gcc
def test_setting_a_parameter_by_name_changes_behaviour():
    b = Board()
    b.line("N STEER_LOCK_DEG 17.5")
    assert any(l.startswith("!p ") and l.endswith("17.5000") for l in b.lines)
    b.send(ActionIntent(mode=SteerMode.DIRECT, steer_deg=17.5, speed_pwm=50))
    assert b.servo == pytest.approx(MAX_LEFT, abs=0.2), \
        "full lock is 17.5 deg now, so 17.5 must reach the stop"


@needs_gcc
def test_the_pushed_flag_shows_up_in_telem():
    b = Board()
    b.step(n=3)
    assert not (b.last.status & 0x80)
    b.line("N HEAD_KP 2.5")
    b.step(n=3)
    assert b.last.status & 0x80


@needs_gcc
def test_bad_parameter_lines_are_refused_cleanly():
    b = Board()
    b.line("N HEAD_KP 999")
    b.line("N NOT_A_THING 1")
    b.line("P99 1")
    b.line("N HEAD_KP")
    assert "!E range" in b.lines
    assert "!E name" in b.lines
    assert "!E id" in b.lines
    assert "!E syntax" in b.lines


@needs_gcc
def test_the_link_timeout_is_itself_tunable():
    b = Board()
    b.line("N LINK_STALE_MS 1000")
    b.drive(hold(), 100)
    b.step(n=50)                           # 500 ms of silence
    assert b.motor == 60, "stopped at the old 200 ms timeout"
    b.step(n=60)                           # past 1000 ms
    assert b.motor == 0


# ------------------------------------------------- the two halves together
#
# The Pi's FSM, planner and mapper deciding; pack_drive sending; the REAL
# firmware executing; its servo and motor moving a kinematic car around the
# lap-sim world; its TELEM - heading, odometry, floor colour, ARC_DONE -
# coming back as the FSM's only view of the car. Every line of production
# code on both sides of the wire is in this loop. The car is still a toy
# (bicycle model, perfect sensors), so this proves the sequencing and the
# handshakes, not the handling.

def _road_wheel_deg(servo):
    """Invert the firmware's own servoForSteer(): servo angle -> road-wheel
    degrees, + = left."""
    if servo <= STRAIGHT:
        return (STRAIGHT - servo) / (STRAIGHT - MAX_LEFT) * STEER_LOCK
    return -(servo - STRAIGHT) / (MAX_RIGHT - STRAIGHT) * STEER_LOCK


def run_closed_loop(p, max_s=240.0, colour=1):
    import math
    from control.fsm import COLOUR_NONE, Ctx, FSM, State
    from control.mapper import CommandMapper, MEASURED_RADIUS_MM
    from test_lap_sim import World
    from worldstate import wrap180

    b = Board()
    fsm = FSM(p, emit=lambda s: None)
    mapper = CommandMapper()
    w = World(p)
    tpm = TICKS_PER_CM / 10.0
    odo_frac = 0.0

    def ctx():
        left, right = w.sides(fsm)
        cl, cr = w.cones()
        tel = b.last
        return Ctx(p=p, now=b.t / 1000.0, telem=tel, telem_fresh=True,
                   lidar_live=True, lidar_dead=False, new_frame=True,
                   new_rev=True, front_mm=w.front(), left_mm=left,
                   right_mm=right, cone_left=cl if cl < 1100 else None,
                   cone_right=cr if cr < 1100 else None,
                   wall_ang=wrap180(w.heading - w.lane), ticks_per_mm=tpm)

    b.step(n=4)                               # a TELEM frame to start from
    fsm.step(ctx())                           # arm
    fsm.request_start()
    frames = int(max_s * 50)
    for _ in range(frames):
        # the floor under the car: the first corner's line
        b.floor = colour if (w.corner_here() and fsm.corner_count == 0
                             and not fsm.dc_colour_armed) else 0
        before = fsm.corner_count
        intent = mapper.map(fsm.step(ctx()))
        if fsm.corner_count != before:
            w.turn(fsm)
        left, right = w.sides(fsm)
        frame = pack_drive(b.seq, intent, left, w.front(), right, 0,
                           True, True, False)
        b.seq += 1
        # two 10 ms firmware passes per 20 ms frame, the car moving under
        # whatever the firmware is actually commanding
        for k in range(2):
            v = 400.0 * b.motor / 60.0                       # mm/s
            d = v * 0.01
            delta = math.radians(_road_wheel_deg(b.servo))
            w.heading = wrap180(w.heading + math.degrees(
                d * math.tan(delta) / math.tan(math.radians(STEER_LOCK))
                / MEASURED_RADIUS_MM))
            yaw = math.radians(wrap180(w.heading - w.lane))
            w.along += d * math.cos(yaw)
            w.lat = max(-480.0, min(480.0, w.lat + d * math.sin(yaw)))
            odo_frac += d * tpm
            dt, odo_frac = int(odo_frac), odo_frac - int(odo_frac)
            b.heading = w.heading
            b.t += 10
            data = frame if k == 0 else b""
            b.lib.fw_step(b.t, b.heading, dt, b.floor, 1, data, len(data))
        b._drain()
        if fsm.state is State.FINISHED and fsm.corner_count > 0:
            break
    return fsm, w, b


@needs_gcc
def test_the_pi_and_the_real_firmware_complete_a_run(p):
    from control.fsm import State
    fsm, w, b = run_closed_loop(p)
    assert fsm.state is State.FINISHED, \
        f"stuck in {fsm.state.name} at corner {fsm.corner_count}"
    assert fsm.corner_count == p["TARGET_CORNERS"]
    assert fsm.clockwise
    assert b.motor == 0, "finished, but the firmware is still driving"
    # Stale is correct until the Pi's first DRIVE frame; after that, never.
    first = next(i for i, t in enumerate(b.telem) if not t.link_stale)
    assert first <= 3
    assert not any(t.link_stale for t in b.telem[first:]), \
        "the link dropped mid-run"
    assert not any(t.recovering for t in b.telem), "hit a wall on a bare lap"


@needs_gcc
def test_the_pi_and_the_real_firmware_run_anticlockwise_too(p):
    from control.fsm import State
    fsm, w, b = run_closed_loop(p, colour=2)
    assert fsm.state is State.FINISHED, \
        f"stuck in {fsm.state.name} at corner {fsm.corner_count}"
    assert not fsm.clockwise
    assert fsm.corner_count == p["TARGET_CORNERS"]
