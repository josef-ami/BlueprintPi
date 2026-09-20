"""
Byte-layout tests for control/percept_link.py.

The Python side packs with struct; the STM32 side reads with memcpy at literal
offsets. Those two descriptions of the same frame can drift apart silently and
the only symptom on the mat is a car that steers at a plausible-looking wrong
angle. So this file re-implements BOTH C-side halves from the offsets written
in ObstacleLap.cpp and checks they agree with the codec.

If you change a frame, change it in three places and run this: the .cpp, the
codec, and here.

NOTE on AVOID_COMMIT / leg_mm: both are still wire-legal values (the byte
layouts below are unchanged) but control/supervisor.py no longer sends
AVOID_COMMIT, and leg_mm is now only ever a backstop cap, never a distance
the STM32 counts down to — see the docstrings in supervisor.py and the AVOID
state in ObstacleLap.cpp. This file tests the byte layout, which did not
change, so it exercises those values exactly as before.
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import percept_link as pl


def xor8(b):
    x = 0
    for v in b:
        x ^= v
    return x


# --- transcribed from applyPercept() in ObstacleLap.cpp --------------------

def c_apply_percept(f):
    assert f[0] == 0xAA and f[1] == 0x55
    assert len(f) == 17
    assert xor8(f[2:16]) == f[16]
    flags = f[3]
    return dict(
        seq=f[2],
        lidar_ok=bool(flags & 0x01),
        cam_ok=bool(flags & 0x02),
        action=(flags & 0x0C) >> 2,
        green=bool(flags & 0x10),
        hello=bool(flags & 0x20),
        left=struct.unpack_from("<H", f, 4)[0],
        front=struct.unpack_from("<H", f, 6)[0],
        right=struct.unpack_from("<H", f, 8)[0],
        rev=f[10],
        heading=struct.unpack_from("<h", f, 11)[0] / 10.0,
        leg=struct.unpack_from("<H", f, 13)[0],
        cmd=f[15],
    )


# --- transcribed from sendTelemetry() in ObstacleLap.cpp -------------------

def c_send_telemetry(status=0, odo=0, spd=0, head_dd=0, yaw_dd=0, front=0xFFFF,
                     state=0, corners=0, rem=0, floor=0, seq=0):
    f = bytearray(22)
    f[0], f[1], f[2], f[3] = 0x55, 0xAA, seq, status
    struct.pack_into("<i", f, 4, odo)
    struct.pack_into("<h", f, 8, spd)
    struct.pack_into("<h", f, 10, head_dd)
    struct.pack_into("<h", f, 12, yaw_dd)
    struct.pack_into("<H", f, 14, front)
    f[16], f[17] = state, corners
    struct.pack_into("<H", f, 18, rem)
    f[20] = floor
    f[21] = xor8(f[2:21])
    return bytes(f)


# ------------------------------------------------------------------ tests --

def test_percept_length_and_sync():
    frame = pl.pack_percept(1, 100, 200, 300, 7, True, True, True,
                            pl.P_AVOID_SHIFT and 1, False, 0.0, 0)
    assert len(frame) == pl.PERCEPT_LEN == 17
    assert frame[:2] == pl.PERCEPT_SYNC == b"\xAA\x55"


def test_percept_round_trips_through_the_c_parser():
    frame = pl.pack_percept(seq=42, left_mm=412.0, front_mm=1873.0, right_mm=655.0,
                            rev=200, lidar_ok=True, cam_ok=True, hello=True,
                            action=2, green=True,
                            target_heading_deg=-12.3, leg_mm=587.4)
    got = c_apply_percept(frame)
    assert got["seq"] == 42
    assert (got["left"], got["front"], got["right"]) == (412, 1873, 655)
    assert got["rev"] == 200
    assert got["lidar_ok"] and got["cam_ok"] and got["hello"] and got["green"]
    assert got["action"] == 2
    assert got["heading"] == -12.3
    assert got["leg"] == 587
    assert got["cmd"] == pl.CMD_NONE          # default: no command


def test_negative_headings_survive_as_int16():
    for deg in (-179.9, -45.0, -0.1, 0.0, 0.1, 45.0, 179.9):
        f = pl.pack_percept(0, 0, 0, 0, 0, True, True, False, 0, False, deg, 0)
        assert c_apply_percept(f)["heading"] == round(deg, 1)


def test_missing_ranges_become_the_invalid_code():
    f = pl.pack_percept(0, None, float("inf"), 99999, 0, False, False, False,
                        0, False, 0.0, 0)
    got = c_apply_percept(f)
    assert got["left"] == got["front"] == got["right"] == 0xFFFF
    assert not got["lidar_ok"]


def test_telemetry_round_trips_from_the_c_packer():
    frame = c_send_telemetry(
        status=pl.S_RUNNING | pl.S_DIR_LOCKED | pl.S_IMU_OK,
        odo=-12345, spd=402, head_dd=-901, yaw_dd=150, front=734,
        state=pl.ST_AVOID, corners=7, rem=331, floor=2, seq=9)
    t = pl.parse_telemetry(frame[2:21])
    assert t.seq_ack == 9
    assert t.running and t.dir_locked and t.imu_ok
    assert not t.recovering and not t.lidar_stale
    assert t.odo_mm == -12345
    assert t.speed_mmps == 402
    assert t.heading_deg == -90.1
    assert t.yaw_rate_dps == 15.0
    assert t.front_mm == 734
    assert t.state == pl.ST_AVOID and t.state_name == "AVOID"
    assert t.corner_count == 7
    assert t.avoid_remaining_mm == 331
    assert t.floor_colour == 2


def test_invalid_front_decodes_to_inf_never_zero():
    t = pl.parse_telemetry(c_send_telemetry(front=0xFFFF)[2:21])
    assert t.front_mm == float("inf")


def test_telemetry_payload_is_exactly_nineteen_bytes():
    assert struct.calcsize("<BBihhhHBBHB") == 19
    assert 2 + 19 + 1 == pl.TELEM_LEN


def test_link_parser_separates_log_lines_from_frames():
    """The firmware's '#' prints share the port. 0xAA is not ASCII, so a log
    line can never contain the TELEM sync word."""
    lines = []
    link = pl.PerceptLink(on_log=lines.append)
    frame = c_send_telemetry(state=pl.ST_HEADING, corners=3)
    stream = b"# HEADING\n" + frame + b"# TURN 4/12\n" + frame

    buf = bytearray(stream)
    text = bytearray()
    link._parse(buf, text)          # the same routine run() calls
    text.extend(buf)
    link._drain_text(text)

    assert lines == ["# HEADING", "# TURN 4/12"]
    assert link.telemetry().corner_count == 3


# ============================================================================
# CMD byte, STOPPED and STATUS  (added with the dashboard's obstacle-run tab)
# ============================================================================

# --- transcribed from serviceLink() + applyPercept() + serviceCommands() ----

class CFirmware:
    """The byte hunter plus the command logic, one byte at a time."""
    def __init__(self, state=pl.ST_HEADING):
        self.buf = bytearray()
        self.p_cmd = pl.CMD_NONE
        self.p_cmd_run = 0
        self.state = state
        self.percepts = []
        self.fired = []                     # "STOP" / "REBOOT"

    def feed(self, data):
        for c in data:
            if len(self.buf) == 0:
                if c == 0xAA:
                    self.buf.append(c)
                continue
            if len(self.buf) == 1:
                if c == 0x55:
                    self.buf.append(c)
                else:
                    self.buf = bytearray([c]) if c == 0xAA else bytearray()
                continue
            self.buf.append(c)
            if len(self.buf) == 17:
                f = bytes(self.buf)
                self.buf = bytearray()
                if xor8(f[2:16]) == f[16]:
                    self._apply(f)

    def _apply(self, f):
        self.percepts.append(c_apply_percept(f))
        self.p_cmd_run = min(255, self.p_cmd_run + 1) if f[15] == self.p_cmd else 1
        self.p_cmd = f[15]
        self._service_commands()            # loop() runs it after every read

    def _service_commands(self):
        if self.p_cmd_run < 3:
            return
        if self.p_cmd == 3:
            self.fired.append("REBOOT")
        if self.p_cmd == 2 and self.state not in (pl.ST_STOPPED, pl.ST_FINISH):
            self.fired.append("STOP")
            self.state = pl.ST_STOPPED


def cmd_frames(cmd, n, seq0=0):
    return b"".join(pl.pack_command_frame(seq0 + i, cmd) for i in range(n))


def test_cmd_byte_round_trips_through_the_c_parser():
    for cmd in (pl.CMD_NONE, pl.CMD_RERUN, pl.CMD_STOP, pl.CMD_REBOOT):
        f = pl.pack_percept(5, 400, 1500, 600, 9, True, True, True, 1, False,
                            -3.5, 250, cmd=cmd)
        assert len(f) == 17
        got = c_apply_percept(f)
        assert got["cmd"] == cmd and got["leg"] == 250 and got["action"] == 1


def test_command_only_frame_never_refreshes_the_lidar():
    got = c_apply_percept(pl.pack_command_frame(1, pl.CMD_STOP))
    assert got["cmd"] == pl.CMD_STOP
    assert not got["lidar_ok"] and not got["cam_ok"] and not got["hello"]
    assert got["left"] == got["front"] == got["right"] == 0xFFFF
    assert got["action"] == 0


def test_stop_needs_three_frames_in_a_row():
    fw = CFirmware()
    fw.feed(cmd_frames(pl.CMD_STOP, 2))
    assert fw.fired == [] and fw.state == pl.ST_HEADING
    fw.feed(cmd_frames(pl.CMD_STOP, 1, seq0=2))
    assert fw.fired == ["STOP"] and fw.state == pl.ST_STOPPED
    fw.feed(cmd_frames(pl.CMD_STOP, 10, seq0=3))       # holding it: fires once
    assert fw.fired == ["STOP"]


def test_interleaved_commands_never_confirm():
    fw = CFirmware()
    for i in range(30):
        fw.feed(pl.pack_command_frame(i, pl.CMD_REBOOT if i % 2 else pl.CMD_STOP))
    assert fw.fired == []


def test_stop_is_ignored_once_finished():
    fw = CFirmware(state=pl.ST_FINISH)
    fw.feed(cmd_frames(pl.CMD_STOP, 5))
    assert fw.fired == [] and fw.state == pl.ST_FINISH


def test_reboot_burst_from_send_command_burst_default_confirms():
    fw = CFirmware()
    fw.feed(cmd_frames(pl.CMD_REBOOT, 2 * pl.CMD_CONFIRM_FRAMES))
    assert "REBOOT" in fw.fired


def test_a_single_corrupt_frame_cannot_fire_anything():
    """Fuzz: random single-byte damage to a stream of NONE frames never
    produces three identical good STOP/REBOOT frames."""
    import random
    rnd = random.Random(11)
    fw = CFirmware()
    for seq in range(5000):
        f = bytearray(pl.pack_percept(seq, rnd.uniform(0, 3000), rnd.uniform(0, 3000),
                                      rnd.uniform(0, 3000), rnd.randrange(256), True,
                                      True, True, rnd.randrange(3), rnd.random() < .5,
                                      rnd.uniform(-180, 180), rnd.uniform(0, 1500)))
        if rnd.random() < 0.2:
            f[rnd.randrange(2, 17)] ^= 1 << rnd.randrange(8)
        fw.feed(bytes(f))
    assert fw.fired == []


# --- transcribed from finishStep() / stoppedStep(): the RERUN edge rule ----

class CRerun:
    def __init__(self):
        self.armed = False          # cleared on entry to FINISH / STOPPED
        self.reruns = 0

    def step(self, p_cmd, link_stale=False):
        if p_cmd != pl.CMD_RERUN:
            self.armed = True
            return
        if not self.armed or link_stale:
            return
        self.reruns += 1
        self.armed = False          # goState(BOOT); re-entering FINISH clears it


def test_rerun_held_from_before_the_finish_does_nothing():
    r = CRerun()
    for _ in range(50):
        r.step(pl.CMD_RERUN)
    assert r.reruns == 0


def test_rerun_takes_a_rising_edge_and_gives_exactly_one():
    r = CRerun()
    r.step(pl.CMD_NONE)
    for _ in range(50):
        r.step(pl.CMD_RERUN)
    assert r.reruns == 1


def test_a_held_stop_arms_the_rerun_in_stopped():
    r = CRerun()
    r.step(pl.CMD_STOP)             # the Pi was still holding STOP
    r.step(pl.CMD_RERUN)
    assert r.reruns == 1


# --- transcribed from sendStatus() in ObstacleLap.cpp ----------------------

def c_send_status(version=1, state=0, flags=0, flags2=0, pflags=0, pcmd=0,
                  pcmd_run=0, run_no=0, state_ms=0, countdown=0, grace=0,
                  avoid_ms=0, lane_dd=0, tgt_dd=0, servo_dd=765, pwm=0,
                  revs_l=0, revs_r=0, rec_tries=0, rec_ret=1, fin=0,
                  lid_l=0xFFFF, lid_r=0xFFFF, straight=0, corner_lock=0,
                  avoid_lock=0, segment=0, fin_tgt=1000, front_start=0xFFFF,
                  phase=0, leg=0, link_age=0xFFFF, lidar_age=0xFFFF, frames=0):
    f = bytearray(61)
    f[0], f[1] = 0x55, 0xA5
    (f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]) = (
        version, state, flags, flags2, pflags, pcmd, pcmd_run, run_no)
    struct.pack_into("<H", f, 10, state_ms)
    struct.pack_into("<H", f, 12, countdown)
    struct.pack_into("<H", f, 14, grace)
    struct.pack_into("<H", f, 16, avoid_ms)
    struct.pack_into("<h", f, 18, lane_dd)
    struct.pack_into("<h", f, 20, tgt_dd)
    struct.pack_into("<h", f, 22, servo_dd)
    struct.pack_into("<h", f, 24, pwm)
    f[26], f[27], f[28], f[29], f[30], f[31] = revs_l, revs_r, rec_tries, rec_ret, fin, 0
    for off, v in ((32, lid_l), (34, lid_r), (36, straight), (38, corner_lock),
                   (40, avoid_lock), (42, segment), (44, fin_tgt),
                   (46, front_start), (48, phase), (50, leg),
                   (52, link_age), (54, lidar_age)):
        struct.pack_into("<H", f, off, v)
    struct.pack_into("<I", f, 56, frames)
    f[60] = xor8(f[2:60])
    return bytes(f)


def test_status_payload_is_exactly_fifty_eight_bytes():
    assert struct.calcsize(pl.STATUS_FMT) == 58
    assert 2 + 58 + 1 == pl.STATUS_LEN == 61


def test_status_round_trips_from_the_c_packer():
    frame = c_send_status(
        state=pl.ST_AVOID,
        flags=pl.SF_FSM_STARTED | pl.SF_WALL_SEEN_L | pl.SF_WALL_SEEN_R | pl.SF_BLIND,
        flags2=pl.SF2_REAL_OPEN_R,
        pflags=pl.PF_HELLO | pl.PF_LIDAR_OK | pl.PF_GREEN | (2 << pl.PF_ACTION_SHIFT),
        pcmd=pl.CMD_STOP, pcmd_run=2, run_no=3,
        state_ms=1234, avoid_ms=873, lane_dd=-900, tgt_dd=-1123,
        servo_dd=1105, pwm=-90, revs_l=0, revs_r=2, rec_tries=1,
        rec_ret=pl.ST_AVOID, fin=pl.FINISH_BY_WALL, lid_l=455, lid_r=1880,
        straight=2345, corner_lock=120, avoid_lock=77, segment=2601,
        fin_tgt=1400, front_start=1650, phase=301, leg=588, link_age=18,
        lidar_age=22, frames=70000)
    s = pl.parse_status(frame[2:60])
    assert s.version == 1 and s.state == pl.ST_AVOID and s.state_name == "AVOID"
    assert s.fsm_started and s.wall_seen_l and s.wall_seen_r and s.blind
    assert not (s.boot_ready or s.lidar_hold or s.link_stale or s.rerun_armed)
    assert s.real_open_r and not s.real_open_l and not s.lock_needs_real
    assert s.p_hello and s.p_lidar_ok and s.p_green and not s.p_cam_ok
    assert s.p_action == 2 and s.p_cmd == pl.CMD_STOP and s.p_cmd_run == 2
    assert s.run_number == 3
    assert (s.state_ms, s.avoid_ms) == (1234, 873)
    assert s.lane_heading_deg == -90.0 and s.target_heading_deg == -112.3
    assert s.servo_deg == 110.5 and s.motor_pwm == -90
    assert (s.open_revs_l, s.open_revs_r, s.recover_tries) == (0, 2, 1)
    assert s.recover_return_state == pl.ST_AVOID
    assert s.finish_reason == pl.FINISH_BY_WALL
    assert (s.lidar_l_mm, s.lidar_r_mm) == (455, 1880)
    assert s.straight_mm == 2345
    assert (s.corner_lockout_mm, s.avoid_lockout_mm) == (120, 77)
    assert (s.segment_mm, s.finish_target_mm) == (2601, 1400)
    assert s.front_at_start_mm == 1650
    assert (s.phase_mm, s.avoid_leg_mm) == (301, 588)
    assert (s.link_age_ms, s.lidar_age_ms) == (18, 22)
    assert s.percept_frames == 70000


def test_status_invalid_codes_decode_to_inf_and_none():
    s = pl.parse_status(c_send_status()[2:60])
    assert s.lidar_l_mm == s.lidar_r_mm == float("inf")
    assert s.front_at_start_mm == float("inf")
    assert s.link_age_ms is None and s.lidar_age_ms is None


def test_state_names_cover_the_cpp_enum_including_stopped():
    assert pl.STATE_NAMES == ["BOOT", "HEADING", "AVOID", "TURN90",
                              "FINISH", "RECOVER", "STOPPED"]
    assert pl.ST_STOPPED == 6


def test_parser_separates_log_telem_and_status_even_when_split():
    lines = []
    link = pl.PerceptLink(on_log=lines.append)
    stream = (b"# BOOT waiting for Pi\n"
              + c_send_telemetry(state=pl.ST_BOOT, corners=0)
              + c_send_status(state=pl.ST_BOOT, countdown=4200,
                              flags=pl.SF_FSM_STARTED | pl.SF_BOOT_READY)
              + b"# Pi ready - 5s countdown\n"
              + c_send_telemetry(state=pl.ST_HEADING, corners=1)
              + b"# GO  frontAtStart=1650\n")
    buf, text = bytearray(), bytearray()
    for i in range(0, len(stream), 7):          # arrive in awkward chunks
        buf.extend(stream[i:i + 7])
        link._parse(buf, text)
        link._drain_text(text)
    text.extend(buf)
    link._drain_text(text)
    assert lines == ["# BOOT waiting for Pi", "# Pi ready - 5s countdown",
                     "# GO  frontAtStart=1650"]
    assert link.telemetry().state == pl.ST_HEADING
    assert link.telemetry().corner_count == 1
    st = link.status()
    assert st is not None and st.boot_ready and st.countdown_ms == 4200
    assert link.rx_frames == 2 and link.rx_status == 1 and link.rx_bad == 0


def test_corrupt_status_is_counted_and_skipped():
    link = pl.PerceptLink(on_log=lambda _l: None)
    bad = bytearray(c_send_status(state=pl.ST_TURN90))
    bad[20] ^= 0x01
    buf = bytearray(bytes(bad) + c_send_telemetry(state=pl.ST_TURN90))
    link._parse(buf, bytearray())
    assert link.status() is None
    assert link.rx_bad == 1
    assert link.telemetry().state == pl.ST_TURN90


def test_no_status_from_older_firmware_leaves_status_none():
    link = pl.PerceptLink(on_log=lambda _l: None)
    link._parse(bytearray(c_send_telemetry(state=pl.ST_HEADING)), bytearray())
    assert link.status() is None
