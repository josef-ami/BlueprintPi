"""
Byte-layout tests for control/percept_link.py.

The Python side packs with struct; the STM32 side reads with memcpy at literal
offsets. Those two descriptions of the same frame can drift apart silently and
the only symptom on the mat is a car that steers at a plausible-looking wrong
angle. So this file re-implements BOTH C-side halves from the offsets written
in ObstacleLap.cpp and checks they agree with the codec.

If you change a frame, change it in three places and run this: the .cpp, the
codec, and here.
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
    assert xor8(f[2:15]) == f[15]
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
    assert len(frame) == pl.PERCEPT_LEN == 16
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
    # drive the same loop run() uses
    while True:
        i = buf.find(pl.TELEM_SYNC)
        if i < 0:
            if len(buf) > 1:
                text.extend(buf[:-1])
                del buf[:-1]
            break
        if i > 0:
            text.extend(buf[:i])
            del buf[:i]
        if len(buf) < pl.TELEM_LEN:
            break
        f = bytes(buf[:pl.TELEM_LEN])
        assert pl._xor8(f[2:21]) == f[21]
        link._telem = pl.parse_telemetry(f[2:21])
        del buf[:pl.TELEM_LEN]
    text.extend(buf)
    link._drain_text(text)

    assert lines == ["# HEADING", "# TURN 4/12"]
    assert link.telemetry().corner_count == 3
