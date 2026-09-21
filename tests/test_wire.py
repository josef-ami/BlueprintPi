"""
The DRIVE and TELEM frames, against docs/PI_STM32_PROTOCOL.md.

These matter more than they look: nothing here can be checked at runtime, and
a layout that disagrees with the firmware by one byte produces a car that
drives somewhere plausible and wrong. The decoders below are written
independently of link.py, the way the C++ reads the bytes, so a test passing
means the two sides agree rather than that one file is self-consistent.
"""

import struct

import pytest

from control.intent import ActionIntent, SteerMode
from control.link import (DRIVE_LEN, DRIVE_PAYLOAD, F_CAM_OK, F_ENABLE,
                          F_LIDAR_OK, F_MODE_MASK, F_MODE_SHIFT,
                          F_PILLAR_SEEN, F_REVERSE, RANGE_NONE, S_ARC_DONE,
                          S_RECOVERING, TELEM_LEN, TELEM_PAYLOAD, Telemetry,
                          pack_drive, unpack_telem, xor8)


# --------------------------------------------------- an independent reader

def read_drive(frame):
    """Decode a DRIVE frame the way ObstacleRound.cpp's acceptDrive() does:
    by offset, not by struct format."""
    assert len(frame) == DRIVE_LEN
    assert frame[0] == 0xAA and frame[1] == 0x55, "sync"
    p = frame[2:2 + DRIVE_PAYLOAD]
    assert xor8(p) == frame[-1], "checksum"
    return {
        "seq": p[0],
        "flags": p[1],
        "mode": (p[1] & F_MODE_MASK) >> F_MODE_SHIFT,
        "heading": struct.unpack_from("<h", p, 2)[0],
        "steer": struct.unpack_from("<h", p, 4)[0],
        "speed": p[6],
        "arc_lock": p[7],
        "left": struct.unpack_from("<H", p, 8)[0],
        "front": struct.unpack_from("<H", p, 10)[0],
        "right": struct.unpack_from("<H", p, 12)[0],
        "rev": p[14],
        "cmd": p[15],
    }


def make_telem(seq=0, status=0, heading=0.0, yaw=0.0, odo=0, servo=0.0,
               floor=0, tries=0, boot=0):
    """Build a TELEM frame the way the firmware's sendTelem() does."""
    p = bytearray(TELEM_PAYLOAD)
    p[0] = seq & 0xFF
    p[1] = status
    struct.pack_into("<h", p, 2, int(round(heading * 10)))
    struct.pack_into("<h", p, 4, int(round(yaw * 10)))
    struct.pack_into("<i", p, 6, odo)
    struct.pack_into("<h", p, 10, int(round(servo * 10)))
    p[12] = floor
    p[13] = tries
    struct.pack_into("<I", p, 14, boot)
    return bytes(p)


# ----------------------------------------------------------------- sizes

def test_frame_sizes_match_the_spec():
    assert DRIVE_LEN == 19 and DRIVE_PAYLOAD == 16
    assert TELEM_LEN == 21 and TELEM_PAYLOAD == 18
    assert DRIVE_LEN == 2 + DRIVE_PAYLOAD + 1
    assert TELEM_LEN == 2 + TELEM_PAYLOAD + 1


def test_sync_words_are_reversed_and_non_ascii():
    """A frame must never be mistakable for one going the other way, and a
    log line must never be mistakable for a frame."""
    f = pack_drive(0, ActionIntent.stop(), None, None, None, 0,
                   False, False, False)
    assert f[:2] == b"\xAA\x55"
    t = b"\x55\xAA"
    assert f[:2] == bytes(reversed(t))
    assert 0xAA > 0x7F, "0xAA cannot appear in ASCII text"


# ----------------------------------------------------------------- DRIVE

def test_drive_round_trip():
    intent = ActionIntent(mode=SteerMode.HEADING_HOLD,
                          target_heading_deg=-12.3, steer_deg=4.5,
                          speed_pwm=60, arc_lock=0.7)
    f = pack_drive(7, intent, 412, 1873, 655, 200,
                   lidar_ok=True, cam_ok=True, pillar_seen=False)
    d = read_drive(f)
    assert d["seq"] == 7
    assert d["mode"] == SteerMode.HEADING_HOLD
    assert d["heading"] == -123
    assert d["steer"] == 45
    assert d["speed"] == 60
    assert d["arc_lock"] == 70
    assert (d["left"], d["front"], d["right"]) == (412, 1873, 655)
    assert d["rev"] == 200


def test_drive_flags():
    i = ActionIntent(mode=SteerMode.ARC, speed_pwm=60)
    d = read_drive(pack_drive(0, i, None, None, None, 0,
                              lidar_ok=True, cam_ok=True, pillar_seen=True))
    assert d["flags"] & F_ENABLE
    assert d["flags"] & F_LIDAR_OK
    assert d["flags"] & F_CAM_OK
    assert d["flags"] & F_PILLAR_SEEN
    assert d["mode"] == SteerMode.ARC


def test_enable_is_clear_when_stopped():
    """ENABLE must never be set by a STOP, whatever the speed field says."""
    d = read_drive(pack_drive(0, ActionIntent(mode=SteerMode.STOP,
                                              speed_pwm=200),
                              None, None, None, 0, True, True, False))
    assert not (d["flags"] & F_ENABLE)


def test_enable_is_clear_at_zero_speed():
    d = read_drive(pack_drive(0, ActionIntent(mode=SteerMode.HEADING_HOLD,
                                              speed_pwm=0),
                              None, None, None, 0, True, True, False))
    assert not (d["flags"] & F_ENABLE)


def test_reverse_flag():
    i = ActionIntent(mode=SteerMode.DIRECT, speed_pwm=60, reverse=True)
    d = read_drive(pack_drive(0, i, None, None, None, 0, False, False, False))
    assert d["flags"] & F_REVERSE


def test_missing_ranges_are_the_sentinel_not_zero():
    """A beam with no return must read FAR, never near - a zero here would
    look like a wall touching the car and fire the panic reflex."""
    d = read_drive(pack_drive(0, ActionIntent.stop(), None, None, None, 0,
                              False, False, False))
    assert d["left"] == d["front"] == d["right"] == RANGE_NONE


def test_out_of_range_distance_becomes_the_sentinel():
    d = read_drive(pack_drive(0, ActionIntent.stop(), 70000, -5, float("inf"),
                              0, False, False, False))
    assert d["left"] == d["front"] == d["right"] == RANGE_NONE


def test_seq_and_rev_wrap_without_overflowing():
    d = read_drive(pack_drive(300, ActionIntent.stop(), None, None, None, 777,
                              False, False, False))
    assert d["seq"] == 300 & 0xFF
    assert d["rev"] == 777 & 0xFF


def test_heading_is_wrapped_by_the_mapper_not_the_packer():
    """pack_drive clamps rather than wraps, so a caller that skips the mapper
    gets a saturated value instead of a silently wrong one."""
    i = ActionIntent(mode=SteerMode.HEADING_HOLD, target_heading_deg=5000.0,
                     speed_pwm=60)
    d = read_drive(pack_drive(0, i, None, None, None, 0, False, False, False))
    assert d["heading"] == 32767


def test_checksum_covers_every_payload_byte():
    f = bytearray(pack_drive(1, ActionIntent(mode=SteerMode.HEADING_HOLD,
                                             speed_pwm=60),
                             100, 200, 300, 5, True, True, True))
    for i in range(2, 2 + DRIVE_PAYLOAD):
        bad = bytearray(f)
        bad[i] ^= 0x01
        assert xor8(bytes(bad[2:2 + DRIVE_PAYLOAD])) != bad[-1], \
            f"a flipped bit at offset {i} must break the checksum"


# ----------------------------------------------------------------- TELEM

def test_telem_round_trip():
    t = unpack_telem(make_telem(seq=9, status=S_ARC_DONE, heading=-45.6,
                                yaw=120.0, odo=-98765, servo=76.5, floor=1,
                                tries=2, boot=0xDEADBEEF))
    assert t.seq == 9
    assert t.heading_deg == pytest.approx(-45.6)
    assert t.yaw_rate_dps == pytest.approx(120.0)
    assert t.odo_ticks == -98765
    assert t.servo_deg == pytest.approx(76.5)
    assert t.floor == 1 and t.floor_name == "orange"
    assert t.recover_tries == 2
    assert t.boot_id == 0xDEADBEEF
    assert t.arc_done and not t.recovering


def test_telem_odometry_is_signed():
    """The encoder free-runs and can go negative when the car reverses."""
    t = unpack_telem(make_telem(odo=-2_000_000_000))
    assert t.odo_ticks == -2_000_000_000


def test_telem_status_bits():
    t = unpack_telem(make_telem(status=S_RECOVERING))
    assert t.recovering and not t.arc_done
    assert not t.enabled


def test_telem_freshness():
    t = Telemetry(stamp=100.0)
    assert t.fresh(now=100.1, stale_s=0.3)
    assert not t.fresh(now=100.5, stale_s=0.3)


def test_default_telemetry_is_stale():
    """A Telemetry nobody has filled in must not look like a live STM32."""
    assert not Telemetry().fresh(now=1000.0)


# ----------------------------------------------------------- mode values

def test_mode_values_are_the_wire_values():
    """The firmware switches on these integers directly."""
    assert int(SteerMode.STOP) == 0
    assert int(SteerMode.HEADING_HOLD) == 1
    assert int(SteerMode.DIRECT) == 2
    assert int(SteerMode.ARC) == 3


def test_every_mode_survives_the_flags_byte():
    for m in SteerMode:
        i = ActionIntent(mode=m, speed_pwm=60)
        d = read_drive(pack_drive(0, i, None, None, None, 0,
                                  False, False, False))
        assert d["mode"] == int(m)
