"""TELEM decoding, against frames built exactly as the firmware builds them."""

import struct

import pytest

from perception import telemetry as tm
from perception.nav import Navigator


def build(heading_dd=0, yaw_dd=0, odo=0, servo=0, floorc=0, tries=0,
          boot=1, seq=0, status=tm.S_IMU_OK, corrupt=False):
    """Mirror firmware/ObstacleRound.cpp::sendTelem()."""
    payload = struct.pack(tm._FMT, seq, status, heading_dd, yaw_dd, odo,
                          servo, floorc, tries, boot)
    chk = tm.xor8(payload)
    if corrupt:
        chk ^= 0xFF
    return tm.SYNC + payload + bytes([chk])


def test_frame_is_21_bytes_with_the_right_sync():
    f = build()
    assert len(f) == tm.TELEM_LEN == 21
    assert f[0:2] == b"\x55\xaa"


def test_decode_round_trips_heading_and_odometry():
    f = build(heading_dd=-1234, odo=98765, boot=7)
    t = tm.decode(f)
    assert t is not None
    assert t["heading_deg"] == pytest.approx(-123.4)
    assert t["odo_ticks"] == 98765
    assert t["boot_id"] == 7
    assert t["imu_ok"] is True


def test_bad_checksum_is_rejected():
    assert tm.decode(build(corrupt=True)) is None


def test_reader_feeds_navigator_and_handles_reboot():
    nav = Navigator()
    nav.state = "LOCKED"                      # so predict() is applied
    r = tm.TelemetryReader(nav, log=lambda *_a: None)

    r._feed(tm.decode(build(heading_dd=0, odo=0, boot=1)))
    x0, y0, _th = nav.wb.pose
    # 1000 ticks forward at the documented scale
    r._feed(tm.decode(build(heading_dd=0, odo=1000, boot=1)))
    moved = abs(nav.wb.pose[0] - x0) + abs(nav.wb.pose[1] - y0)
    assert moved == pytest.approx(1000 / nav.ticks_per_mm, rel=0.02)

    # a reboot restarts the encoder: must re-baseline, not dead-reckon the jump
    pose_before = nav.wb.pose
    r._feed(tm.decode(build(heading_dd=0, odo=0, boot=2)))
    assert nav.wb.pose == pose_before


def test_odometry_freshness_gates_the_tracking_mode():
    nav = Navigator()
    assert nav.has_odometry is False           # nothing received yet
    nav.state = "LOCKED"
    nav.on_telemetry(0.0, 0)
    nav.on_telemetry(0.0, 100)
    assert nav.has_odometry is True
