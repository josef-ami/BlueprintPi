"""
ObstacleLap command handling — the CMD byte (PERCEPT byte 15) as the loop
holds it. Fake lidar, fake link, no hardware.

obstacle_lap imports the camera module, which needs OpenCV; the Pi-only
bindings (libcamera, rplidarc1) are stubbed here so this runs on a laptop.
"""

import os
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("cv2")
pytest.importorskip("numpy")
if "libcamera" not in sys.modules:
    try:
        import libcamera  # noqa: F401
    except ImportError:
        m = types.ModuleType("libcamera")
        m.Transform = lambda **k: None
        sys.modules["libcamera"] = m
if "rplidarc1" not in sys.modules:
    try:
        import rplidarc1  # noqa: F401
    except ImportError:
        m = types.ModuleType("rplidarc1")
        m.RPLidar = object
        sys.modules["rplidarc1"] = m

import obstacle_lap as ol                                          # noqa: E402
from control import percept_link as pl                             # noqa: E402
from worldstate import SharedState                                 # noqa: E402

CFG = {"lidar": {"bearing_tol_deg": 2}, "fusion": {},
       "avoid": {"wall_guard": False}}


class FakeLidar:
    def __init__(self, live=True):
        self.rev = 41
        self.live = live
        self._ranges = [float("inf")] * 360
        self._quals = [0] * 360
        for d in (0, 90, 270):
            self._ranges[d] = 800.0
            self._quals[d] = 40

    def is_alive(self):
        return True

    @property
    def last_point_t(self):
        return time.monotonic() if self.live else 0.0

    def queue_depth(self):
        return 0


class FakeLink:
    def __init__(self):
        self.frames = []
        self.state = pl.ST_HEADING

    def telemetry(self):
        return pl.Telemetry(stamp=time.time(), state=self.state)

    def status(self):
        return None

    def send(self, **kw):
        f = pl.pack_percept(len(self.frames), **kw)
        self.frames.append(f)
        return f


def lap(live=True, dry=False):
    lines = []
    link = FakeLink()
    o = ol.ObstacleLap(CFG, SharedState(), FakeLidar(live), link=link, dry=dry,
                       quiet=True, emit=lines.append)
    return o, link, lines


def cmd_of(frame):
    return frame[15]


def test_no_command_by_default():
    o, link, _ = lap()
    o.tick()
    assert cmd_of(link.frames[-1]) == pl.CMD_NONE


def test_stop_is_held_until_telem_shows_stopped():
    o, link, lines = lap()
    o.request(pl.CMD_STOP)
    for _ in range(5):
        o.tick()
    assert all(cmd_of(f) == pl.CMD_STOP for f in link.frames[-5:])
    link.state = pl.ST_STOPPED
    time.sleep(0.001)
    o.tick()
    assert cmd_of(link.frames[-1]) == pl.CMD_NONE
    assert o.pending_command() == pl.CMD_NONE
    assert any("STOP acknowledged" in l for l in lines)


def test_rerun_is_held_until_boot():
    o, link, lines = lap()
    link.state = pl.ST_FINISH
    o.request(pl.CMD_RERUN)
    o.tick()
    assert cmd_of(link.frames[-1]) == pl.CMD_RERUN
    link.state = pl.ST_BOOT
    time.sleep(0.001)
    o.tick()
    assert cmd_of(link.frames[-1]) == pl.CMD_NONE
    assert any("RERUN acknowledged" in l for l in lines)


def test_a_command_goes_out_even_with_the_lidar_dead_but_never_as_ranges():
    o, link, _ = lap(live=False)
    o.tick()
    assert link.frames == []                     # silence is the failure signal
    o.request(pl.CMD_STOP)
    o.tick()
    f = link.frames[-1]
    assert cmd_of(f) == pl.CMD_STOP
    flags = f[3]
    assert not flags & pl.P_LIDAR_OK and not flags & pl.P_HELLO
    assert f[4:10] == b"\xff" * 6                # no ranges
    assert f[10] == 41                           # repeats the real rev count


def test_an_unacknowledged_command_is_dropped_after_the_timeout(monkeypatch):
    o, link, lines = lap()
    o.request(pl.CMD_STOP)
    o.tick()
    real = time.time
    monkeypatch.setattr(ol.time, "time", lambda: real() + ol.CMD_TIMEOUT_S + 1)
    o.tick()
    assert o.pending_command() == pl.CMD_NONE
    assert any("not acknowledged" in l for l in lines)


def test_dry_runs_ignore_commands():
    o, link, lines = lap(dry=True)
    assert o.request(pl.CMD_STOP) is False
    o.tick()
    assert link.frames == []
    assert any("ignored" in l for l in lines)


def test_reboot_is_not_a_held_command():
    o, _, _ = lap()
    with pytest.raises(ValueError):
        o.request(pl.CMD_REBOOT)


def test_tick_snapshot_carries_what_was_sent():
    o, link, _ = lap()
    o.request(pl.CMD_STOP)
    t = o.tick()
    assert t.sent and t.cmd == pl.CMD_STOP
    assert t.lidar_live and t.front == 800.0 and t.left == 800.0
    assert t.picks[0][0] == 0 and t.picks[1][0] == 90 and t.picks[2][0] == 270
