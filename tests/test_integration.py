"""
The wiring, end to end.

The unit tests exercise the logic; this exercises the plumbing - the run loop
with a fake LiDAR, and every dashboard endpoint through Flask's test client.
These are the paths where a typo produces an AttributeError at 50 Hz on the
mat rather than a failing assertion on a bench.
"""

import json
import math
import threading
import time

import numpy as np
import pytest

import params as prm
from worldstate import CameraResult, Pillar, SharedState


class FakeLidar:
    """Enough of a LidarThread for the loop to read."""

    def __init__(self, live=True):
        self._ranges = [float("inf")] * 360
        self._quals = [0] * 360
        self.rev = 0
        self.last_point_t = time.monotonic()
        self._live = live
        # a corridor: walls at 500 mm each side
        for d in range(-40, 41):
            for side, sign in (("l", 1), ("r", -1)):
                a = int(round(90 * sign + d)) % 360
                c = math.cos(math.radians(d))
                if c > 0.2:
                    self._ranges[a] = 500.0 / c
                    self._quals[a] = 40
        self._ranges[0] = 2400.0
        self._quals[0] = 40

    def is_alive(self):
        return self._live

    def queue_depth(self):
        return 0


def make_loop(dry=True):
    from control.obstacle_round import ObstacleRound
    p, stm = prm.PiParams(), prm.StmParams()
    shared = SharedState()
    lidar = FakeLidar()
    loop = ObstacleRound(p, stm, shared, lidar, link=None, dry=dry, quiet=True,
                         emit=lambda s: None)
    return loop, shared, lidar, p


# ----------------------------------------------------------- the run loop

def test_the_loop_ticks_without_a_robot():
    loop, shared, lidar, p = make_loop()
    for _ in range(5):
        t = loop.tick()
    assert t is not None
    assert t.intent is not None
    assert t.lidar_live


def test_the_loop_reads_the_walls():
    loop, shared, lidar, p = make_loop()
    loop.tick()
    t = loop.tick()
    assert t.wall.left_mm == pytest.approx(500, rel=0.1)
    assert t.wall.right_mm == pytest.approx(500, rel=0.1)


def test_the_loop_locates_a_pillar_the_camera_reports():
    loop, shared, lidar, p = make_loop()
    # a pillar 700 mm ahead, slightly left
    for d in range(-3, 4):
        a = int(round(math.degrees(math.atan2(60, 700)))) + d
        lidar._ranges[a % 360] = math.hypot(700, 60)
        lidar._quals[a % 360] = 40
    shared.set_camera(CameraResult(
        timestamp=time.time(),
        pillars=[Pillar(colour="RED", bearing_deg=4.9, err_px=40, area=700)],
        best=Pillar(colour="RED", bearing_deg=4.9, err_px=40, area=700),
        seq=1))
    loop.tick()
    t = loop.tick()
    assert t.cam_live
    assert t.pillar_colour == "RED"
    assert t.pillar_xy is not None
    assert t.pillar_xy[0] == pytest.approx(700, abs=150)


def test_a_full_start_stop_cycle():
    loop, shared, lidar, p = make_loop()
    loop.tick()
    loop.start()
    for _ in range(4):
        t = loop.tick()
    assert loop.fsm.state.name == "DRIVE_TO_CORNER"
    loop.stop()
    t = loop.tick()
    assert loop.fsm.state.name == "FINISHED"


def test_a_stale_lidar_does_not_crash_the_loop():
    loop, shared, lidar, p = make_loop()
    lidar._live = False
    lidar.last_point_t = time.monotonic() - 10
    for _ in range(3):
        t = loop.tick()
    assert not t.lidar_live
    assert t.intent is not None


def test_the_tick_is_json_safe():
    """The dashboard serialises it every 200 ms."""
    loop, shared, lidar, p = make_loop()
    loop.tick()
    t = loop.tick()
    json.dumps(t.fsm)


def test_the_banner_does_not_blow_up():
    loop, shared, lidar, p = make_loop()
    loop.banner()


# ----------------------------------------------------------- the dashboard

@pytest.fixture
def client():
    import dashboard
    dashboard.app.config["TESTING"] = True
    dashboard.lidar_thread = None
    dashboard.vision = None
    return dashboard.app.test_client()


def test_the_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    for tab in ("Run", "Tune", "Calibrate"):
        assert tab in body


def test_state_endpoint(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    d = r.get_json()
    for k in ("now", "active", "lidar", "wall", "log", "derived"):
        assert k in d


def test_state_endpoint_with_a_log_cursor(client):
    r = client.get("/api/state?since=0")
    assert r.status_code == 200
    assert "last" in r.get_json()["log"]
    r = client.get("/api/state?since=not-a-number")
    assert r.status_code == 200


def test_params_endpoint(client):
    d = client.get("/api/params").get_json()
    assert len(d["pi"]) == len(prm.PI_SPECS)
    assert "PASS_CLEAR_MM" in d["derived"]
    assert d["stm32_groups"] == prm.STM_GROUPS
    one = d["pi"][0]
    for k in ("name", "value", "lo", "hi", "kind", "group", "help"):
        assert k in one


def test_setting_a_parameter(client):
    r = client.post("/api/param", json={"side": "pi", "name": "CORRIDOR_MM",
                                        "value": 1100})
    d = r.get_json()
    assert d["ok"] and d["value"] == 1100
    # the derived values must come back with it, or the page shows stale ones
    assert d["derived"]["LANE_LIMIT_MM"] == pytest.approx(550 - 57 - 45)


def test_setting_a_bad_parameter(client):
    assert client.post("/api/param", json={"side": "pi", "name": "NOPE",
                                           "value": 1}).status_code == 404
    assert client.post("/api/param", json={"side": "what", "name": "x",
                                           "value": 1}).status_code == 400


def test_setting_an_stm32_parameter_the_firmware_has_not_declared(client):
    """Before the table has been read there is nothing to set, and the page
    must get a clean error rather than a traceback."""
    assert client.post("/api/param", json={"side": "stm32", "name": "HEAD_KP",
                                           "value": 3}).status_code == 404


def test_run_control_without_a_session(client):
    for path in ("/api/car/start", "/api/car/stop", "/api/run/end"):
        r = client.post(path)
        assert r.status_code == 409
        assert "no session" in r.get_json()["error"]


def test_starting_a_run_without_a_lidar_is_refused(client):
    r = client.post("/api/run/start", json={"dry": True})
    assert r.status_code == 409
    assert "lidar" in r.get_json()["error"]


def test_calibration_endpoints(client):
    d = client.get("/api/cal/state").get_json()
    assert [c["name"] for c in d["classes"]] == ["mat", "red", "green"]
    assert "vals" in d

    assert client.post("/api/cal/fitparams",
                       json={"margin": 12}).get_json()["fit"]["margin"] == 12
    assert client.post("/api/cal/dist",
                       json={"d": 800}).get_json()["dist"] == 800
    assert client.post("/api/cal/undo", json={"cls": "mat"}).get_json()["ok"]
    assert client.post("/api/cal/clear", json={"cls": "mat"}).get_json()["ok"]

    # no samples yet, so the fit must refuse rather than produce nonsense
    r = client.post("/api/cal/fit").get_json()
    assert not r["ok"] and "need at least" in r["msg"]


def test_sampling_without_a_camera_is_a_clean_error(client):
    r = client.post("/api/cal/sample", json={"cls": "mat", "x": 10, "y": 10})
    assert r.status_code == 503
    assert "no camera" in r.get_json()["error"]


def test_unknown_sample_class(client):
    r = client.post("/api/cal/sample", json={"cls": "banana", "x": 1, "y": 1})
    assert r.status_code == 400


def test_save_and_revert(client):
    import dashboard
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        real = prm.CONFIG_PATH
        prm.CONFIG_PATH = path
        try:
            assert client.post("/api/save").get_json()["ok"]
            assert os.path.exists(path)
            assert client.post("/api/revert",
                               json={"side": "pi"}).get_json()["ok"]
            assert client.post("/api/revert",
                               json={"side": "nope"}).status_code == 400
        finally:
            prm.CONFIG_PATH = real
            # Leave the process as we found it: these are module-level
            # singletons the other endpoint tests share.
            prm.load(dashboard.PI, dashboard.STM)
