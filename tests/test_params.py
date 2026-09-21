"""
The parameter registry and its persistence.

config.json is the single source of truth now, so the thing most worth
guarding is that saving tuning cannot destroy the camera calibration sitting
next to it in the same file.
"""

import json
import os
import tempfile

import pytest

import params as prm


def test_every_spec_has_a_sane_range():
    for s in prm.PI_SPECS:
        assert s.lo <= s.hi, s.name
        if s.kind != "b":
            assert s.lo <= s.default <= s.hi, f"{s.name} default out of range"


def test_names_are_unique():
    names = [s.name for s in prm.PI_SPECS]
    assert len(names) == len(set(names))


def test_groups_are_all_known_to_the_page():
    """The Tune tab lays out by group; a group it does not know about would
    silently hide its parameters."""
    page_groups = {"camera", "hsv", "lab", "filter", "cone", "locate", "cand",
                   "plan", "pass", "level", "turn", "corner", "mnv", "run",
                   "link"}
    assert {s.group for s in prm.PI_SPECS} <= page_groups


def test_coercion_clamps_and_types(p):
    assert p.set("MIN_AREA_PROC", 10_000) == 5000      # hi
    assert p.set("MIN_AREA_PROC", -5) == 20            # lo
    assert isinstance(p.set("MIN_AREA_PROC", 250.7), int)
    assert p.set("USE_LAB", "yes") is True
    assert p.set("USE_LAB", "off") is False


def test_nan_is_rejected(p):
    with pytest.raises(ValueError):
        p.set("CORRIDOR_MM", float("nan"))


def test_unknown_name_raises(p):
    with pytest.raises(KeyError):
        p.set("NOT_A_PARAMETER", 1)


def test_set_many_ignores_unknown_names(p):
    out = p.set_many({"CORRIDOR_MM": 1100, "NONSENSE": 5})
    assert out == {"CORRIDOR_MM": 1100.0}


def test_reset_restores_every_default(p):
    p.set("CORRIDOR_MM", 1234)
    p.set("USE_LAB", True)
    p.reset()
    assert p["CORRIDOR_MM"] == 1000.0
    assert p["USE_LAB"] is False


def test_readers_never_see_a_half_set(p):
    """Values live in one dict that is REPLACED, never mutated."""
    before = p.snapshot()
    p.set("CORRIDOR_MM", 1234)
    assert before is not p.snapshot()
    assert before["CORRIDOR_MM"] == 1000.0, "the old snapshot is intact"


def test_hsv_config_shape(p):
    cfg = p.hsv_config()
    assert len(cfg["RED"]) == 2, "red needs two ranges - the hue wraps"
    assert len(cfg["GREEN"]) == 1
    for ranges in cfg.values():
        for lo, hi in ranges:
            assert len(lo) == 3 and len(hi) == 3


def test_lab_config_shape(p):
    cfg = p.lab_config()
    for lo, hi in cfg.values():
        assert len(lo) == 3 and len(hi) == 3


def test_pillar_filter_has_both_halves(p):
    """The merge: four tests from the obstacle round, plus the linked-floor
    test from the calibration detector."""
    pf = p.pillar_filter()
    assert {"aspect_min", "solidity_min", "floor_below_min",
            "contrast_s_min"} <= set(pf)
    assert {"linked_test", "dark_v_max", "ignore_bottom_px"} <= set(pf)


# --------------------------------------------------------- persistence

def test_save_preserves_the_blocks_it_does_not_own():
    """Saving tuning must never drop the camera calibration."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        intrinsics = {"model": "fisheye", "K": [[383, 0, 330]], "D": [0.05]}
        with open(path, "w") as f:
            json.dump({"camera_intrinsics": intrinsics,
                       "lidar": {"port": "/dev/ttyUSB0"},
                       "serial": {"port": "/dev/ttyACM0"}}, f)

        p, stm = prm.PiParams(), prm.StmParams()
        p.set("CORRIDOR_MM", 1234)
        prm.save(p, stm, path=path)

        got = json.load(open(path))
        assert got["camera_intrinsics"] == intrinsics
        assert got["lidar"]["port"] == "/dev/ttyUSB0"
        assert got["serial"]["port"] == "/dev/ttyACM0"
        assert got["params"]["CORRIDOR_MM"] == 1234


def test_load_round_trips():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        p, stm = prm.PiParams(), prm.StmParams()
        p.set("CORRIDOR_MM", 1234)
        p.set("USE_LAB", True)
        stm.desired["HEAD_KP"] = 3.5
        prm.save(p, stm, path=path)

        p2, stm2 = prm.PiParams(), prm.StmParams()
        prm.load(p2, stm2, path=path)
        assert p2["CORRIDOR_MM"] == 1234
        assert p2["USE_LAB"] is True
        assert stm2.desired["HEAD_KP"] == 3.5


def test_load_from_a_missing_file_is_not_fatal():
    p, stm = prm.PiParams(), prm.StmParams()
    msg = prm.load(p, stm, path="/nonexistent/config.json")
    assert "no config" in msg
    assert p["CORRIDOR_MM"] == 1000.0


def test_save_is_atomic():
    """A power cut must not truncate the file."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        prm.save(prm.PiParams(), prm.StmParams(), path=path)
        assert not os.path.exists(path + ".tmp")
        json.load(open(path))


# ------------------------------------------------------------ STM mirror

def test_stm_adopts_the_firmware_default_when_it_has_no_opinion():
    stm = prm.StmParams()
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    assert stm.desired["HEAD_KP"] == 2.0
    assert stm.live["HEAD_KP"] == 2.0


def test_stm_keeps_a_saved_value_over_the_firmware_default():
    stm = prm.StmParams()
    stm.desired["HEAD_KP"] = 3.5
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    assert stm.desired["HEAD_KP"] == 3.5, "ours wins; the firmware has no storage"
    assert stm.live["HEAD_KP"] == 2.0


def test_stm_queues_only_what_the_firmware_lacks():
    stm = prm.StmParams()
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    stm.on_line("!P 1 HEAD_KI 0 0.0000 0.0000 5.0000 0")
    stm.desired["HEAD_KP"] = 3.5           # differs
    stm.queue_all()
    lines = stm.drain(10)
    assert any("HEAD_KP" in l for l in lines)
    assert not any("HEAD_KI" in l for l in lines), "already correct"


def test_stm_set_clamps_to_the_firmware_range():
    stm = prm.StmParams()
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    assert stm.set("HEAD_KP", 99.0) == 10.0
    assert stm.set("HEAD_KP", -5.0) == 0.0


def test_stm_set_unknown_raises():
    stm = prm.StmParams()
    with pytest.raises(KeyError):
        stm.set("NOPE", 1)


def test_a_new_boot_id_triggers_a_reread():
    """The firmware has no storage: a reset means it is back at compiled-in
    defaults, so the whole table has to be re-read and re-pushed."""
    stm = prm.StmParams()
    stm.on_boot("4", 20, "1111")
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    assert stm.table
    note = stm.on_boot(None, None, "2222")
    assert note and "boot" in note
    assert not stm.table, "the mirror was cleared"
    assert not stm.synced
    assert "?P" in stm.drain(10)


def test_the_same_boot_id_changes_nothing():
    stm = prm.StmParams()
    stm.on_boot("4", 20, "1111")
    stm.on_line("!P 0 HEAD_KP 0 2.0000 0.0000 10.0000 0")
    stm.drain(10)
    assert stm.on_boot(None, None, "1111") is None
    assert stm.table


def test_synced_once_the_whole_table_has_arrived():
    stm = prm.StmParams()
    stm.on_boot("4", 2, "1111")
    stm.on_line("!P 0 A 0 1.0000 0.0000 10.0000 0")
    assert not stm.synced
    stm.on_line("!P 1 B 0 1.0000 0.0000 10.0000 0")
    assert stm.synced


def test_ticks_per_mm_comes_from_the_firmware():
    """It is a hardware fact, so the STM32 owns it - but the Pi's odometry
    needs it, so it is mirrored up."""
    stm = prm.StmParams()
    assert stm.ticks_per_mm() == pytest.approx(1.4853), "known-good fallback"
    stm.on_line("!P 0 TICKS_PER_CM 0 20.0000 1.0000 100.0000 0")
    assert stm.ticks_per_mm() == pytest.approx(2.0)


def test_error_lines_are_reported():
    stm = prm.StmParams()
    assert "rejected" in (stm.on_line("!E range") or "")


def test_malformed_lines_are_ignored():
    stm = prm.StmParams()
    assert stm.on_line("!P garbage") is None
    assert stm.on_line("!p") is None
    assert stm.on_line("!V one two") is None
