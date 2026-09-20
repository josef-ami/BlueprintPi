"""
Floor-contact pillar filter (sensors/camera.py) on synthetic frames:
white mat, black wall band, hall behind it. Only red/green that stands on the
mat, on the same floor the car is on, may become a pillar.

Needs OpenCV + numpy; the Pi-only libcamera binding is stubbed.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")
if "libcamera" not in sys.modules:
    try:
        import libcamera  # noqa: F401
    except ImportError:
        m = types.ModuleType("libcamera")
        m.Transform = lambda **k: None
        sys.modules["libcamera"] = m

import sensors.camera as camera                                     # noqa: E402

HSV = {"RED": [[[0, 120, 70], [10, 255, 255]], [[170, 120, 70], [180, 255, 255]]],
       "GREEN": [[[40, 80, 60], [85, 255, 255]]]}
RED, GREEN = (220, 30, 30), (40, 190, 70)
FLOOR = dict(camera.FLOOR_DEFAULTS)


def scene():
    """480x640 RGB: hall (rows 0-179), black wall (180-229), white mat (230-479)."""
    f = np.zeros((480, 640, 3), np.uint8)
    f[:180] = (120, 90, 60)          # the hall: brownish, not white, not dark
    f[180:230] = (20, 20, 20)        # the black wall
    f[230:] = (225, 225, 220)        # the white mat
    return f


def rect(f, x0, y0, x1, y1, rgb):
    f[y0:y1, x0:x1] = rgb


def detect(f, floor=FLOOR):
    rej, masks = [], {}
    blobs = camera.detect_blobs(f, HSV, 300, masks_out=masks, floor=floor,
                                rejected_out=rej)
    return blobs, rej, masks


def at(blobs, x):
    return [b for b in blobs if b["x"] <= x < b["x"] + b["w"]]


def test_a_pillar_on_the_mat_is_kept():
    f = scene()
    rect(f, 300, 260, 330, 330, RED)
    blobs, rej, _ = detect(f)
    assert len(blobs) == 1 and blobs[0]["colour"] == "RED"
    assert blobs[0]["floor"] == "on floor" and blobs[0]["white_frac"] > 0.9
    assert rej == []


def test_a_red_banner_beyond_the_wall_is_rejected():
    f = scene()
    rect(f, 100, 60, 160, 120, RED)
    blobs, rej, _ = detect(f)
    assert blobs == []
    assert rej[0]["reason"] == camera.REASON_NO_FLOOR


def test_green_on_a_white_card_beyond_the_wall_is_not_linked():
    """White under it, but the black wall separates it from our floor."""
    f = scene()
    rect(f, 400, 40, 500, 170, (235, 235, 235))       # a white card in the hall
    rect(f, 430, 60, 470, 100, GREEN)
    blobs, rej, _ = detect(f)
    assert blobs == []
    assert rej[0]["reason"] == camera.REASON_NOT_LINKED


def test_an_orange_corner_line_does_not_break_the_link():
    f = scene()
    rect(f, 0, 350, 640, 362, (230, 120, 30))         # orange line across the floor
    rect(f, 0, 380, 640, 392, (30, 80, 220))          # and a blue one
    rect(f, 200, 250, 230, 320, GREEN)                # a pillar beyond both
    blobs, rej, _ = detect(f)
    assert [b["colour"] for b in blobs] == ["GREEN"] and rej == []


def test_a_pillar_against_the_wall_is_kept():
    f = scene()
    rect(f, 560, 200, 585, 252, RED)                  # overlaps the wall band
    blobs, rej, _ = detect(f)
    assert len(at(blobs, 570)) == 1 and rej == []


def test_a_pillar_whose_base_is_below_the_view_is_kept():
    f = scene()
    rect(f, 500, 400, 560, 480, RED)
    blobs, _, _ = detect(f)
    assert blobs[0]["floor"] == camera.REASON_BELOW_VIEW


def test_the_cars_own_body_at_the_bottom_can_be_ignored():
    f = scene()
    f[455:] = (15, 15, 15)                            # dark chassis across the bottom
    rect(f, 100, 60, 160, 120, RED)                   # banner beyond the wall
    rect(f, 300, 260, 330, 330, RED)                  # pillar on the mat
    floor = dict(FLOOR, ignore_bottom_px=25)
    blobs, rej, _ = detect(f, floor)
    assert len(blobs) == 1 and blobs[0]["x"] == 300
    assert len(rej) == 1


def test_masks_hold_only_accepted_pillars_plus_the_floor():
    f = scene()
    rect(f, 100, 60, 160, 120, RED)                   # rejected
    rect(f, 300, 260, 330, 330, RED)                  # kept
    _, _, masks = detect(f)
    assert masks["RED"][90, 130] == 0                 # banner pixels gone
    assert masks["RED"][300, 315] == 255              # pillar pixels kept
    assert masks["FLOOR"][400, 50] == 255 and masks["FLOOR"][200, 50] == 0


def test_disabled_filter_keeps_everything_and_raw_masks():
    f = scene()
    rect(f, 100, 60, 160, 120, RED)
    rect(f, 300, 260, 330, 330, RED)
    masks = {}
    blobs = camera.detect_blobs(f, HSV, 300, masks_out=masks,
                                floor=dict(FLOOR, enabled=False))
    assert len(blobs) == 2 and "FLOOR" not in masks
    assert masks["RED"][90, 130] == 255


def test_no_floor_argument_is_the_old_behaviour():
    f = scene()
    rect(f, 100, 60, 160, 120, RED)
    assert len(camera.detect_blobs(f, HSV, 300)) == 1


def test_link_check_can_be_switched_off():
    f = scene()
    rect(f, 400, 40, 500, 170, (235, 235, 235))
    rect(f, 430, 60, 470, 100, GREEN)
    blobs, _, _ = detect(f, dict(FLOOR, linked=False))
    assert len(blobs) == 1


def test_missing_keys_fall_back_to_defaults():
    p = camera.floor_params({"white_v_min": 150, "bogus": 1})
    assert p["white_v_min"] == 150 and p["strip_px"] == FLOOR["strip_px"]
    assert "bogus" not in p
