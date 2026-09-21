"""
The merged pillar detector.

This is the one place where two previously separate filters became one, so it
is the one most worth pinning down. Each test builds a synthetic frame whose
correct verdict is obvious by construction: a block standing on white mat is a
pillar; the same block on black, or lying flat, or floating over a wall, is
not.

The five tests, in the order the detector applies them:
    A  aspect     h >= aspect_min * w
    S  solidity   area >= solidity_min * bbox
    F  on floor   the strip under the blob is mostly white mat
    L  linked     that mat connects to the mat in front of the car
    C  contrast   the blob is more colourful than the mat under it
"""

import numpy as np
import pytest

import sensors.camera as camera
from sensors.camera import (PillarDetector, REJECT_ASPECT, REJECT_CONTRAST,
                            REJECT_FLOOR, REJECT_LINKED, REJECT_SOLIDITY)

W, H = camera.PROC_SIZE          # 320 x 240

WHITE = (235, 235, 235)
BLACK = (12, 12, 12)
RED = (220, 30, 30)
GREEN = (30, 200, 60)
GREY = (120, 120, 120)


def frame(bg=BLACK):
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :] = bg
    return f


def mat(f, top=140):
    """White mat filling the bottom of the view, which is the floor the car
    is standing on."""
    f[top:, :] = WHITE
    return f


def block(f, colour, cx, bottom, w=26, h=46):
    """An upright block standing with its base at `bottom`."""
    x0, x1 = cx - w // 2, cx + w // 2
    y0, y1 = bottom - h, bottom
    f[max(0, y0):y1, max(0, x0):x1] = colour
    return f


def verdicts(det):
    return {code for _, _, code, _ in det.rejected}


def detect(p, f, **masks):
    return PillarDetector(p).detect(f, **masks)


# ------------------------------------------------------------- accepted

def test_a_pillar_standing_on_the_mat_is_accepted(p):
    f = block(mat(frame()), RED, 160, 180)
    det = detect(p, f)
    assert len(det.pillars) == 1
    assert det.pillars[0].colour == "RED"
    assert det.rejected == []


def test_green_is_named_green(p):
    f = block(mat(frame()), GREEN, 160, 180)
    det = detect(p, f)
    assert [x.colour for x in det.pillars] == ["GREEN"]


def test_two_pillars_are_sorted_largest_first(p):
    f = mat(frame())
    block(f, RED, 90, 180, w=20, h=34)
    block(f, GREEN, 220, 190, w=34, h=58)
    det = detect(p, f)
    assert len(det.pillars) == 2
    assert det.pillars[0].area > det.pillars[1].area
    assert det.pillars[0].colour == "GREEN"


def test_best_is_the_largest_accepted(p):
    f = mat(frame())
    block(f, RED, 90, 180, w=20, h=34)
    block(f, GREEN, 220, 190, w=34, h=58)
    det = detect(p, f)
    assert det.best is det.pillars[0]


def test_nothing_there_is_nothing(p):
    det = detect(p, mat(frame()))
    assert det.pillars == [] and det.best is None


# --------------------------------------------------------------- A and S

def test_a_flat_stripe_is_rejected_on_aspect(p):
    """An orange or red floor line is wide and flat."""
    f = mat(frame())
    f[178:184, 60:260] = RED            # 200 x 6
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_ASPECT in verdicts(det)


def test_aspect_threshold_is_the_tunable(p):
    f = mat(frame())
    f[150:180, 140:180] = RED           # 40 wide, 30 high -> h/w = 0.75
    assert detect(p, f).pillars == [], "0.75 < aspect_min 0.8"
    p.set("aspect_min", 0.5)
    assert len(detect(p, f).pillars) == 1


def test_a_ragged_shape_is_rejected_on_solidity(p):
    """A diagonal streak is tall enough to pass the aspect test but fills its
    bounding box poorly - which is what solidity is for. (A scatter of specks
    would not work here: the mask's 5x5 opening erases them before the shape
    tests ever see it.)"""
    import cv2
    f = mat(frame())
    cv2.line(f, (142, 122), (180, 180), RED, 9)
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_SOLIDITY in verdicts(det)


# ------------------------------------------------------------------- F

def test_a_block_not_on_the_mat_is_rejected(p):
    """Red high on a black background - a shirt, a banner - has no mat under
    it at all."""
    f = frame()
    mat(f, top=200)
    block(f, RED, 160, 120)             # base at 120, well above the mat
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_FLOOR in verdicts(det)


def test_floor_below_min_is_the_tunable(p):
    """A block whose base sits just above the mat: the contact strip is part
    background, part mat, so the threshold decides."""
    f = frame()
    mat(f, top=190)
    block(f, RED, 160, 186)             # strip y188..194 straddles the edge
    assert len(detect(p, f).pillars) == 1, "0.45 accepts a mostly-mat strip"
    p.set("floor_below_min", 0.95)
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_FLOOR in verdicts(det)


def test_a_pillar_too_close_to_see_its_base_is_accepted(p):
    """Its base runs off the bottom of the frame. Nothing but a pillar can be
    that close, so F, L and C are skipped."""
    f = frame()
    mat(f, top=100)
    f[120:H, 140:180] = RED             # base at the image bottom
    det = detect(p, f)
    assert len(det.pillars) == 1, "base below view must still be a pillar"


# ------------------------------------------------------------------- L

def test_mat_beyond_a_wall_does_not_count(p):
    """The linked test is what this merge added. A red block standing on
    white, with a black wall between that white and the car, is on the NEXT
    straight's mat - not ours."""
    f = frame()
    mat(f, top=190)                     # our mat, at the bottom of the view
    f[120:150, :] = WHITE               # another patch of mat, further away
    f[150:190, :] = BLACK               # a wall between the two
    block(f, RED, 160, 120, h=40)       # standing ON the far patch, so the
                                        # strip under it is that far mat
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_LINKED in verdicts(det), \
        "the contact strip IS mat - only the link tells them apart"


def test_the_same_block_is_accepted_once_the_wall_is_gone(p):
    """Same geometry, no wall: now the mat under it is our mat."""
    f = frame()
    mat(f, top=120)                     # one continuous mat
    block(f, RED, 160, 150, h=40)
    det = detect(p, f)
    assert len(det.pillars) == 1


def test_linked_test_can_be_switched_off(p):
    p.set("linked_test", False)
    f = frame()
    mat(f, top=190)
    f[120:150, :] = WHITE
    f[150:190, :] = BLACK
    block(f, RED, 160, 120, h=40)
    det = detect(p, f)
    assert len(det.pillars) == 1, "without L, the contact test alone passes it"


def test_a_corner_line_does_not_break_the_link(p):
    """Orange and blue corner lines are not wall-dark, so mat either side of
    one is still connected."""
    f = frame()
    mat(f, top=120)
    f[160:172, :] = (200, 120, 40)      # an orange line across the mat
    block(f, GREEN, 160, 150, h=40)
    det = detect(p, f)
    assert len(det.pillars) == 1


# ------------------------------------------------------------------- C

def test_a_washed_out_blob_is_rejected_on_contrast(p):
    """Something barely more colourful than the mat under it is a reflection
    or a shadow, not a pillar."""
    f = mat(frame())
    block(f, (200, 90, 90), 160, 180)     # saturation 140: masked, but muted
    p.set("contrast_s_min", 200)          # demand more than it has
    det = detect(p, f)
    assert det.pillars == []
    assert REJECT_CONTRAST in verdicts(det)
    p.set("contrast_s_min", 50)
    assert len(detect(p, f).pillars) == 1


# ------------------------------------------------------- the Lab pathway

def test_lab_mode_finds_the_same_pillar(p):
    """Lab exists because HSV gates on saturation and loses a matte pillar in
    dim light. It must find an ordinary pillar too."""
    p.set("USE_LAB", True)
    p.set("RED_A_LO", 150)
    p.set("RED_A_HI", 255)
    p.set("RED_L_LO", 20)
    p.set("RED_L_HI", 255)
    p.set("RED_B_LO", 128)
    p.set("RED_B_HI", 255)
    f = block(mat(frame()), RED, 160, 180)
    det = detect(p, f)
    assert det.space == "Lab"
    assert len(det.pillars) == 1 and det.pillars[0].colour == "RED"


def test_space_is_reported(p):
    assert detect(p, mat(frame())).space == "HSV"
    p.set("USE_LAB", True)
    assert detect(p, mat(frame())).space == "Lab"


# ---------------------------------------------------------------- output

def test_min_area_filters_specks(p):
    f = mat(frame())
    f[176:180, 158:162] = RED            # 4 x 4
    assert detect(p, f).pillars == []


def test_boxes_are_in_full_frame_pixels(p):
    """Detection runs at 320x240; everything the page and the bearing model
    see is 640x480."""
    f = block(mat(frame()), RED, 160, 180)
    b = detect(p, f).pillars[0].box
    assert 0 <= b[0] < b[2] <= camera.FRAME_W
    assert 0 <= b[1] < b[3] <= camera.FRAME_H
    assert b[2] > W, "scaled up from detection resolution"


def test_err_px_is_signed_right_of_centre_positive(p):
    left = block(mat(frame()), RED, 80, 180)
    right = block(mat(frame()), RED, 240, 180)
    assert detect(p, left).pillars[0].err_px < 0
    assert detect(p, right).pillars[0].err_px > 0


def test_bearing_is_left_positive(p):
    """+ = LEFT of forward, matching the LiDAR frame. Getting this backwards
    mirrors every pass."""
    p.set("USE_INTRINSICS", False)       # the model with no calibration file
    left = block(mat(frame()), RED, 80, 180)
    right = block(mat(frame()), RED, 240, 180)
    bl = detect(p, left).pillars[0].bearing_deg
    br = detect(p, right).pillars[0].bearing_deg
    assert bl > br
    assert bl > p["CAMERA_OFFSET_DEG"] > br


def test_camera_offset_shifts_every_bearing(p):
    p.set("USE_INTRINSICS", False)
    p.set("CAMERA_OFFSET_DEG", 0.0)
    f = block(mat(frame()), RED, 160, 180)
    a = detect(p, f).pillars[0].bearing_deg
    p.set("CAMERA_OFFSET_DEG", 10.0)
    b = detect(p, f).pillars[0].bearing_deg
    assert b - a == pytest.approx(10.0, abs=0.5)


def test_masks_are_only_built_when_asked(p):
    f = block(mat(frame()), RED, 160, 180)
    assert detect(p, f).masks == {}
    det = detect(p, f, want_masks=True)
    assert set(det.masks) >= {"RED", "GREEN", "FLOOR"}


def test_rejected_carries_a_code_the_page_can_explain(p):
    f = mat(frame())
    f[178:184, 60:260] = RED
    det = detect(p, f)
    for _, _, code, _ in det.rejected:
        assert code in camera.REJECT_TEXT
