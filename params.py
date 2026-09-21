#!/usr/bin/env python3
"""
params.py - the single place that knows what is tunable, and the only thing
that writes config.json.

There are two halves and they are deliberately different:

  PiParams    values this process reads directly every tick. Setting one takes
              effect on the next tick - nothing is restarted, and the camera
              and LiDAR threads are never touched. This is now most of the
              robot: the FSM, the lane planner, the pass planner, the corner
              trigger and the wall levelling all run on the Pi, so all of
              their numbers live here.

  StmParams   a MIRROR of the firmware's table, which is now small on purpose.
              The STM32 keeps only what describes the hardware and the loops
              that cannot run over a 50 Hz link: the servo geometry, the
              heading PID, the arc gains and the recovery reflex. It owns no
              storage, so it boots with compiled-in defaults and announces a
              new boot id; this side pushes the saved set back.

WHERE THE NUMBERS LIVE
    config.json
      camera_intrinsics   the fisheye calibration (structured, not a tunable)
      lidar / serial      device paths and rates (structured, not tunables)
      params              every Pi tunable below, flat, by name
      stm32               the desired value of every firmware tunable

  One file, one owner per number. There is no tuning.json any more; if you
  have one from an older build, `python3 -m params --migrate` folds it in.

Nothing here blocks. Pushes to the STM32 are queued and drained a few per loop
by whoever owns the serial port, so a tuning change can never stall the 50 Hz
frame feed.

Readers are lock-free: values live in one dict that is REPLACED, never
mutated, so a reader either sees the whole old set or the whole new one.
"""

import json
import os
import queue
import threading

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config.json")

# Legacy file from the split-config era, read once by --migrate.
LEGACY_TUNING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "tuning.json")


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------

class Spec:
    """One tunable: name, default, range, how to show it."""

    __slots__ = ("name", "default", "lo", "hi", "kind", "group", "help")

    def __init__(self, name, default, lo, hi, group, kind="f", help=""):
        self.name, self.default = name, default
        self.lo, self.hi = lo, hi
        self.kind = kind              # f float | i int | b bool
        self.group, self.help = group, help

    def coerce(self, v):
        """Value -> correct type, clamped. Raises ValueError on junk."""
        if self.kind == "b":
            if isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        v = float(v)
        if v != v:                    # NaN
            raise ValueError("not a number")
        v = max(self.lo, min(self.hi, v))
        return int(round(v)) if self.kind == "i" else v

    def as_dict(self, value):
        return {"name": self.name, "value": value, "default": self.default,
                "lo": self.lo, "hi": self.hi, "kind": self.kind,
                "group": self.group, "help": self.help}


# --------------------------------------------------------------------------
# Pi-side tunables
# --------------------------------------------------------------------------
#
# Groups are only how the Tune page lays them out.
#   camera   frame handling and the camera model
#   hsv      HSV colour ranges
#   lab      Lab colour ranges (calibrated by clicking, on the Calibrate tab)
#   filter   the "is this blob really a pillar" tests
#   cone     the 45 deg wall fits that give lane offset and wall angle
#   locate   turning a camera bearing + LiDAR range into a pillar position
#   cand     LiDAR-only objects whose colour is not known yet
#   plan     lane geometry and the lane-position planner
#   pass     passing a pillar: clearances, aim distances, giving up
#   level    pulling the IMU lane heading onto the fitted wall direction
#   turn     the corner trigger
#   corner   corner exit shaping - where the arc leaves the car
#   mnv      the 3-point corner
#   run      run-level constants
#   link     serial framing and rates
#
# A "(was firmware)" note marks a value that used to live in ObstacleRound.cpp
# and moved here when the FSM did. Its default is the firmware's, unchanged.

PI_SPECS = [
    # ---------------- camera ----------------
    Spec("MIN_AREA_PROC", 250, 20, 5000, "camera", "i",
         "smallest blob (in 320x240 px) that counts as a pillar"),
    Spec("VISION_STALE_S", 0.2, 0.05, 2.0, "camera", "f",
         "no camera frame for this long -> colour is treated as none"),
    Spec("JPEG_QUALITY", 50, 10, 95, "camera", "i", "preview stream quality"),
    Spec("CAMERA_FWD_MM", 0.0, -200, 400, "camera", "f",
         "camera ahead of the LiDAR; yours sits above it, so 0"),
    Spec("CAMERA_OFFSET_DEG", 5.0, -30, 30, "camera", "f",
         "camera-to-LiDAR rotational alignment"),
    Spec("HFOV_DEG", 160.0, 40, 200, "camera", "f",
         "lens horizontal field of view, used by the equidistant model"),
    Spec("USE_INTRINSICS", True, 0, 1, "camera", "b",
         "bearing from the calibrated fisheye K/D (off = equidistant HFOV_DEG). "
         "Your K says +/-48 deg; hfov_deg says +/-80. Measure before trusting either"),
    Spec("SWAP_RB", True, 0, 1, "camera", "b", "swap red and blue channels"),
    Spec("USE_LAB", False, 0, 1, "camera", "b",
         "classify pillars in CIE Lab instead of HSV. Lab separates red from green on "
         "the a channel without needing saturation, so a matte pillar under dim light "
         "still passes. Set by the Calibrate tab when you fit Lab ranges"),

    # ---------------- Lab ranges ----------------
    # OpenCV 8-bit: L 0-255, a/b 0-255 with 128 = neutral. Fitted by clicking
    # pillars and mat on the Calibrate tab. Defaults are deliberately
    # wide-open placeholders - calibrate before switching USE_LAB on.
    Spec("RED_L_LO",    20, 0, 255, "lab", "i", "red pillar: lightness"),
    Spec("RED_L_HI",   230, 0, 255, "lab", "i", ""),
    Spec("RED_A_LO",   150, 0, 255, "lab", "i", "a: >128 is red, <128 is green"),
    Spec("RED_A_HI",   255, 0, 255, "lab", "i", ""),
    Spec("RED_B_LO",   128, 0, 255, "lab", "i", "b: >128 is yellow, <128 is blue"),
    Spec("RED_B_HI",   255, 0, 255, "lab", "i", ""),
    Spec("GREEN_L_LO",  20, 0, 255, "lab", "i", "green pillar: lightness"),
    Spec("GREEN_L_HI", 230, 0, 255, "lab", "i", ""),
    Spec("GREEN_A_LO",   0, 0, 255, "lab", "i", ""),
    Spec("GREEN_A_HI", 110, 0, 255, "lab", "i", ""),
    Spec("GREEN_B_LO",   0, 0, 255, "lab", "i", ""),
    Spec("GREEN_B_HI", 255, 0, 255, "lab", "i", ""),
    Spec("FLOOR_L_MIN", 120, 0, 255, "lab", "i",
         "white mat in Lab: at least this bright ..."),
    Spec("FLOOR_AB_TOL", 14, 1, 80, "lab", "i",
         "... and within this of neutral (128) on both a and b"),
    Spec("LAB_CHROMA_MIN", 20, 0, 128, "lab", "i",
         "pillar chroma minus mat chroma under it; the Lab version of contrast_s_min"),

    # ---------------- HSV ranges ----------------
    Spec("RED1_H_LO",   0, 0, 180, "hsv", "i", "red range 1"),
    Spec("RED1_S_LO", 120, 0, 255, "hsv", "i", ""),
    Spec("RED1_V_LO",  70, 0, 255, "hsv", "i", ""),
    Spec("RED1_H_HI",  10, 0, 180, "hsv", "i", ""),
    Spec("RED1_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED1_V_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED2_H_LO", 170, 0, 180, "hsv", "i", "red range 2 (hue wraps)"),
    Spec("RED2_S_LO", 120, 0, 255, "hsv", "i", ""),
    Spec("RED2_V_LO",  70, 0, 255, "hsv", "i", ""),
    Spec("RED2_H_HI", 180, 0, 180, "hsv", "i", ""),
    Spec("RED2_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED2_V_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("GREEN_H_LO", 40, 0, 180, "hsv", "i", "green range"),
    Spec("GREEN_S_LO", 80, 0, 255, "hsv", "i", ""),
    Spec("GREEN_V_LO", 60, 0, 255, "hsv", "i", ""),
    Spec("GREEN_H_HI", 85, 0, 180, "hsv", "i", ""),
    Spec("GREEN_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("GREEN_V_HI", 255, 0, 255, "hsv", "i", ""),

    # ---------------- merged pillar filter ----------------
    # Four tests from the obstacle-round detector (A aspect, S solidity,
    # F floor contact, C contrast) plus the linked-floor test from the
    # calibration detector (L): the white under the blob must connect to the
    # white in front of the car through pixels that are not wall-dark, which
    # is what rejects a red shirt seen over the wall.
    Spec("floor_s_max", 60, 0, 255, "filter", "i", "mat: saturation at or below this"),
    Spec("floor_v_min", 120, 0, 255, "filter", "i", "... and value at or above this"),
    Spec("strip_px", 6, 1, 40, "filter", "i", "rows checked just under a blob"),
    Spec("floor_below_min", 0.45, 0.0, 1.0, "filter", "f",
         "fraction of that strip that must be mat"),
    Spec("aspect_min", 0.8, 0.0, 5.0, "filter", "f", "height / width"),
    Spec("solidity_min", 0.5, 0.0, 1.0, "filter", "f", "contour area / bbox area"),
    Spec("contrast_s_min", 50, 0, 255, "filter", "i",
         "blob saturation minus mat saturation under it (HSV mode)"),
    Spec("bottom_margin_px", 3, 0, 40, "filter", "i",
         "this close to the image bottom = base out of view, skip the floor tests"),
    Spec("linked_test", True, 0, 1, "filter", "b",
         "also require the mat under the blob to connect to the mat in front of the "
         "car. Rejects red/green things standing beyond the walls"),
    Spec("dark_v_max", 50, 0, 255, "filter", "i",
         "wall-dark: value at or below this breaks the link"),
    Spec("ignore_bottom_px", 0, 0, 200, "filter", "i",
         "bottom rows hidden by the car's own body"),

    # ---------------- cone wall fit ----------------
    Spec("CONE_DEG", 45, 10, 120, "cone", "i", "width of each side cone"),
    Spec("CONE_MAX_RANGE_MM", 1500, 300, 4000, "cone", "i", ""),
    Spec("CONE_MIN_RANGE_MM", 60, 10, 500, "cone", "i", ""),
    Spec("CONE_INLIER_MM", 25, 3, 200, "cone", "i", "RANSAC inlier band"),
    Spec("CONE_MIN_INLIERS", 8, 3, 60, "cone", "i", ""),
    Spec("CONE_MIN_SPAN_MM", 150, 20, 800, "cone", "i",
         "wall length the fit must cover; a 50 mm pillar face cannot"),
    Spec("CONE_AGREE_DEG", 6.0, 0.5, 45.0, "cone", "f",
         "left/right yaw must agree this well to be averaged"),

    # ---------------- pillar location ----------------
    Spec("RAY_WINDOW_DEG", 8.0, 1.0, 30.0, "locate", "f",
         "LiDAR returns this close to the camera ray are candidates"),
    Spec("PILLAR_MAX_MM", 2000.0, 300, 4000, "locate", "f", ""),
    Spec("AREA_K", 14000.0, 2000, 60000, "locate", "f",
         "distance ~ AREA_K / sqrt(area), the fallback when no LiDAR return agrees"),
    Spec("FACE_TO_CENTRE_MM", 25.0, 0, 100, "locate", "f",
         "half a pillar: LiDAR sees the face, the planner wants the centre"),

    # ---------------- LiDAR-only candidates ----------------
    Spec("CAND_MAX_MM", 1800.0, 300, 4000, "cand", "f", ""),
    Spec("CAND_WALL_MM", 70.0, 0, 400, "cand", "f", "keep this clear of a fitted wall"),
    Spec("CAND_GAP_MM", 60.0, 10, 400, "cand", "f", "split clusters at this gap"),
    Spec("CAND_MAX_WIDTH_MM", 120.0, 40, 600, "cand", "f",
         "wider than this is a wall run, not a pillar"),
    Spec("CAND_MIN_POINTS", 2, 1, 30, "cand", "i", ""),

    # ---------------- lane planner ---------------- (was firmware)
    Spec("CORRIDOR_MM", 1000.0, 300, 2000, "plan", "f",
         "wall-to-wall width of a straight"),
    Spec("CAR_HALF_W_MM", 57.0, 20, 200, "plan", "f", ""),
    Spec("PILLAR_HALF_MM", 25.0, 5, 100, "plan", "f", ""),
    Spec("PASS_MARGIN_MM", 80.0, 0, 300, "plan", "f",
         "air gap between the car's side and the pillar's face"),
    Spec("WALL_MARGIN_MM", 45.0, 0, 300, "plan", "f", ""),
    Spec("CENTRE_AIM_MM", 600.0, 100, 2000, "plan", "f",
         "centring: aim at the lane centre this far ahead"),
    Spec("OFF_JUMP_MM", 120.0, 20, 500, "plan", "f",
         "lane offset cannot move this much in one scan unless it persists"),
    Spec("CENTRE_YAW_MAX", 20.0, 2, 60, "plan", "f",
         "deg, plain centring. Deliberately gentle so the car does not weave"),
    Spec("LANE_VALID_MAX_MM", 1100, 300, 3000, "plan", "i",
         "a cone farther than this means no wall was fitted on that side"),
    Spec("PLAN_MAX_AHEAD_MM", 1600.0, 300, 3000, "plan", "f",
         "ignore sightings farther ahead than this"),
    Spec("PILLAR_MAX_LAT_MM", 420.0, 100, 1000, "plan", "f",
         "seats are well inside the corridor; farther out is a pillar of ANOTHER "
         "straight seen across the corner"),
    Spec("CROSS_MAX_LAT_MM", 1300.0, 400, 3000, "plan", "f",
         "a cross-corner sighting beyond this is not the next straight at all"),

    # ---------------- passing a pillar ---------------- (was firmware)
    Spec("PASS_LEAD_MM", 160.0, 0, 600, "pass", "f",
         "be at the pass position this far BEFORE the pillar"),
    Spec("PASS_AIM_MIN_MM", 120.0, 40, 600, "pass", "f",
         "shortest aim distance, i.e. the sharpest swerve"),
    Spec("HOLD_AIM_MM", 250.0, 50, 1000, "pass", "f",
         "alongside a pillar: gentle hold"),
    Spec("PASS_YAW_MAX", 75.0, 5, 89, "pass", "f",
         "deg, while a pillar is in play"),
    Spec("PASS_HOLD_MM", 250.0, 0, 600, "pass", "f",
         "keep a pillar's side until it is this far behind the car centre"),
    Spec("UNK_COMMIT_MM", 400.0, 50, 1500, "pass", "f",
         "an unknown-colour object this close -> dodge to the roomier side"),
    Spec("TURN_RADIUS_MM", 270.0, 100, 800, "pass", "f",
         "full-lock radius (worse side), for the reach estimate"),
    Spec("GIVEUP_LEAD_MM", 60.0, 0, 400, "pass", "f",
         "reach is judged to this far before the pillar"),
    Spec("AVOID_MARGIN_MM", 15.0, 0, 150, "pass", "f",
         "bare miss distance when no correct pass is reachable"),
    Spec("ALLOW_GIVE_UP", True, 0, 1, "pass", "b",
         "correct side unreachable -> pass on the other side rather than hit it "
         "(check your rulebook penalty)"),
    Spec("TRACK_MATCH_MM", 200.0, 50, 600, "pass", "f",
         "same pillar if within this, along and lateral"),
    Spec("TRACK_CONFIRM", 2, 1, 20, "pass", "i",
         "sightings before a pillar steers the car"),
    Spec("TRACK_FORGET_MM", 300.0, 50, 1500, "pass", "f",
         "unconfirmed and not seen for this far -> dropped"),

    # ---------------- wall levelling ---------------- (was firmware)
    Spec("LEVEL_GAIN", 0.05, 0.0, 1.0, "level", "f",
         "fraction of the error removed per LiDAR revolution"),
    Spec("LEVEL_MAX_STEP", 0.3, 0.0, 5.0, "level", "f", "deg per rev, hard cap"),
    Spec("LEVEL_MAX_DIFF", 8.0, 0.0, 45.0, "level", "f",
         "deg - a bigger disagreement is a bad fit, not drift"),
    Spec("LEVEL_MAX_WALLANG", 20.0, 0.0, 45.0, "level", "f",
         "deg - car too yawed for a clean fit"),

    # ---------------- corner trigger ---------------- (was firmware)
    Spec("SIDE_OPEN_MM", 1500, 300, 3000, "turn", "i",
         "turn side above this = the inner wall has ended"),
    Spec("SIDE_OPEN_FRAMES", 3, 1, 50, "turn", "i",
         "consecutive NEW frames before believing it"),
    Spec("SIDE_WALL_FRAMES", 3, 0, 50, "turn", "i",
         "the inner wall must have been seen this many times first"),
    Spec("TURN_TRIGGER_MAX_YAW", 25.0, 5, 90, "turn", "f",
         "only trigger while the car is this close to the lane direction; "
         "mid-swerve the side beam is not sideways and fakes an open corner"),
    Spec("TURN_LOCK_FRACTION", 0.70, 0.2, 1.0, "turn", "f",
         "arc lock as a fraction of full travel, sent in every ARC frame"),
    Spec("TURN_CAP_CM", 150.0, 20, 400, "turn", "f",
         "odometry backstop: leave the arc after this much travel"),
    Spec("TARGET_CORNERS", 12, 1, 48, "turn", "i", ""),
    Spec("FINAL_STRAIGHT_CM", 100.0, 0, 400, "turn", "f", ""),
    Spec("SEARCH_SAFETY_CM", 400.0, 50, 1000, "turn", "f",
         "no turn trigger within this -> restart the straight"),
    Spec("POST_CORNER_LOCKOUT_CM", 50.0, 0, 200, "turn", "f",
         "levelling stays off for this much of a new straight"),

    # ---------------- corner exit shaping ---------------- (was firmware)
    Spec("TURN_DELAY_MM", 0.0, 0, 1000, "corner", "f",
         "minimum run-out after the corner is seen, before the arc starts"),
    Spec("TURN_FRONT_MM", 600.0, 0, 2000, "corner", "f",
         "start the arc when the wall ahead is this close (0 = off)"),
    Spec("TURN_ARM_MAX_MM", 1600.0, 50, 2000, "corner", "f",
         "backstop: arc anyway this far after arming"),
    Spec("CORNER_ARC_ADAPT", True, 0, 1, "corner", "b",
         "shape the arc to the planned exit side"),
    Spec("TURN_FRONT_INNER_MM", 1050.0, 0, 2000, "corner", "f",
         "... start the arc earlier when exiting INNER"),
    Spec("TURN_LOCK_INNER", 1.00, 0.2, 1.0, "corner", "f", "... and tighter"),
    Spec("TURN_FRONT_OUTER_MM", 600.0, 0, 2000, "corner", "f",
         "... run on further before the arc when exiting OUTER. Equal to "
         "TURN_FRONT_MM = the wide arc is off, which is the default: in sim "
         "500 bought an outer-side green ~60 mm of clearance but cost 5 "
         "finished runs in 24 to outer-wall contacts. 550 is worth a try on "
         "the real mat, where the turning radii are not the sim's"),
    Spec("TURN_LOCK_OUTER", 0.70, 0.2, 1.0, "corner", "f", "... and looser"),
    Spec("CORNER_EXIT_MM", 0.0, -400, 400, "corner", "f",
         "exit lane offset when the next straight's colour is NOT known. "
         "0 = centred, which keeps both passing sides reachable"),
    Spec("CORNER_EXIT_BIAS_MM", 200.0, 0, 400, "corner", "f",
         "exit offset when the next straight's colour IS known"),
    Spec("POST_CORNER_YAW_MAX", 45.0, 5, 75, "corner", "f",
         "steering authority for the first POST_CORNER_BOOST_MM of a straight"),
    Spec("POST_CORNER_BOOST_MM", 600.0, 0, 1500, "corner", "f", ""),
    Spec("PRE_CORNER_ZONE_MM", 900.0, 0, 1500, "corner", "f",
         "hold the pre-corner swing once the wall ahead is this close. "
         "0 = pre-corner swing off"),
    Spec("PRE_CORNER_SWING_MM", 250.0, -400, 400, "corner", "f",
         "enter the corner wide to leave it tight: a 90 deg arc throws the "
         "car AWAY from the side it started on, so the way out near the "
         "inner wall is in near the outer one. Held on the side opposite "
         "the planned exit"),
    Spec("CARRY_TRACKS", True, 0, 1, "corner", "b",
         "carry a cross-corner sighting's COLOUR through the arc (not its "
         "coordinates, which the corner invalidates)"),
    Spec("USE_SECONDARY_CORNER", True, 0, 1, "corner", "b",
         "let the SECOND pillar - the next-largest accepted blob, usually "
         "the next straight's first - decide which side to exit on"),
    Spec("SECONDARY_ZONE_MM", 1500.0, 0, 3000, "corner", "f",
         "how recently, in lane distance, the second pillar must have been "
         "seen for its colour to still count at the corner"),

    # ---------------- 3-point corner ---------------- (was firmware)
    Spec("USE_CORNER_MANEUVER", True, 0, 1, "mnv", "b",
         "clockwise only: after an inner-side pass just before the corner, "
         "AND when the next straight needs the inner side too, replace the "
         "arc with a 3-point corner. Gated that way it is a clear win; fired "
         "on every inner pass it costs the outer-side cases 60-300 mm of "
         "clearance for nothing"),
    Spec("MNV_ZONE_MM", 700.0, 0, 2000, "mnv", "f",
         "inner-side pillar within this before the trigger"),
    Spec("MNV_SWING_LAT_MM", 150.0, -400, 400, "mnv", "f",
         "magnitude of the swing, applied to the planned exit side"),
    Spec("MNV_SWING_YAW_MAX", 20.0, 0, 60, "mnv", "f", ""),
    Spec("MNV_DEEP_FRONT_MM", 520, 100, 1500, "mnv", "i",
         "start the forward arc when the wall ahead is this close"),
    Spec("MNV_DEEP_CAP_CM", 90.0, 10, 250, "mnv", "f", "odometry backstop for SWING"),
    Spec("MNV_ARC_FWD_DEG", 55.0, 10, 89, "mnv", "f",
         "forward arc until this much of the 90 is done"),
    Spec("MNV_ARC_STOP_FRONT_MM", 170, 80, 800, "mnv", "i", ""),
    Spec("MNV_STOP_MS", 150, 0, 2000, "mnv", "i", ""),
    Spec("MNV_REV_CAP_CM", 40.0, 5, 120, "mnv", "f", "reverse at most this far per leg"),
    Spec("MNV_DONE_DEG", 6.0, 1, 30, "mnv", "f",
         "facing the new lane within this = done"),
    Spec("MNV_MAX_LEGS", 3, 1, 10, "mnv", "i", ""),
    Spec("MNV_LEG_STEP_DEG", 12.0, 0, 60, "mnv", "f",
         "extra rotation each forward leg must add. Measured from where the leg "
         "STARTED, not from the old lane"),

    # ---------------- run ----------------
    Spec("DRIVE_PWM", 60, 0, 255, "run", "i",
         "one constant PWM for driving, turning and reversing"),
    Spec("TELEM_STALE_S", 0.3, 0.05, 2.0, "run", "f",
         "no TELEM for this long and the FSM is blind -> command STOP"),
    Spec("LIDAR_STALE_S", 0.3, 0.05, 2.0, "run", "f",
         "no LiDAR point for this long -> LIDAR_OK is cleared"),
    Spec("LIDAR_DEAD_S", 1.0, 0.1, 10.0, "run", "f",
         "no LiDAR point for this long -> the corner gate falls back to the "
         "floor colour alone"),
    Spec("COLOR_CONFIRM_S", 0.006, 0.0, 0.5, "run", "f",
         "the floor colour must read the same for this long to arm the gate. "
         "The classification itself is the STM32's; this debounce is ours"),

    # ---------------- link ----------------
    Spec("SEND_HZ", 50, 5, 200, "link", "i", "DRIVE frames per second"),
    Spec("BEARING_TOL_DEG", 2, 0, 15, "link", "i",
         "+/- degrees when picking the 0/90/270 beams"),
    Spec("STM_PUSH_PER_LOOP", 4, 1, 40, "link", "i",
         "tuning lines sent to the STM32 per loop; keep well under the SEND_HZ budget"),
    Spec("CMD_REPEAT", 3, 1, 10, "link", "i", "copies of a REBOOT command"),
]


class PiParams:
    """Lock-free for readers: p['NAME']. Writers swap the whole dict."""

    def __init__(self, specs=PI_SPECS):
        self.specs = {s.name: s for s in specs}
        self.order = [s.name for s in specs]
        self._vals = {s.name: s.default for s in specs}
        self._derived = {}
        self._recompute()

    def __getitem__(self, name):
        return self._vals[name]

    def __contains__(self, name):
        return name in self._vals

    def get(self, name, default=None):
        return self._vals.get(name, default)

    def snapshot(self):
        return self._vals

    # ---- derived values ----
    #
    # The firmware recomputed these on every set so the order parameters
    # arrived in never mattered. Same here.

    def _recompute(self):
        p = self._vals
        pass_clear = p["PILLAR_HALF_MM"] + p["CAR_HALF_W_MM"] + p["PASS_MARGIN_MM"]
        lane_limit = max(0.0, p["CORRIDOR_MM"] / 2.0 - p["CAR_HALF_W_MM"]
                         - p["WALL_MARGIN_MM"])
        avoid_clear = p["PILLAR_HALF_MM"] + p["CAR_HALF_W_MM"] + p["AVOID_MARGIN_MM"]
        self._derived = {"PASS_CLEAR_MM": pass_clear,
                         "LANE_LIMIT_MM": lane_limit,
                         "AVOID_CLEAR_MM": avoid_clear}

    @property
    def derived(self):
        return self._derived

    def set(self, name, value):
        """Returns the stored value. Raises KeyError / ValueError."""
        spec = self.specs[name]
        v = spec.coerce(value)
        new = dict(self._vals)
        new[name] = v
        self._vals = new            # atomic swap; readers never see a half-set
        self._recompute()
        return v

    def set_many(self, mapping):
        new = dict(self._vals)
        out = {}
        for name, value in mapping.items():
            if name not in self.specs:
                continue
            out[name] = new[name] = self.specs[name].coerce(value)
        self._vals = new
        self._recompute()
        return out

    def reset(self):
        self._vals = {n: self.specs[n].default for n in self.order}
        self._recompute()

    def describe(self):
        return [self.specs[n].as_dict(self._vals[n]) for n in self.order]

    # ---- derived views the hot loop wants ready-made ----

    def hsv_config(self):
        p = self._vals
        return {
            "RED": [[[p["RED1_H_LO"], p["RED1_S_LO"], p["RED1_V_LO"]],
                     [p["RED1_H_HI"], p["RED1_S_HI"], p["RED1_V_HI"]]],
                    [[p["RED2_H_LO"], p["RED2_S_LO"], p["RED2_V_LO"]],
                     [p["RED2_H_HI"], p["RED2_S_HI"], p["RED2_V_HI"]]]],
            "GREEN": [[[p["GREEN_H_LO"], p["GREEN_S_LO"], p["GREEN_V_LO"]],
                       [p["GREEN_H_HI"], p["GREEN_S_HI"], p["GREEN_V_HI"]]]],
        }

    def lab_config(self):
        """{'RED': (lo, hi), 'GREEN': (lo, hi)} as (L, a, b) triples."""
        p = self._vals
        return {
            "RED": ((p["RED_L_LO"], p["RED_A_LO"], p["RED_B_LO"]),
                    (p["RED_L_HI"], p["RED_A_HI"], p["RED_B_HI"])),
            "GREEN": ((p["GREEN_L_LO"], p["GREEN_A_LO"], p["GREEN_B_LO"]),
                      (p["GREEN_L_HI"], p["GREEN_A_HI"], p["GREEN_B_HI"])),
        }

    def pillar_filter(self):
        p = self._vals
        return {k: p[k] for k in ("floor_s_max", "floor_v_min", "strip_px",
                                  "floor_below_min", "aspect_min", "solidity_min",
                                  "contrast_s_min", "bottom_margin_px",
                                  "linked_test", "dark_v_max", "ignore_bottom_px")}


# --------------------------------------------------------------------------
# STM32 mirror
# --------------------------------------------------------------------------

class StmParams:
    """Mirror of the firmware's (now small) table, plus the push queue.

    The firmware is the authority on what EXISTS (names, ids, ranges, groups);
    this side is the authority on what the VALUES should be. `desired` is what
    config.json says; `live` is what the firmware last acknowledged. When they
    differ the name is queued for a push.
    """

    def __init__(self):
        self.table = {}          # name -> {id, kind, lo, hi, group}
        self.by_id = {}
        self.live = {}           # name -> value the firmware confirmed
        self.desired = {}        # name -> value we want
        self.version = None
        self.count = None
        self.boot = None
        self.synced = False      # a full dump has been received
        self.pending = queue.Queue()
        self._lock = threading.Lock()

    # ---- incoming firmware lines ----
    def on_line(self, line):
        """Handle one '!' line. Returns a note for the log, or None."""
        if line.startswith("!P "):
            try:
                _, pid, name, kind, val, lo, hi, group = line.split()
            except ValueError:
                return None
            with self._lock:
                self.table[name] = {"id": int(pid), "kind": int(kind),
                                    "lo": float(lo), "hi": float(hi),
                                    "group": int(group)}
                self.by_id[int(pid)] = name
                self.live[name] = float(val)
                if name not in self.desired:
                    self.desired[name] = float(val)   # adopt the firmware default
                if self.count and len(self.table) >= self.count:
                    self.synced = True
            return None

        if line.startswith("!p "):
            try:
                _, pid, val = line.split()
            except ValueError:
                return None
            name = self.by_id.get(int(pid))
            if name:
                with self._lock:
                    self.live[name] = float(val)
            return None

        if line.startswith("!V "):
            try:
                _, ver, count, boot = line.split()
            except ValueError:
                return None
            return self.on_boot(ver, int(count), boot)

        if line.startswith("!E"):
            return "[pi] STM32 rejected a tuning line: " + line
        return None

    def on_boot(self, ver, count, boot):
        """A version report, from a '!V' line or from TELEM's boot_id field.

        TELEM carries boot_id in every frame, so a mid-run reset is noticed
        within one frame instead of waiting for a '?V' round trip.
        """
        boot = str(boot)
        new_boot = (self.boot is not None and boot != self.boot)
        if ver is not None:
            self.version = ver
        if count is not None:
            self.count = count
        self.boot = boot
        if new_boot or not self.table:
            # The firmware restarted (or we have never seen it). Its table is
            # back at compiled-in defaults, so ask for it and re-push.
            with self._lock:
                self.synced = False
                self.table.clear()
                self.by_id.clear()
                self.live.clear()
            self.request_dump()
            return f"[pi] STM32 boot {boot} - re-reading and re-pushing tuning"
        return None

    def ticks_per_mm(self, default=1.4853):
        """The Pi's odometry conversion, owned by the firmware because it is a
        hardware fact. Falls back to the known-good value until the table has
        been read."""
        v = self.live.get("TICKS_PER_CM") or self.desired.get("TICKS_PER_CM")
        return (float(v) / 10.0) if v else default

    # ---- outgoing ----
    def request_dump(self):
        self.pending.put("?P")

    def queue_all(self):
        """Push every desired value that the firmware does not already have."""
        with self._lock:
            names = [n for n in self.table if n in self.desired]
        for n in names:
            self._queue_if_stale(n)

    def _queue_if_stale(self, name):
        want = self.desired.get(name)
        have = self.live.get(name)
        if want is None:
            return
        if have is None or abs(float(have) - float(want)) > 1e-6:
            self.pending.put(f"N {name} {want:.6g}")

    def set(self, name, value):
        """Returns the coerced value. Raises KeyError / ValueError."""
        meta = self.table.get(name)
        if meta is None:
            raise KeyError(name)
        v = float(value)
        if v != v:
            raise ValueError("not a number")
        v = max(meta["lo"], min(meta["hi"], v))
        if meta["kind"] in (1, 2, 3, 4, 5):      # int-ish / bool
            v = float(round(v))
        with self._lock:
            self.desired[name] = v
        self.pending.put(f"N {name} {v:.6g}")
        return v

    def reset(self):
        """Back to the firmware's own defaults: forget desired, re-read."""
        with self._lock:
            self.desired = dict(self.live)
        self.request_dump()

    def drain(self, n):
        """Up to n queued lines, for the serial owner to write. Never blocks."""
        out = []
        for _ in range(n):
            try:
                out.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return out

    def describe(self):
        with self._lock:
            rows = []
            for name, meta in self.table.items():
                rows.append({"name": name, "value": self.desired.get(name, 0),
                             "live": self.live.get(name), "lo": meta["lo"],
                             "hi": meta["hi"],
                             "kind": "b" if meta["kind"] == 5 else
                                     ("f" if meta["kind"] == 0 else "i"),
                             "group": meta["group"], "id": meta["id"]})
        rows.sort(key=lambda r: r["id"])
        return rows


# The firmware's PGroup enum, in order. Much shorter than it was: everything
# about planning and passing is a Pi group now.
STM_GROUPS = ["drive", "turn", "safety", "link"]


# --------------------------------------------------------------------------
# persistence - one file, both halves
# --------------------------------------------------------------------------

# Every path below defaults to None rather than to CONFIG_PATH directly: a
# default argument is bound when the function is DEFINED, so `params.CONFIG_PATH
# = other` would have had no effect and a tool or a test repointing it would
# have silently written to the real file.
def read_config(path=None):
    """config.json as a dict, or {} if it is missing or unreadable."""
    path = path or CONFIG_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(pi: PiParams, stm: StmParams, path=None):
    """Fill both halves from config.json. Returns a note for the log."""
    path = path or CONFIG_PATH
    cfg = read_config(path)
    if not cfg:
        return "[pi] no config.json - using defaults"
    n_pi = len(pi.set_many(cfg.get("params", {})))
    stm_saved = cfg.get("stm32", {})
    stm.desired.update({k: float(v) for k, v in stm_saved.items()})
    return f"[pi] loaded config: {n_pi} Pi, {len(stm_saved)} STM32"


def save(pi: PiParams, stm: StmParams, path=None):
    """Atomic, and non-destructive: the structured blocks this module does not
    own (camera_intrinsics, lidar, serial) are read back and written out
    unchanged, so saving tuning can never drop your calibration."""
    path = path or CONFIG_PATH
    cfg = read_config(path)
    cfg["params"] = dict(pi.snapshot())
    cfg["stm32"] = dict(stm.desired)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return f"[pi] saved to {os.path.basename(path)}"


# --------------------------------------------------------------------------
# migration from the old split config
# --------------------------------------------------------------------------

def migrate(path=None, tuning=None):
    """Fold an old tuning.json, and the pre-merge config.json blocks, into the
    one-file layout. Safe to run twice."""
    path = path or CONFIG_PATH
    tuning = tuning or LEGACY_TUNING_PATH
    cfg = read_config(path)
    pi, stm = PiParams(), StmParams()
    vals = dict(cfg.get("params", {}))

    # old tuning.json
    try:
        with open(tuning) as f:
            old = json.load(f)
        vals.update(old.get("pi", {}))
        cfg.setdefault("stm32", {}).update(old.get("stm32", {}))
    except (OSError, ValueError):
        pass

    # old config.json blocks -> flat names
    if "hsv" in cfg:
        h = cfg["hsv"]
        if len(h.get("RED", [])) == 2:
            (r1lo, r1hi), (r2lo, r2hi) = h["RED"]
            for pre, tri in (("RED1_", r1lo), ("RED2_", r2lo)):
                vals[pre + "H_LO"], vals[pre + "S_LO"], vals[pre + "V_LO"] = tri
            for pre, tri in (("RED1_", r1hi), ("RED2_", r2hi)):
                vals[pre + "H_HI"], vals[pre + "S_HI"], vals[pre + "V_HI"] = tri
        if len(h.get("GREEN", [])) == 1:
            glo, ghi = h["GREEN"][0]
            vals["GREEN_H_LO"], vals["GREEN_S_LO"], vals["GREEN_V_LO"] = glo
            vals["GREEN_H_HI"], vals["GREEN_S_HI"], vals["GREEN_V_HI"] = ghi
    fl = cfg.get("floor", {})
    for src, dst in (("white_s_max", "floor_s_max"), ("white_v_min", "floor_v_min"),
                     ("strip_px", "strip_px"), ("min_white_frac", "floor_below_min"),
                     ("linked", "linked_test"), ("dark_v_max", "dark_v_max"),
                     ("ignore_bottom_px", "ignore_bottom_px")):
        if src in fl:
            vals[dst] = fl[src]
    for src, dst in (("min_blob_area", "MIN_AREA_PROC"), ("hfov_deg", "HFOV_DEG"),
                     ("swap_rb", "SWAP_RB"), ("camera_offset_deg", "CAMERA_OFFSET_DEG")):
        if src in cfg:
            vals[dst] = cfg[src]
    if "bearing_tol_deg" in cfg.get("lidar", {}):
        vals["BEARING_TOL_DEG"] = cfg["lidar"]["bearing_tol_deg"]

    pi.set_many(vals)
    cfg["params"] = dict(pi.snapshot())
    # blocks that are gone: the Pi no longer fuses by bearing lookup, and the
    # tangent solver they belonged to has been deleted.
    for dead in ("hsv", "floor", "fusion", "avoid", "min_blob_area", "hfov_deg",
                 "swap_rb", "camera_offset_deg"):
        cfg.pop(dead, None)
    cfg.get("lidar", {}).pop("bearing_tol_deg", None)

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return f"migrated {len(vals)} values into {os.path.basename(path)}"


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv:
        print(migrate())
    else:
        print(f"{len(PI_SPECS)} Pi parameters in "
              f"{len({s.group for s in PI_SPECS})} groups")
        for g in dict.fromkeys(s.group for s in PI_SPECS):
            names = [s.name for s in PI_SPECS if s.group == g]
            print(f"  {g:8s} {len(names):3d}  {', '.join(names[:4])}...")
