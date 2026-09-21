"""
fsm.py - the obstacle round's state machine. The whole run lives here.

Ported from ObstacleRound.cpp, which owned this until the FSM moved to the
Pi. The shape is unchanged - same seven states, same triggers, same corner
measurements - because it is what actually drives the car. What changed is
where it runs and therefore what it can see: the planner's tracks, the wall
fits and the camera are all local now instead of arriving as sixteen numbers
on a serial line.

    WAIT_START        armed, motor off, waiting for Start on the page
    DRIVE_TO_CORNER   hold the lane heading + the planner's yaw; watch for
                      the corner, the colour gate, the finish
    TURNING           eased 90 deg arc, run by the STM32, terminated on IMU
                      heading or the odometry backstop
    CORNER_MANEUVER   the 3-point corner, after an inner-side pass
    FINAL_STRAIGHT    drive L - A from the end of the last corner
    RECOVER           the STM32's wall-panic reflex owns the car; we wait
    FINISHED          stopped

WHAT STAYED ON THE STM32, AND WHY
The heading PID runs at IMU rate with a 2.5 deg/update servo slew, the arc's
inner loop terminates on an IMU reading, and the wall-panic reflex has to be
faster than the crash it prevents. None of those survive a 50 Hz link, so
none of them moved. This file names a heading and a speed; the STM32 holds
them. See docs/PI_STM32_PROTOCOL.md.

ODOMETRY
The firmware used to call zeroEncoder() at each corner and measure with
absEnc(readEncoder()). With the FSM up here that would be a race - we would
ask for a zero and not know which frames were taken before it landed - so
`odo` free-runs on the STM32 and this file keeps its own baselines. Every
"distance since X" below is a subtraction, and `_since_corner_*` is the
direct replacement for the firmware's absEnc(readEncoder()).
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from worldstate import clamp, wrap180

from .intent import ActionIntent, SteerMode
from .planner import LanePlanner, PlanInput

# floor colours, as the STM32 reports them
COLOUR_NONE, COLOUR_ORANGE, COLOUR_BLUE = 0, 1, 2
COLOUR_NAMES = {COLOUR_NONE: "none", COLOUR_ORANGE: "orange", COLOUR_BLUE: "blue"}

RED, GREEN = "RED", "GREEN"


class State(Enum):
    WAIT_START = auto()
    DRIVE_TO_CORNER = auto()
    TURNING = auto()
    CORNER_MANEUVER = auto()
    FINAL_STRAIGHT = auto()
    RECOVER = auto()
    FINISHED = auto()


class MnvPhase(Enum):
    SWING = auto()
    ARC_FWD = auto()
    STOP = auto()
    ARC_REV = auto()


@dataclass
class Ctx:
    """Everything the FSM can see this tick. Built fresh by the run loop."""
    p: object                          # PiParams
    now: float = 0.0
    telem: object = None               # control.link.Telemetry
    telem_fresh: bool = False

    # perception
    lidar_live: bool = False
    lidar_dead: bool = True
    new_frame: bool = False
    new_rev: bool = False
    front_mm: float = float("inf")
    left_mm: float = float("inf")
    right_mm: float = float("inf")
    cone_left: float = None
    cone_right: float = None
    wall_ang: float = None
    pillar_xy: tuple = None
    pillar_colour: str = None
    sec_xy: tuple = None               # the SECOND pillar, located
    sec_colour: str = None             # ... and its colour
    unknown_xy: tuple = None

    ticks_per_mm: float = 1.4853

    @property
    def heading(self):
        return self.telem.heading_deg if self.telem else 0.0

    @property
    def odo(self):
        return self.telem.odo_ticks if self.telem else 0

    @property
    def floor(self):
        return self.telem.floor if self.telem else COLOUR_NONE

    @property
    def recovering(self):
        return bool(self.telem and self.telem.recovering)

    @property
    def arc_done(self):
        return bool(self.telem and self.telem.arc_done)


class FSM:
    """One instance per run session. step() once per tick."""

    def __init__(self, params, emit=print):
        self.p = params
        self.emit = emit
        self.planner = LanePlanner()

        self.state = State.WAIT_START
        self.entered = False
        self.intent = ActionIntent.stop("boot")

        self.lane_heading = 0.0
        self.target_heading = 0.0
        self.locked_colour = COLOUR_NONE
        self.last_first_colour = COLOUR_NONE
        self.clockwise = True
        self.corner_count = 0

        # the run's distance bookkeeping (firmware: zeroEncoder + absEnc)
        self._corner_odo = 0            # odo at the last corner / at GO
        self._dc_base = 0               # odo at DRIVE entry
        self._dc_arm_odo = 0
        self._turn_start_odo = 0
        self._mnv_base_odo = 0

        # the final-straight geometry
        self.first_segment_cm = 0.0
        self.full_start_straight_cm = 0.0
        self.have_full_straight = False
        self.final_distance_cm = 0.0

        # corner trigger
        self.side_open_count = 0
        self.side_wall_count = 0
        self.dc_colour_armed = False
        self.dc_want_colour = COLOUR_NONE
        self.dc_turn_armed = False
        self.dc_turn_maneuver = False

        # the arc actually commanded this corner
        self.turn_front_cmd = 0.0
        self.turn_lock_cmd = 0.70
        self.turn_target = 0.0
        self.turn_amount = 0.0

        # 3-point corner
        self.mnv_phase = MnvPhase.SWING
        self.mnv_new_lane = 0.0
        self.mnv_legs = 0
        self.mnv_leg_start_deg = 0.0
        self.mnv_t0 = 0.0

        # floor-colour debounce (the classification itself is the STM32's)
        self._pending_colour = COLOUR_NONE
        self._pending_since = 0.0
        self.colour_muted = False
        self._colour_mute_from = 0

        # the STM32's recovery reflex, observed
        self._was_recovering = False
        self._recover_return = (State.DRIVE_TO_CORNER, False)

        self.level_enabled = False
        self.planner_enabled = False
        self.start_requested = False
        self.stop_requested = False
        self.note = ""

    # ------------------------------------------------------------ requests

    def request_start(self):
        self.start_requested = True

    def request_stop(self):
        self.stop_requested = True

    # ------------------------------------------------------------- helpers

    def _go(self, state):
        if state is not self.state:
            self.emit(f"[fsm] {self.state.name} -> {state.name}")
        self.state = state
        self.entered = False

    def _since_corner_mm(self, c: Ctx):
        """The firmware's absEnc(readEncoder()) - distance since the last
        corner, which is where it called zeroEncoder()."""
        return abs(c.odo - self._corner_odo) / max(1e-6, c.ticks_per_mm)

    def _since_corner_cm(self, c: Ctx):
        return self._since_corner_mm(c) / 10.0

    def _mm_since(self, c: Ctx, base):
        return abs(c.odo - base) / max(1e-6, c.ticks_per_mm)

    def turn_is_clockwise(self):
        """Which way the NEXT corner goes. Before the direction locks this is
        the floor colour's guess, not the (default) locked value."""
        if self.locked_colour == COLOUR_NONE:
            return self.last_first_colour == COLOUR_ORANGE
        return self.clockwise

    def direction_known(self):
        """Has any floor colour been read yet?

        Until one has, turn_is_clockwise() returns a guess - anticlockwise.
        Anything that uses the turn direction to REJECT evidence has to know
        that, or on the first straight it throws away everything on one side
        for no reason at all.
        """
        return (self.locked_colour != COLOUR_NONE
                or self.last_first_colour != COLOUR_NONE)

    def turn_side_mm(self, c: Ctx):
        return c.right_mm if self.turn_is_clockwise() else c.left_mm

    def pillar_seen(self, c: Ctx):
        return c.lidar_live and c.pillar_colour is not None

    # ---- floor colour, debounced ----

    def _reset_colour(self):
        self._pending_colour = COLOUR_NONE
        self._pending_since = 0.0

    def _detect_colour(self, c: Ctx, want):
        if self.colour_muted:
            self._reset_colour()
            return COLOUR_NONE
        raw = c.floor
        if want != COLOUR_NONE and raw != want:
            raw = COLOUR_NONE
        if raw == COLOUR_NONE:
            self._reset_colour()
            return COLOUR_NONE
        if raw != self._pending_colour:
            self._pending_colour = raw
            self._pending_since = c.now
            return COLOUR_NONE
        if c.now - self._pending_since >= c.p["COLOR_CONFIRM_S"]:
            self._reset_colour()
            return raw
        return COLOUR_NONE

    # ------------------------------------------------------------ the tick

    def step(self, c: Ctx) -> ActionIntent:
        p = c.p

        # ---- the planner runs before the states, as it did in the firmware
        self.planner.update(PlanInput(
            p=p, odo_ticks=c.odo, ticks_per_mm=c.ticks_per_mm,
            heading=c.heading, lane_heading=self.lane_heading,
            new_frame=c.new_frame, new_rev=c.new_rev,
            lidar_stale=not c.lidar_live, front_mm=c.front_mm,
            cone_left=c.cone_left, cone_right=c.cone_right,
            wall_ang=c.wall_ang, pillar_xy=c.pillar_xy,
            pillar_colour=c.pillar_colour, unknown_xy=c.unknown_xy,
            sec_xy=c.sec_xy, sec_colour=c.sec_colour,
            direction_known=self.direction_known(),
            clockwise=self.clockwise, turn_clockwise=self.turn_is_clockwise(),
            corner_count=self.corner_count, enabled=self.planner_enabled))
        if self.planner.note:
            self.emit("[plan] " + self.planner.note)

        if self.level_enabled:
            step = self.planner.level_step(PlanInput(
                p=p, heading=c.heading, lane_heading=self.lane_heading,
                new_rev=c.new_rev, lidar_stale=not c.lidar_live,
                cone_left=c.cone_left, cone_right=c.cone_right,
                wall_ang=c.wall_ang))
            if step:
                self.lane_heading = wrap180(self.lane_heading + step)
                self.target_heading = self.lane_heading

        # ---- blind is a stop. Without TELEM there is no heading and no
        # odometry, so there is nothing useful the FSM can decide.
        if not c.telem_fresh and self.state not in (State.WAIT_START,
                                                    State.FINISHED):
            self.planner_enabled = False
            self.level_enabled = False
            self.intent = ActionIntent.stop("no TELEM - STM32 not reporting")
            return self.intent

        # ---- Stop from the page works in every state
        if self.stop_requested:
            self.stop_requested = False
            if self.state is not State.FINISHED:
                self.emit("[fsm] STOP from the page")
                self._go(State.FINISHED)

        # ---- the colour mute the recovery reflex set
        if self.colour_muted and not c.recovering and c.odo >= self._colour_mute_from:
            self.colour_muted = False
            self.emit("[fsm] colour re-enabled")

        # ---- the STM32's wall-panic reflex, observed
        self._watch_recovery(c)

        handler = {
            State.WAIT_START: self._wait_start,
            State.DRIVE_TO_CORNER: self._drive,
            State.TURNING: self._turning,
            State.CORNER_MANEUVER: self._maneuver,
            State.FINAL_STRAIGHT: self._final_straight,
            State.RECOVER: self._recover,
            State.FINISHED: self._finished,
        }[self.state]
        self.intent = handler(c)
        return self.intent

    # ----------------------------------------------------- recovery overlay

    def _watch_recovery(self, c: Ctx):
        """The reflex is the STM32's, but its consequences are ours: while it
        owns the car the planner and the levelling must be off, and the floor
        colour must be muted until the car has driven back past where it
        started reversing (the firmware's colorMuteFrom)."""
        if c.recovering and not self._was_recovering:
            if self.state not in (State.RECOVER, State.WAIT_START,
                                  State.FINISHED):
                self._recover_return = (self.state, self.entered)
                self.level_enabled = False
                self.planner_enabled = False
                self.colour_muted = True
                self._colour_mute_from = c.odo
                self.emit(f"[fsm] STM32 RECOVER, front={c.front_mm:.0f}")
                self._go(State.RECOVER)
        elif self._was_recovering and not c.recovering:
            if self.state is State.RECOVER:
                if c.telem and c.telem.recover_capped:
                    self.emit(f"[fsm] recover capped, try "
                              f"{c.telem.recover_tries}")
                else:
                    self.emit("[fsm] recover clear")
                self.state, self.entered = self._recover_return
        self._was_recovering = c.recovering

    def _recover(self, c: Ctx):
        # The firmware is reversing; mode and steering are ignored while
        # RECOVERING is set. Keep naming the heading we want so the moment it
        # hands back, the car is already pointing the right way.
        self.entered = True
        return ActionIntent.hold(self.target_heading, c.p["DRIVE_PWM"],
                                 "STM32 recovery reflex")

    # ------------------------------------------------------------ WAIT_START

    def _wait_start(self, c: Ctx):
        if not self.entered:
            self.entered = True
            self.start_requested = False   # a Start before arming is ignored
            self.emit("[fsm] WAIT_START armed - press Start on the page")

        if not self.start_requested:
            return ActionIntent.stop("armed - waiting for Start")
        self.start_requested = False

        if not c.lidar_live:
            self.emit("[fsm] START refused: lidar stale")
            return ActionIntent.stop("START refused - lidar stale")
        if not c.telem_fresh:
            self.emit("[fsm] START refused: no TELEM from the STM32")
            return ActionIntent.stop("START refused - STM32 not reporting")

        self.locked_colour = COLOUR_NONE
        self.last_first_colour = COLOUR_NONE
        self.clockwise = True
        self.corner_count = 0
        self.colour_muted = False
        self.first_segment_cm = 0.0
        self.full_start_straight_cm = 0.0
        self.have_full_straight = False
        self.final_distance_cm = c.p["FINAL_STRAIGHT_CM"]
        self.lane_heading = c.heading
        self.target_heading = self.lane_heading
        self.level_enabled = False          # DRIVE turns it on
        self.planner.level_total = 0.0
        self.planner.corner_exit_cmd = 0.0
        self.planner.clear_tracks()
        self.planner.reset_lane_along(c.odo)
        self._corner_odo = c.odo            # the firmware's zeroEncoder()
        self.emit("[fsm] GO")
        self._go(State.DRIVE_TO_CORNER)
        return ActionIntent.hold(self.target_heading, c.p["DRIVE_PWM"], "GO")

    # ------------------------------------------------------------- FINISHED

    def _finished(self, c: Ctx):
        if not self.entered:
            self.entered = True
            self.emit("[fsm] FINISHED")
            self.level_enabled = False
            self.planner_enabled = False
            self.start_requested = False     # only a NEW press restarts
        # Start again re-arms and runs a fresh three laps. Put the car back in
        # the start section first.
        if self.start_requested:
            self._go(State.WAIT_START)
            self.entered = False
            out = self._wait_start(c)        # arm ...
            self.start_requested = True      # ... then honour this press
            return out
        return ActionIntent.stop("finished")

    # -------------------------------------------------------- DRIVE_TO_CORNER

    def _drive(self, c: Ctx):
        p = c.p
        if not self.entered:
            self.entered = True
            self.dc_want_colour = self.locked_colour
            self.dc_colour_armed = False
            self.dc_turn_armed = False
            self.dc_turn_maneuver = False
            self.side_open_count = 0
            self.side_wall_count = 0
            self._reset_colour()
            self._dc_base = c.odo
            self.emit("[fsm] DRIVE")

        # Safety: no corner within SEARCH_SAFETY_CM means something is wrong
        # with the trigger. Restart the straight rather than drive on forever.
        if self._mm_since(c, self._dc_base) >= p["SEARCH_SAFETY_CM"] * 10.0:
            self.emit("[fsm] WARN no turn trigger within the safety distance, "
                      "retrying")
            self.entered = False
            return ActionIntent.hold(self.target_heading, p["DRIVE_PWM"],
                                     "safety distance - restarting straight")

        # The planner is live from the first tick after the arc, lockout
        # included: a pillar right after the corner is handled immediately.
        self.planner_enabled = True
        drive = ActionIntent.hold(
            wrap180(self.target_heading + self.planner.lat_yaw_cmd),
            p["DRIVE_PWM"], self._drive_reason())

        since_corner = self._since_corner_mm(c)
        lockout_mm = (p["POST_CORNER_LOCKOUT_CM"] * 10.0
                      if self.corner_count > 0 else 0.0)
        if since_corner <= lockout_mm:
            self.level_enabled = False
            return drive
        self.level_enabled = True

        direction_locked = self.locked_colour != COLOUR_NONE

        # ---- 1. wall evidence ----
        # This MUST run before the colour gate's early return. side_wall_count
        # is the "I have seen the inner wall on this straight" proof that stops
        # a false corner; on the FIRST corner the colour gate does not arm
        # until the car crosses the orange line, which on this mat is already
        # past the end of the inner wall. Counting only after the gate
        # therefore never saw the wall, side_open_count could never rise, and
        # the turn never fired.
        side_now = self.turn_side_mm(c)
        aligned = abs(wrap180(c.heading - self.lane_heading)) \
            < p["TURN_TRIGGER_MAX_YAW"]
        if not c.lidar_live or not aligned:
            self.side_open_count = 0
        elif c.new_frame:
            if side_now > p["SIDE_OPEN_MM"]:
                if self.side_wall_count >= p["SIDE_WALL_FRAMES"] \
                        and self.side_open_count < 250:
                    self.side_open_count += 1
            else:
                self.side_open_count = 0
                if self.side_wall_count < 250:
                    self.side_wall_count += 1

        # ---- 2. colour gate: first corner, or the lidar-dead fallback ----
        if not self.dc_colour_armed and (not direction_locked or c.lidar_dead):
            col = self._detect_colour(c, self.dc_want_colour)
            if col != COLOUR_NONE:
                self.dc_colour_armed = True
                if not direction_locked:
                    self.last_first_colour = col
                self.side_open_count = 0
                self._reset_colour()
                self.emit(f"[fsm] gate {COLOUR_NAMES[col].upper()} "
                          f"side={self.turn_side_mm(c):.0f}")
        if not direction_locked and not self.dc_colour_armed:
            return drive

        # ---- 3. arm the turn ----
        # The corner measurements are taken at the ARM point, because that is
        # where the corner really is. The delay below only moves where the arc
        # STARTS, which is what sets the exit lane position.
        side_open = self.side_open_count >= p["SIDE_OPEN_FRAMES"]
        if not self.dc_turn_armed and (side_open
                                       or (c.lidar_dead and self.dc_colour_armed)):
            self._arm_turn(c, side_open, side_now, direction_locked)

        if not self.dc_turn_armed:
            return drive

        # ---- 4. when to actually start the arc ----
        # The turn side opening says the corner is HERE; it does not say the
        # car is deep enough into the corner square to arc. Firing on the
        # opening alone starts the 90 deg arc at the mouth of the square and
        # lands the car hard on the inner wall of the next straight. The
        # physical cue is the wall AHEAD, about one turn radius away.
        since_arm = self._mm_since(c, self._dc_arm_odo)
        if since_arm < p["TURN_DELAY_MM"]:
            return drive
        if self.turn_front_cmd > 0.0 and since_arm < p["TURN_ARM_MAX_MM"] \
                and (not c.lidar_live or c.front_mm > self.turn_front_cmd):
            return drive

        if self.dc_turn_maneuver:
            self.emit("[fsm] inner-side pillar before the corner -> 3-point corner")
            self._go(State.CORNER_MANEUVER)
        else:
            self._go(State.TURNING)
        return drive

    def _drive_reason(self):
        pl = self.planner
        if pl.pass_active:
            return f"pass to {pl.lat_target:+.0f} (at {pl.lane_off:+.0f})"
        return f"centre {pl.lat_target:+.0f} (at {pl.lane_off:+.0f})"

    def _arm_turn(self, c: Ctx, side_open, side_now, direction_locked):
        p = c.p
        self.dc_turn_armed = True
        self._dc_arm_odo = c.odo
        if side_open:
            self.emit(f"[fsm] turn: side open {side_now:.0f}")
        else:
            self.emit("[fsm] turn: colour only (lidar dead)")

        if not direction_locked:
            self.locked_colour = self.last_first_colour
            self.clockwise = self.locked_colour == COLOUR_ORANGE
            self.emit("[fsm] LOCKED CW (orange)" if self.clockwise
                      else "[fsm] LOCKED CCW (blue)")

        seg_cm = self._since_corner_cm(c)
        if self.corner_count == 0:
            self.first_segment_cm = seg_cm
            self.emit(f"[fsm] A={self.first_segment_cm:.1f}")
        elif self.corner_count == 4:
            self.full_start_straight_cm = seg_cm
            self.have_full_straight = True
        elif self.corner_count == 8 and self.have_full_straight:
            self.full_start_straight_cm = 0.5 * (self.full_start_straight_cm
                                                 + seg_cm)
        if self.have_full_straight:
            self.final_distance_cm = max(
                0.0, self.full_start_straight_cm - self.first_segment_cm)
            self.emit(f"[fsm] L={self.full_start_straight_cm:.1f} "
                      f"final={self.final_distance_cm:.1f}")

        self._plan_corner_exit(c)       # needs next_straight, still valid here
        self.dc_turn_maneuver = self.planner.inner_pillar_near_corner(
            p, self.clockwise)

    # ------------------------------------------------------ corner exit plan

    def _plan_corner_exit(self, c: Ctx):
        """Where the arc should leave the car, and the arc that gets it there.

        Measured in the sim (bare lap, exit offset just after each corner,
        + = outer, lane limit +/-398):

            TURN_FRONT_MM  0.55   0.70   0.85   1.00   <- TURN_LOCK_FRACTION
                  500      +382   +339   +282   +241
                  600      +335   +241   +190   +150
                  750      +194   +108    +51    -27
                  900       +48    -41   -109   -160

        So the exit is commandable across the whole corridor: firing the arc
        while the wall ahead is still far, and tightening it, walks the car
        from the outer wall to the inner one. Nothing the car does afterwards
        has that authority - a pillar 200 mm past the corner leaves no room to
        cross - which is why this is decided BEFORE the arc, not after.
        """
        p = c.p
        pl = self.planner
        outer = 1.0 if self.clockwise else -1.0   # + lane offset = left of centre

        # Which colour is waiting on the next straight? The SECOND pillar is
        # direct evidence and wins; the cross-corner sighting is the fallback
        # for a frame in which only one pillar was visible at all.
        want = pl.planned_next_colour(p)
        src = ("2nd pillar" if pl.secondary_is_live(p) else "across the corner")

        inner = False
        if want is not None:
            # The passing rule is about the LANE, not the lap: red is passed on
            # its right, which is the right-hand side of the lane (negative
            # offset), whichever way round the car is going. Whether that side
            # is the inner or the outer one then depends on the direction.
            # Tying red to "inner" is right for CW and exactly backwards for
            # CCW.
            sign = -1.0 if want == RED else 1.0
            pl.corner_exit_cmd = sign * p["CORNER_EXIT_BIAS_MM"]
            inner = (pl.corner_exit_cmd > 0.0) != (outer > 0.0)
            shape = ("SHORT corner, exit INNER" if inner
                     else "WIDE corner, exit OUTER")
            self.emit(f"[fsm] next straight {want} ({src}) -> {shape}")
        else:
            pl.corner_exit_cmd = outer * p["CORNER_EXIT_MM"]

        # Three shapes, not two:
        #   INNER    turn early and tight - the short, near corner
        #   OUTER    run on, then turn on a looser arc - the wide corner
        #   unknown  the middle setting, which exits near the lane centre and
        #            leaves both passing sides reachable
        if not p["CORNER_ARC_ADAPT"] or want is None:
            self.turn_front_cmd = p["TURN_FRONT_MM"]
            self.turn_lock_cmd = p["TURN_LOCK_FRACTION"]
        elif inner:
            self.turn_front_cmd = p["TURN_FRONT_INNER_MM"]   # larger = earlier
            self.turn_lock_cmd = p["TURN_LOCK_INNER"]
        else:
            self.turn_front_cmd = p["TURN_FRONT_OUTER_MM"]   # smaller = runs on
            self.turn_lock_cmd = p["TURN_LOCK_OUTER"]

    # -------------------------------------------------------------- TURNING

    def _turning(self, c: Ctx):
        p = c.p
        if not self.entered:
            self.entered = True
            self.level_enabled = False
            self.planner_enabled = False
            self.planner.clear_tracks()
            self.emit(f"[fsm] level {self.planner.level_total:+.2f}")
            self.planner.level_total = 0.0
            self.corner_count += 1
            self.emit(f"[fsm] TURN {self.corner_count}/{p['TARGET_CORNERS']}")
            self.turn_amount = 90.0 if self.clockwise else -90.0
            self.turn_target = wrap180(self.lane_heading - self.turn_amount)
            self._turn_start_odo = c.odo

        capped = self._mm_since(c, self._turn_start_odo) >= p["TURN_CAP_CM"] * 10.0
        if capped or c.arc_done:
            self.lane_heading = wrap180(self.lane_heading - self.turn_amount)
            self._corner_odo = c.odo               # the firmware's zeroEncoder()
            self.planner.reset_lane_along(c.odo)
            self.planner.clear_tracks()   # anything seen mid-turn was in the
                                          # old lane frame
            self.emit(f"[fsm] lane heading {self.lane_heading:+.1f}"
                      f"{' (odometry cap)' if capped and not c.arc_done else ''}")
            return self._finish_corner(c)

        return ActionIntent.arc(self.turn_target, p["DRIVE_PWM"],
                                self.turn_lock_cmd,
                                f"arc to {self.turn_target:+.0f}")

    def _finish_corner(self, c: Ctx):
        self.target_heading = self.lane_heading
        if self.corner_count >= c.p["TARGET_CORNERS"]:
            self._go(State.FINAL_STRAIGHT)
        else:
            self._go(State.DRIVE_TO_CORNER)
        return ActionIntent.hold(self.target_heading, c.p["DRIVE_PWM"],
                                 "corner done")

    # ------------------------------------------------------ CORNER_MANEUVER

    def _maneuver(self, c: Ctx):
        """The 3-point corner. CLOCKWISE ONLY.

        A RED pillar passed on its right - the inner side - just before the
        corner leaves the car hugging the inner wall. A normal arc from there
        lands it on the inner side of the next straight, facing a pillar it
        has not seen yet with no room to cross. Instead:

            SWING    drive deeper into the corner, moving toward the planned
                     exit side
            ARC_FWD  full lock toward the turn until MNV_ARC_FWD_DEG of the 90
                     is done, or the wall ahead gets close
            STOP     settle, and swing the wheels the other way
            ARC_REV  reverse at the OPPOSITE full lock - which keeps rotating
                     the car the SAME way - until it faces the new lane

        It ends high in the corner square facing down the next straight, where
        the camera sees that straight's first pillar early and the car has room
        to go either side of it.
        """
        p = c.p
        # Full lock toward the corner, and the opposite lock for reversing.
        # In road-wheel terms: + = left. Clockwise turns right.
        lock_turn = -1.0 if self.clockwise else 1.0

        if not self.entered:
            self.entered = True
            self.level_enabled = False
            self.planner_enabled = False
            self.planner.clear_tracks()
            self.emit(f"[fsm] level {self.planner.level_total:+.2f}")
            self.planner.level_total = 0.0
            self.corner_count += 1
            self.emit(f"[fsm] TURN {self.corner_count}/{p['TARGET_CORNERS']} 3-POINT")
            self.turn_amount = 90.0 if self.clockwise else -90.0
            self.mnv_new_lane = wrap180(self.lane_heading - self.turn_amount)
            self.mnv_phase = MnvPhase.SWING
            self._mnv_base_odo = c.odo
            self.mnv_legs = 0
            self.mnv_leg_start_deg = 0.0

        from .mapper import STEER_LOCK_DEG
        turned = self._mnv_turned(c)

        if self.mnv_phase is MnvPhase.SWING:
            # Swing toward the side the corner is meant to come out on.
            # MNV_SWING_LAT_MM is a magnitude applied to the planned exit side.
            pl = self.planner
            side = 1.0 if pl.corner_exit_cmd >= 0.0 else -1.0
            if pl.corner_exit_cmd == 0.0:
                side = 1.0 if self.clockwise else -1.0
            target = side * abs(p["MNV_SWING_LAT_MM"])
            yaw = 0.0
            if pl.lane_off_ok:
                import math
                aim = max(c.front_mm - p["MNV_DEEP_FRONT_MM"], 150.0)
                yaw = clamp(math.degrees(math.atan2(target - pl.lane_off, aim)),
                            -p["MNV_SWING_YAW_MAX"], p["MNV_SWING_YAW_MAX"])
            deep = ((c.lidar_live and c.front_mm <= p["MNV_DEEP_FRONT_MM"])
                    or self._mm_since(c, self._mnv_base_odo)
                    >= p["MNV_DEEP_CAP_CM"] * 10.0)
            if deep:
                self.emit(f"[fsm] 3pt arc, front={c.front_mm:.0f}")
                self.mnv_phase = MnvPhase.ARC_FWD
            return ActionIntent.hold(wrap180(self.lane_heading + yaw),
                                     p["DRIVE_PWM"], "3pt swing")

        if self.mnv_phase is MnvPhase.ARC_FWD:
            if abs(wrap180(self.mnv_new_lane - c.heading)) < p["MNV_DONE_DEG"]:
                return self._mnv_finish(c)
            # Each forward leg must add MNV_LEG_STEP_DEG of NEW rotation.
            # Measuring against the total turned since the old lane does not
            # work: reversing at the opposite lock keeps rotating the car the
            # SAME way, so by the time a reverse leg ends the total already
            # exceeds the next threshold and every later forward leg ends on
            # its own first pass without moving.
            want = p["MNV_ARC_FWD_DEG"] if self.mnv_legs == 0 else p["MNV_LEG_STEP_DEG"]
            if turned - self.mnv_leg_start_deg >= want \
                    or (c.lidar_live and c.front_mm <= p["MNV_ARC_STOP_FRONT_MM"]):
                self.mnv_t0 = c.now
                self.mnv_phase = MnvPhase.STOP
                return ActionIntent.stop("3pt settle")
            return ActionIntent.direct(lock_turn * STEER_LOCK_DEG,
                                       p["DRIVE_PWM"], reason="3pt forward leg")

        if self.mnv_phase is MnvPhase.STOP:
            if c.now - self.mnv_t0 >= p["MNV_STOP_MS"] / 1000.0:
                self._mnv_base_odo = c.odo
                self.mnv_leg_start_deg = turned
                self.mnv_legs += 1
                self.emit(f"[fsm] 3pt reverse, turned {turned:.1f}")
                self.mnv_phase = MnvPhase.ARC_REV
            # steering swings the other way while stopped, ready to reverse
            return ActionIntent.direct(-lock_turn * STEER_LOCK_DEG, 0,
                                       reason="3pt settle")

        # ARC_REV
        left = wrap180(self.mnv_new_lane - c.heading)
        if abs(left) < p["MNV_DONE_DEG"] or turned > 90.0:
            return self._mnv_finish(c)
        if self._mm_since(c, self._mnv_base_odo) >= p["MNV_REV_CAP_CM"] * 10.0:
            if self.mnv_legs >= p["MNV_MAX_LEGS"]:
                return self._mnv_finish(c)       # good enough: DRIVE squares up
            self._mnv_base_odo = c.odo
            self.mnv_leg_start_deg = turned
            self.mnv_phase = MnvPhase.ARC_FWD
            return ActionIntent.stop("3pt leg end")
        return ActionIntent.direct(-lock_turn * STEER_LOCK_DEG, p["DRIVE_PWM"],
                                   reverse=True, reason="3pt reverse leg")

    def _mnv_turned(self, c: Ctx):
        """How far the car has rotated TOWARD the turn since the old lane."""
        d = wrap180(c.heading - self.lane_heading)
        return -d if self.clockwise else d

    def _mnv_finish(self, c: Ctx):
        self.lane_heading = self.mnv_new_lane
        self._corner_odo = c.odo
        self.planner.reset_lane_along(c.odo)
        self.planner.clear_tracks()
        self.emit(f"[fsm] maneuver done, lane heading {self.lane_heading:+.1f}")
        return self._finish_corner(c)

    # ------------------------------------------------------- FINAL_STRAIGHT

    def _final_straight(self, c: Ctx):
        p = c.p
        if not self.entered:
            self.entered = True
            self.emit(f"[fsm] FINAL_STRAIGHT {self.final_distance_cm:.1f} cm")

        since = self._since_corner_mm(c)
        self.level_enabled = since > p["POST_CORNER_LOCKOUT_CM"] * 10.0
        self.planner_enabled = True
        if since >= self.final_distance_cm * 10.0:
            self._go(State.FINISHED)
            return ActionIntent.stop("final straight complete")
        return ActionIntent.hold(
            wrap180(self.lane_heading + self.planner.lat_yaw_cmd),
            p["DRIVE_PWM"], self._drive_reason())

    # ------------------------------------------------------------- snapshot

    def snapshot(self):
        return {
            "state": self.state.name,
            "corner": self.corner_count,
            "target_corners": self.p["TARGET_CORNERS"],
            "lane_heading": round(self.lane_heading, 1),
            "target_heading": round(self.target_heading, 1),
            "clockwise": self.clockwise,
            "locked_colour": COLOUR_NAMES[self.locked_colour],
            "direction_locked": self.locked_colour != COLOUR_NONE,
            "side_open": self.side_open_count,
            "side_wall": self.side_wall_count,
            "colour_armed": self.dc_colour_armed,
            "colour_muted": self.colour_muted,
            "turn_armed": self.dc_turn_armed,
            "turn_front_cmd": round(self.turn_front_cmd, 0),
            "turn_lock_cmd": round(self.turn_lock_cmd, 2),
            "maneuver": self.dc_turn_maneuver,
            "mnv_phase": self.mnv_phase.name if self.state is State.CORNER_MANEUVER else None,
            "mnv_legs": self.mnv_legs,
            "first_segment_cm": round(self.first_segment_cm, 1),
            "final_distance_cm": round(self.final_distance_cm, 1),
            "level_enabled": self.level_enabled,
            "planner_enabled": self.planner_enabled,
            "planner": self.planner.snapshot(),
        }
