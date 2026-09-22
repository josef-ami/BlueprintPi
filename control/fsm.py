"""
fsm.py — the interpretation + state machine layer. OBSTACLE ROUND ONLY.
(The open round is owned end-to-end by OpenRound.cpp; the Pi is not in that loop.)

This file is deliberately a SKELETON. Every state handler is wired, logged,
timed and safe — but the driving decisions inside them are marked TODO and are
yours to write. Read docs/FSM_GUIDE.md before editing; it walks through each
slot with the geometry you need.

The shape, and why:

    Ctx          one read-only-ish snapshot per tick: fused obstacles from the
                 camera+lidar, plus telemetry from the STM32 (heading, odometry,
                 ToF, floor colour). Every handler sees the same world.

    handler      a plain function (ctx, st) -> ActionIntent. It may set
                 st.next to request a transition. It must NOT sleep, block,
                 or loop — it is called once per tick and must return fast.

    FSM.step()   picks the handler, runs it, applies the transition, enforces
                 the global safety overrides. Handlers never call each other.

Nothing here talks to hardware. The intent goes to the mapper, the mapper to
the link, the link to the STM32.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from .intent import ActionIntent
from .link import Telemetry, FLOOR_NONE, FLOOR_ORANGE, FLOOR_BLUE


# ============================================================ states =====

class State(Enum):
    """
    The obstacle round as a sequence of situations. Add states freely — the
    dispatch table is built from this enum, so a new state needs a handler and
    nothing else.
    """
    WAIT_START = auto()    # armed, motor off, waiting for the STM32's button
    DRIVE = auto()         # straight: hold lane heading, watch for events
    AVOID = auto()         # a pillar is engaged; thread past it on the legal side
    TURN = auto()          # corner: 90 deg, terminated on IMU heading
    LANE_CORRECT = auto()  # post-corner lateral fix (see control_architecture.md s6)
    PARK = auto()          # multi-point shuffle into the bay after lap 3
    FINISHED = auto()      # run complete, stopped
    RECOVER = auto()       # about to hit something: back off and retry


# ============================================================ context ====

@dataclass
class Ctx:
    """
    Everything the FSM can see this tick. Built fresh by main.py each loop.

    obstacles   list[Obstacle] from fuse(): .color ("RED"/"GREEN"/"MAGENTA"),
                .bearing_deg (+ = left), .distance_mm (inf if lidar had no
                return at that bearing), .confidence
    telem       Telemetry from the STM32 — heading_deg, distance_mm (cumulative
                odometry), speed_mmps, tof_*_mm, floor_colour, button_pressed
    lidar       raw LidarResult, for anything the fused obstacles don't cover
                (e.g. wall distances at arbitrary bearings)
    now         time.time() at the top of this tick — use this, not time.time(),
                so every handler in a tick agrees on the clock
    dt          seconds since the previous tick
    """
    obstacles: list = field(default_factory=list)
    telem: Telemetry = field(default_factory=Telemetry)
    lidar: object = None
    now: float = 0.0
    dt: float = 0.0

    # -------- convenience accessors: use these, they handle the edge cases --

    def nearest(self, *colors, max_dist_mm=float("inf"), max_bearing_deg=90.0):
        """
        Closest obstacle matching any of the given colours, or None.
        Obstacles with no lidar range (distance inf) are skipped — you cannot
        make a geometric decision without a distance.

            p = ctx.nearest("RED", "GREEN", max_dist_mm=1200)
        """
        best = None
        for o in self.obstacles:
            if colors and o.color not in colors:
                continue
            if o.distance_mm > max_dist_mm:
                continue
            if abs(o.bearing_deg) > max_bearing_deg:
                continue
            if best is None or o.distance_mm < best.distance_mm:
                best = o
        return best

    @property
    def heading(self) -> float:
        return self.telem.heading_deg

    @property
    def odo_mm(self) -> float:
        """Cumulative odometry. Take differences; the absolute value is arbitrary."""
        return self.telem.distance_mm

    @property
    def front_mm(self) -> float:
        return self.telem.tof_front_mm

    @property
    def left_mm(self) -> float:
        return self.telem.tof_left_mm

    @property
    def right_mm(self) -> float:
        return self.telem.tof_right_mm

    @property
    def floor(self) -> int:
        """FLOOR_NONE / FLOOR_ORANGE / FLOOR_BLUE, from the down-facing TCS34725."""
        return self.telem.floor_colour


# ======================================================== run memory =====

@dataclass
class RunState:
    """
    What the FSM remembers ACROSS ticks. Ctx is this tick; RunState is the run.

    Anything a handler needs to remember goes here, never in a module global —
    that way a reset is one line and there is one place to look when debugging.
    """
    state: State = State.WAIT_START
    next: State = None              # set by a handler to request a transition
    entered_at: float = 0.0         # ctx.now when the current state began
    entered_odo_mm: float = 0.0     # ctx.odo_mm when it began
    entered_heading: float = 0.0

    # --- round progress ---
    corner_count: int = 0           # 12 corners = 3 laps
    clockwise: bool = None          # None until the first line pair decodes it
    lane_heading: float = 0.0       # heading the current straight should hold
    laps_done: int = 0

    # --- corner / line handling ---
    line_lockout_until_mm: float = 0.0   # ignore floor colour until odo passes this
    first_line_colour: int = FLOOR_NONE

    # --- avoidance ---
    avoid_color: str = ""           # "RED" or "GREEN" — which pillar we're passing
    avoid_side: int = 0             # +1 = passing on the left, -1 = on the right
    avoid_shift_mm: float = 0.0     # lateral displacement we committed to
    avoid_entry_odo_mm: float = 0.0

    # --- recovery ---
    recover_return_state: State = State.DRIVE
    recover_attempts: int = 0

    # --- parking ---
    park_phase: int = 0

    def time_in_state(self, ctx: Ctx) -> float:
        return ctx.now - self.entered_at

    def dist_in_state(self, ctx: Ctx) -> float:
        """mm travelled since entering this state. Always >= 0."""
        return abs(ctx.odo_mm - self.entered_odo_mm)

    def go(self, state: State):
        """Request a transition at the end of this tick."""
        self.next = state


# ================================================== tuning constants =====
# Everything a handler might want to tune lives here, not buried in the logic.
# These are STARTING POINTS carried over from ObstacleRound.cpp where one
# existed; re-tune on the mat.

CRUISE_SPEED_MMPS = 400.0       # straights
AVOID_SPEED_MMPS = 320.0        # while threading a pillar
TURN_SPEED_MMPS = 280.0         # through a corner
PARK_SPEED_MMPS = 120.0         # shuffle — slow and repeatable
RECOVER_SPEED_MMPS = -200.0     # reverse

PILLAR_ENGAGE_MM = 900.0        # start caring about a pillar at this range
PILLAR_CLEAR_MM = 250.0         # considered passed once nearer than this / gone
WALL_PANIC_MM = 200.0           # front ToF below this = about to hit
FRONT_TURN_MM = 700.0           # corner fires when front wall is this close

TARGET_CORNERS = 12             # 3 laps
POST_CORNER_LOCKOUT_MM = 500.0  # ignore floor lines this far after a turn
TURN_DONE_TOL_DEG = 3.0         # heading error that counts as "turn finished"


# ====================================================== the handlers =====
#
#  Each handler:  (ctx, st) -> ActionIntent
#  - read ctx, read/write st
#  - return what the car should do THIS TICK
#  - call st.go(State.X) to transition (takes effect after this tick)
#  - never sleep, never loop, never block
#
#  The TODO blocks are the control logic. docs/FSM_GUIDE.md explains each.
# =========================================================================


def on_wait_start(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    Armed and stationary until the STM32 reports the start button.

    The button is on the STM32 (PA5, active low) and arrives latched in
    telemetry, so this needs no debounce. Rule 9.11: one button, one start.
    """
    if ctx.telem.button_pressed:
        # Latch the heading we are sitting at as the first lane heading. The
        # IMU was zeroed at boot while stationary, so this is usually ~0.
        st.lane_heading = ctx.heading
        st.go(State.DRIVE)
        return ActionIntent.stop("start pressed -> DRIVE")
    return ActionIntent.stop("waiting for start button")


def on_drive(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    The straight. Hold the lane heading and watch for the three things that
    end this state: a pillar to avoid, a corner to turn, or the run finishing.

    Event priority is NOT arbitrary — see FSM_GUIDE.md "Why corner beats
    pillar". Missing a pillar costs points; missing a corner ends the run.
    """
    # ---- safety first, always ----
    if ctx.front_mm < WALL_PANIC_MM:
        st.recover_return_state = State.DRIVE
        st.go(State.RECOVER)
        return ActionIntent.stop("front wall panic")

    # ---- 1. corner detection -------------------------------------------
    # TODO(corner): decide when a corner is starting.
    #   Inputs you have: ctx.floor (FLOOR_ORANGE / FLOOR_BLUE from the
    #   down-facing TCS34725), ctx.front_mm, ctx.odo_mm.
    #   Remember: every corner has BOTH an orange and a blue line, so a naive
    #   trigger fires ~24 times over 3 laps. Use st.line_lockout_until_mm.
    #   The FIRST line pair of the run decodes driving direction into
    #   st.clockwise — until that is set you do not know which way to turn.
    #   See FSM_GUIDE.md section "Corner detection and direction decode".
    #
    # if <corner confirmed>:
    #     st.go(State.TURN)
    #     return ActionIntent.hold_heading(st.lane_heading, TURN_SPEED_MMPS,
    #                                      reason="corner armed")

    # ---- 2. pillar engagement ------------------------------------------
    # TODO(pillar): decide whether a pillar is close enough to act on.
    #   pillar = ctx.nearest("RED", "GREEN", max_dist_mm=PILLAR_ENGAGE_MM)
    #   Rule 9.19: RED is passed on the RIGHT, GREEN on the LEFT.
    #   Set st.avoid_color / st.avoid_side / st.avoid_entry_odo_mm before
    #   transitioning so AVOID knows what it is doing.
    #
    # if pillar is not None:
    #     st.avoid_color = pillar.color
    #     st.avoid_side  = -1 if pillar.color == "RED" else +1
    #     st.avoid_entry_odo_mm = ctx.odo_mm
    #     st.go(State.AVOID)

    # ---- 3. run complete ------------------------------------------------
    if st.corner_count >= TARGET_CORNERS:
        st.go(State.PARK)
        return ActionIntent.hold_heading(st.lane_heading, PARK_SPEED_MMPS,
                                         reason="12 corners done -> PARK")

    # ---- default: hold the lane ----
    return ActionIntent.hold_heading(st.lane_heading, CRUISE_SPEED_MMPS,
                                     reason="cruise")


def on_avoid(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    Thread past a pillar on the legal side, then come back to the lane.

    ObstacleRound.cpp ran this as six phases (swerve out, straighten, hold,
    return, realign, backup). You can do the same with st.park_phase-style
    sub-phases, or as a single continuous geometric controller — the mapper's
    steer_for_lateral_shift() exists to make the second approach practical.
    """
    if ctx.front_mm < WALL_PANIC_MM:
        st.recover_return_state = State.AVOID
        st.go(State.RECOVER)
        return ActionIntent.stop("front wall panic during avoid")

    # TODO(avoid): the avoidance manoeuvre.
    #   Geometry helpers (from .mapper):
    #       steer_for_lateral_shift(shift_mm, over_distance_mm)
    #       steer_for_radius(radius_mm) / radius_for_static(...)
    #   The corridor is 1000 mm; the pillar is 50 mm; SIDE_SAFE margin was
    #   100 mm in the C firmware. Work out the lateral shift you need, convert
    #   it to an angle, and drive it.
    #   Release when the pillar is gone or passed (PILLAR_CLEAR_MM), then
    #   st.go(State.DRIVE) after returning to st.lane_heading.
    #   See FSM_GUIDE.md section "Pillar avoidance".

    return ActionIntent.hold_heading(st.lane_heading, AVOID_SPEED_MMPS,
                                     reason="avoid: TODO, holding lane")


def on_turn(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    A 90 degree corner, terminated on measured heading — never on steering
    angle or time, so tyre slip cannot corrupt it. ("The heading is the truth.")

    On entry, st.lane_heading should already have been stepped by +-90.
    """
    # TODO(turn): drive the corner.
    #   target = st.lane_heading (already stepped by the corner logic)
    #   error  = wrap180(target - ctx.heading)
    #   Ease steering and speed down as |error| shrinks so you don't overshoot
    #   the exit. Terminate when |error| < TURN_DONE_TOL_DEG.
    #   Add an odometry backstop (st.dist_in_state) so a dead IMU cannot hang
    #   this state forever.
    #   On completion: st.corner_count += 1, then st.go(State.LANE_CORRECT).
    #   See FSM_GUIDE.md section "Turns".

    return ActionIntent.hold_heading(st.lane_heading, TURN_SPEED_MMPS,
                                     reason="turn: TODO")


def on_lane_correct(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    Post-corner lateral correction.

    The orange and blue corner lines fan out radially, so the encoder distance
    between crossing them is a ruler for how far you are from the inner wall.
    control_architecture.md section 6 has the full method and the constants
    that worked (K_LAT 4 deg/cm, deadband 2 cm, correction over 25 cm).

    This state is OPTIONAL. If you want to skip it while bringing the rest up,
    leave the straight-through below and it becomes a no-op.
    """
    # TODO(lane): measure the gap, convert to a lateral error, hold an offset
    #   angle for CORRECTION_DISTANCE, then rejoin the lane heading.

    st.go(State.DRIVE)
    return ActionIntent.hold_heading(st.lane_heading, CRUISE_SPEED_MMPS,
                                     reason="lane correct: pass-through")


def on_park(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    Parallel park into the magenta bay. 22 of 122 points.

    SPECSHEET section 3 is blunt about this: a two-arc park is NOT geometrically
    feasible at 35 deg lock (R/L = 0.95, clearance -25.6 mm). Decision #21 was
    amended to a MULTI-POINT SHUFFLE: enter at whatever angle fits, then
    alternate forward/back against IMU yaw to straighten.

    Touching a limiter is rule 9.24.7 — instant zero for parking. Slow wins.
    """
    # TODO(park): the shuffle.
    #   Find the bay: ctx.nearest("MAGENTA") gives you the limiter blocks.
    #   Phase it with st.park_phase; each phase is a short forward or reverse
    #   leg at PARK_SPEED_MMPS with a fixed steer angle, terminated on
    #   st.dist_in_state() or on heading error.
    #   Parallel means both same-side wheels within 20 mm of the wall, so the
    #   exit condition is a heading test, not a position test.
    #   See FSM_GUIDE.md section "Parking".

    st.go(State.FINISHED)
    return ActionIntent.stop("park: TODO, stopping")


def on_finished(ctx: Ctx, st: RunState) -> ActionIntent:
    """Run over. Stay stopped — rule 9.24.2 wants a demonstrated full stop."""
    return ActionIntent.stop("finished")


def on_recover(ctx: Ctx, st: RunState) -> ActionIntent:
    """
    Something is too close in front. Reverse on mirrored steering until there
    is room, then hand back to whatever state called us.

    Bounded on purpose: st.recover_attempts stops an infinite bump-and-retry
    against a wall the car cannot escape.
    """
    # TODO(recover): reverse until ctx.front_mm > ~350 mm or a distance cap,
    #   then st.go(st.recover_return_state). Increment st.recover_attempts and
    #   give up into FINISHED after ~3 tries.

    return ActionIntent.steer(0.0, RECOVER_SPEED_MMPS, reason="recover: TODO")


# ===================================================== the machine =======

HANDLERS = {
    State.WAIT_START: on_wait_start,
    State.DRIVE: on_drive,
    State.AVOID: on_avoid,
    State.TURN: on_turn,
    State.LANE_CORRECT: on_lane_correct,
    State.PARK: on_park,
    State.FINISHED: on_finished,
    State.RECOVER: on_recover,
}


def wrap180(angle_deg: float) -> float:
    """Normalise to (-180, +180]. Use for every heading subtraction."""
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a


class FSM:
    """
    Owns RunState and dispatches to handlers. main.py calls step() once a tick.

    Safety overrides applied here, above every handler, so no TODO you leave
    in a handler can drive the car into a wall unattended:
      - stale telemetry  -> stop (the STM32 is also running its own watchdog)
      - watchdog tripped -> stop
    """

    def __init__(self, on_transition=None):
        self.st = RunState()
        self._on_transition = on_transition   # optional callback(old, new) for logs

    def reset(self):
        self.st = RunState()

    def step(self, ctx: Ctx) -> ActionIntent:
        st = self.st

        # ---- global overrides ------------------------------------------
        if not ctx.telem.fresh():
            return ActionIntent.stop("telemetry stale — STM32 not reporting")
        if ctx.telem.watchdog_tripped:
            return ActionIntent.stop("STM32 watchdog tripped")

        # ---- run the handler -------------------------------------------
        st.next = None
        handler = HANDLERS.get(st.state)
        if handler is None:
            return ActionIntent.stop(f"no handler for {st.state}")
        intent = handler(ctx, st)

        # ---- apply the transition --------------------------------------
        if st.next is not None and st.next is not st.state:
            old = st.state
            st.state = st.next
            st.entered_at = ctx.now
            st.entered_odo_mm = ctx.odo_mm
            st.entered_heading = ctx.heading
            if self._on_transition:
                self._on_transition(old, st.state)
        st.next = None

        return intent
