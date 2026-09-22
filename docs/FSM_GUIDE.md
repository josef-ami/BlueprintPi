# How to Write the Control Logic

`control/fsm.py` is a working skeleton with the driving decisions left as
`TODO` blocks. This document explains what goes in each one, what tools you
have, and the traps specific to this car and this rulebook.

Read the contract first, then the per-state sections.

---

## The contract every handler obeys

```python
def on_something(ctx: Ctx, st: RunState) -> ActionIntent:
    ...
    return ActionIntent.hold_heading(st.lane_heading, 400, reason="why")
```

Four rules, and breaking any of them breaks the whole loop:

1. **Return an `ActionIntent` every time.** No exceptions, no `None`. The
   control loop is not allowed to skip a tick.
2. **Never block.** No `sleep()`, no `while` waiting for something, no serial
   reads. You are called ~30×/second and must return in well under a
   millisecond. To "wait", stay in the state and check a condition each tick.
3. **Transition with `st.go(State.X)`.** It takes effect *after* your return,
   so you still return the intent for this tick. Handlers never call each other.
4. **Persist in `st`, not in globals.** `ctx` is this tick; `st` is the run.
   Anything remembered across ticks goes in `RunState` so a reset is one line.

### What you can see (`ctx`)

| | |
|---|---|
| `ctx.obstacles` | fused camera+lidar pillars: `.color`, `.bearing_deg` (+ = left), `.distance_mm` (inf if no lidar return), `.confidence` |
| `ctx.nearest("RED","GREEN", max_dist_mm=900)` | closest match, or `None`. **Skips obstacles with no range** — you can't do geometry without a distance |
| `ctx.heading` | degrees, + = left of the boot heading, wrapped ±180 |
| `ctx.odo_mm` | cumulative odometry. Take *differences*; the absolute value is arbitrary |
| `ctx.front_mm` / `.left_mm` / `.right_mm` | ToF, `inf` when invalid |
| `ctx.floor` | `FLOOR_NONE` / `FLOOR_ORANGE` / `FLOOR_BLUE` from the down-facing TCS34725 |
| `ctx.now`, `ctx.dt` | use these, not `time.time()`, so a tick is internally consistent |

### What you remember (`st`)

`st.dist_in_state(ctx)` and `st.time_in_state(ctx)` are how you terminate
legs. Every field in `RunState` has a comment; add more freely.

### What you can ask for

```python
ActionIntent.hold_heading(heading_deg, speed_mmps, reason="")  # straights
ActionIntent.steer(angle_deg, speed_mmps, reason="")           # arcs, + = left
ActionIntent.stop(reason="")                                    # motor off
```

`speed_mmps` may be negative to reverse. The mapper clamps everything to the
car's real limits, so asking for 90° or 5000 mm/s is safe — you get 35° and
700 mm/s. Always fill in `reason`: it prints on the status line and it is what
you'll read when a run goes wrong.

### Geometry helpers (`control.mapper`)

These exist so you write physics, not magic numbers:

```python
steer_for_radius(194.4)                     # → 35.0°  (that's full lock)
radius_for_steer(20)                        # → 374 mm
steer_for_lateral_shift(250, 800)           # +250mm sideways over 800mm → +6.1°
```

Car facts, CAD-verified: wheelbase **136.14 mm**, lock **±35°**, minimum turn
radius **194.4 mm**, top speed **700 mm/s**, body **165 × 114 mm**.

---

## Global invariants

These hold everywhere and the skeleton already enforces the first two:

- **Stale telemetry or a tripped watchdog → STOP**, checked above every
  handler in `FSM.step()`. No TODO you leave can drive the car blind.
- **Front ToF below `WALL_PANIC_MM` (200 mm) → `RECOVER`**, wired into `DRIVE`
  and `AVOID`. Add it to any new state that moves forward.
- **"The heading is the truth."** Terminate manoeuvres on *measured heading*,
  never on steering angle or elapsed time. Tyre slip corrupts the latter two;
  the IMU is immune.
- **Always add an odometry backstop.** Any state that waits for a sensor event
  must also give up after a distance (`st.dist_in_state(ctx)`). A dead sensor
  must not hang the run.

---

## `on_drive` — the straight

This is the dispatcher. Hold the lane, watch for three events.

### Why corner beats pillar

Evaluate the corner check **before** the pillar check, and don't start an
avoidance once a corner has armed. The asymmetry is deliberate:

> Missing a pillar forfeits points. Missing a corner ends the run.

An avoidance already in progress finishes first — it's `AVOID`'s job to
complete, not to be interrupted mid-swerve.

### Corner detection and direction decode

Two things make this harder than "saw a line, turn":

**Every corner has both an orange and a blue line** (rule 13.9). A naive
trigger fires ~24 times over 3 laps, not 12. Use `st.line_lockout_until_mm`:
after a turn completes, ignore all floor colour until `ctx.odo_mm` passes
that mark (`POST_CORNER_LOCKOUT_MM`, 500 mm is the proven value).

**The driving direction is drawn randomly per round** (rule 9.3), so you can't
hardcode turn direction. The first line pair of the run decodes it: orange
first → clockwise → all 12 corners are right turns; blue first → counter-
clockwise → all left. Store it in `st.clockwise` and never re-decide. Until
it's set, you don't know which way to turn.

The line *arms* the corner; the turn *fires* when the front ToF confirms the
wall is close (`FRONT_TURN_MM`, 700 mm) — or on a distance backstop if no ToF
is working. Colour alone is the degraded fallback, not the primary.

On firing: step `st.lane_heading` by ±90 (sign from `st.clockwise`), then
`st.go(State.TURN)`.

### Pillar engagement

```python
pillar = ctx.nearest("RED", "GREEN", max_dist_mm=PILLAR_ENGAGE_MM)
```

Rule 9.19 is absolute: **red is passed on the right, green on the left.**
Set `st.avoid_color`, `st.avoid_side` (+1 left, −1 right), and
`st.avoid_entry_odo_mm` *before* transitioning so `AVOID` knows its job.

---

## `on_avoid` — threading a pillar

The corridor is 1000 mm; the pillar is 50 mm square, sitting somewhere in it.
You need to be on the legal side by the time you arrive, then return to the
lane.

Two approaches, both valid:

**Phased** (what `ObstacleRound.cpp` did): swerve out → straighten → hold →
return → realign, with a sub-phase counter in `st`. Each phase terminates on
distance or heading. Predictable, easy to debug, more code.

**Continuous**: each tick, compute the lateral shift you still need and convert
it to an angle with `steer_for_lateral_shift(shift, distance_to_pillar)`.
Fewer states, naturally handles the pillar moving in the frame, but you must
handle the pillar leaving the frame (`distance_mm` goes `inf`, or `nearest`
returns `None`) without lurching.

Whichever you pick: remember the displacement you committed to
(`st.avoid_shift_mm`) so the return leg is symmetric. Keep a hard floor of
~100 mm from any wall. Release when the pillar is passed
(`PILLAR_CLEAR_MM`) or gone, then rejoin `st.lane_heading` and
`st.go(State.DRIVE)`.

---

## `on_turn` — the corner

```python
error = wrap180(st.lane_heading - ctx.heading)   # lane_heading already stepped ±90
```

Ease both steering and speed down as `|error|` shrinks, so you don't overshoot
the exit. The proven shape from the C firmware:

```
steer = clamp(K_turn × |error|, TURN_MIN_STEER, TURN_MAX_STEER)
speed = clamp(K_v    × |error|, TURN_MIN_SPEED, TURN_MAX_SPEED)
```

Terminate at `|error| < TURN_DONE_TOL_DEG` — no settle delay, residual error
gets absorbed on the next straight. Add an odometry cap (~1200 mm) so a dead
IMU can't hang the state.

On completion: `st.corner_count += 1`, then `st.go(State.LANE_CORRECT)`.

---

## `on_lane_correct` — the radial-line ruler

Optional; the skeleton passes straight through so you can bring everything
else up first.

The insight (full version in `control_architecture.md` §6): the orange and
blue corner lines **fan out radially**, so the encoder distance between
crossing them measures how far you are from the inner wall. The encoder is the
one sensor immune to optical noise.

Measure the gap at corner 1 and store it as `gapRefCm` — self-referenced, so
the car holds whatever line it started on rather than assuming an ideal
position. On later corners, `error = measured − reference`, convert to a
steering offset (4°/cm worked, capped at 30°), hold it for 25 cm at reduced
speed, then ease back. Deadband 2 cm to stop hunting; skip the correction
entirely if the partner line was never seen.

---

## `on_park` — the hard one

**22 of 122 points, and the geometry does not work the obvious way.**

`SPECSHEET.md` §3 is blunt: a two-arc park is **not feasible** at 35° lock.
R/L = 0.95, best clearance −25.6 mm — the front outer corner sweeps into the
entry limiter at ~37° heading. Raising the lock doesn't save it (40° still
collides, 45° merely touches). And the problem is **scale-invariant**: the bay
is always 1.5 × car length, so shortening the car shrinks the bay too.

Decision #21 was amended accordingly: **multi-point shuffle**. Enter at
whatever angle fits, then alternate short forward/reverse legs, straightening
against IMU yaw each time.

Practical shape: phase it with `st.park_phase`. Each phase is one short leg —
fixed steer angle, `PARK_SPEED_MMPS` (slow), terminated on
`st.dist_in_state(ctx)` or on heading error. Find the bay with
`ctx.nearest("MAGENTA")`.

The exit test is **heading**, not position: "parallel" means both same-side
wheels within 20 mm of the wall, which is an angular condition.

Touching a limiter is rule 9.24.7 — **instant zero for parking**. Slow and
repeatable beats fast and clever here. This is also the state that most
depends on `TICKS_PER_CM` being correct; see the protocol doc's warning.

---

## `on_recover` — the failsafe

Reverse on mirrored steering until `ctx.front_mm` is comfortable (~350 mm) or
a distance cap is hit, then `st.go(st.recover_return_state)` — the state that
called you, already stored.

Bound it: increment `st.recover_attempts`, and after ~3 tries give up into
`FINISHED` rather than bumping a wall forever. Ignore floor lines while
reversing, or you'll re-cross a corner line backwards and corrupt the count.

---

## Testing without the car

```bash
python3 main_obstacle.py --dry
```

runs the FSM with no serial port at all. Telemetry stays empty, so the
staleness override keeps it in STOP — useful for confirming perception and the
tick rate, not for logic.

To exercise logic properly, drive the FSM with synthetic telemetry:

```python
from control import FSM, Ctx, Telemetry, State
import time
fsm = FSM(on_transition=lambda o,n: print(o.name, "->", n.name))
t = Telemetry(stamp=time.time(), button_pressed=True,
              heading_deg=0.0, tof_front_mm=900.0, distance_mm=0.0)
intent = fsm.step(Ctx(telem=t, obstacles=[], now=time.time()))
print(fsm.st.state.name, intent.reason)
```

Build the scenario you care about — a pillar at a bearing, a floor colour, a
heading error — and assert the state and intent. This is far faster than
finding logic bugs on the mat, and it costs nothing to keep as a test file.

---

## Order I'd build them in

1. `on_drive` cruise + `on_turn` — get it round one lap with no pillars.
2. Corner counting and the direction decode — get 12 corners reliable.
3. `on_avoid` — add pillars.
4. `on_lane_correct` — tighten up the lane over 3 laps.
5. `on_park` — last, because it's the most geometry and the most tuning.

Each step is independently testable on the mat, and each one is worth points
on its own.
