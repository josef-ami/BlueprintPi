# The obstacle round

How the car gets round three laps without touching a pillar, and which box
each decision is made in.

## The shape

```
    VisionThread ─┐
                  ├─> perception ─> Ctx ─> FSM ─> ActionIntent ─> mapper ─┐
    LidarThread  ─┘        ^          (control/fsm.py)                    │
                           │                                              v
                  WallFit, pillar xy,                                   Link
                  candidates                                              │
                           │                                              v
                           └──────────── Telemetry ◄──────────────────  STM32
```

Everything above the link is Python on the Pi and runs once per tick at 50 Hz.
Everything below it is `firmware/ObstacleRound.cpp` and runs at its own rate.

This used to be split the other way round: the Pi was a sensor pipe that sent
sixteen numbers down an ASCII line, and the firmware held the state machine,
the lane planner and the pillar tracks. The logic is the same logic — it was
ported, not rewritten — but it lives where the camera and the LiDAR already
are, which means the planner can see a `Pillar` object instead of a pair of
integers, and you can put a breakpoint in it.

### Why the split falls where it does

| On the Pi, 50 Hz | On the STM32, its own rate |
|---|---|
| the seven states | heading PID, IMU rate (~100 Hz) |
| lane offset from the wall cones | servo slew, 2.5°/update |
| pillar tracks, pass planning, give-up | the 90° arc's inner loop |
| wall levelling | the wall-panic reflex |
| corner trigger, corner-exit shaping | encoder, IMU, colour sensor |
| the 3-point corner's sequencing | servo and motor PWM |

The old firmware had already made this choice for us. `updatePlanner()` began
with `if (!lidarNewFrame) return;` and `updateLevel()` with a check on
`lidarNewRev` — both were *already* running at the Pi's frame rate, so moving
them up cost no control bandwidth at all. `updateDriveSteer()` was gated on
`gImuFresh` and steps the servo by at most `SERVO_SLEW` per IMU event; that
one could not move and did not.

The wire is `docs/PI_STM32_PROTOCOL.md`.

## Perception

### The camera says which and where; the LiDAR says how far

`sensors/camera.py` finds pillars. A pillar is an upright, solid, saturated
block **standing on the white mat**, and every colour blob has to pass five
tests before the planner ever hears about it:

| code | test | what it throws out |
|---|---|---|
| `A` | aspect: `h >= aspect_min * w` | orange and red floor lines, which are flat |
| `S` | solidity: `area >= solidity_min * bbox` | streaks, ragged noise |
| `F` | the strip just under the blob is mostly mat | a red shirt with no floor under it |
| `L` | that mat connects to the mat in front of the car | anything standing beyond a wall — its mat is the *next* straight's |
| `C` | the blob is more colourful than the mat under it | washed-out reflections and shadows |

`A`, `S`, `F` and `C` came from the obstacle round's detector; `L` came from
the calibration dashboard's. They were two separate filters on two separate
pages and are now one, because there is one robot. A blob whose base runs off
the bottom of the frame skips `F`, `L` and `C` — nothing but a pillar can be
that close.

**Two colour spaces.** HSV separates the pillars on hue but gates on
*saturation*, and a matte pillar under dim light falls under the threshold and
vanishes — green first, because red has two hue bands and survives longer. In
Lab, `a > 128` is red and `a < 128` is green, the mat sits near neutral
whatever the light does to `L`, and brightness never removes the colour.
`USE_LAB` picks. The Calibrate tab fits the Lab ranges by clicking.

**Bearings** come from the fisheye `K`/`D` when `USE_INTRINSICS` is on, and
from an ideal equidistant lens over `HFOV_DEG` when it is off. Which is right
is a *measurement*, not a preference — `config.json` disagrees with itself
here, `K` putting the frame edge at ±48° and `HFOV_DEG` at ±80°. Measure a
pillar at a known angle before trusting either; the answer decides whether the
car can see a pillar just past a corner at all.

### The LiDAR does four jobs

1. **Three beams** at 0/90/270° — the corner trigger and the front panic.
2. **Wall cones.** A line fitted to every return in a 45° cone each side, by
   RANSAC over every point pair. Gives the *perpendicular* distance to each
   wall — which, unlike a single beam, does not grow when the car is yawed —
   and the car's yaw relative to the walls. Ties go to the **farther** line,
   so a pillar between the car and the wall is outliers rather than the fit.
3. **Pillar range.** The camera's bearing picks the ray; the nearest return
   near that ray whose distance agrees with the blob's size gives the
   distance. Not an exact ray match: a scan can be 100 ms old, and while the
   car swerves at ~100°/s its bearings are ~10° stale, so an exact match hits
   the wall behind the pillar.
4. **Candidates.** Small clusters inside the corridor that the camera has not
   named. The camera covers about ±48°, so a pillar near the far wall can stay
   out of view until too late; the LiDAR sees it from the start line.
   Clustering happens *before* filtering — a wall is one long run and is
   thrown out whole by the width test, where filtering first would chop it
   into pillar-sized fragments.

## The planner

`control/planner.py`. Everything is a **lateral position in the lane** (mm,
`+` = left of centre):

```
car      lane_off    from the two cone fits
pillar   track.lat   car lane position + the pillar's car-frame position,
                     rotated by the car's yaw into the lane

red  must be passed on its right  ->  car lat <= pillar lat - PASS_CLEAR
green must be passed on its left  ->  car lat >= pillar lat + PASS_CLEAR
```

The car aims for the lane position closest to the centre that satisfies every
pillar it is approaching or still alongside, and steers there with a yaw
command the STM32's heading PID tracks. With no pillar in play the target is
the lane centre — that **is** the wall centring; there is no separate law.

The yaw points at a **pass point**: the target lane position `PASS_LEAD_MM`
*before* the pillar. So the swerve sharpens as the pillar nears and arrives in
time. A plain proportional law eases off near the target and arrived too late.

### Why not a pixel law

The original held the pillar at a fixed image offset (±150 px). At close range
that makes the car **orbit** the pillar and hit it, and with the 160° lens a
pillar is only big enough to react to in the last ~50 cm. Lane positions have
neither problem.

### Giving up, and then giving up harder

Before committing, the planner asks whether the car can still *reach* the pass
position in the distance left (`lat_reach`: arc at full lock up to
`PASS_YAW_MAX`, then straight). Three outcomes:

1. **Reachable** — go.
2. **Correct side unreachable, other side reachable** — flip the track and
   pass on the wrong side deliberately. A wrong-side pass costs points. Check
   your rulebook penalty; `ALLOW_GIVE_UP` turns this off.
3. **Neither reachable with full clearance** — fall back to the bare
   geometric miss (`AVOID_CLEAR_MM`, no `PASS_MARGIN`) and take whichever side
   is closer. Holding an unreachable target drives the car *into* the pillar;
   the simulator did exactly that on a red 160 mm past a corner. A wrong-side
   pass costs points; a collision costs the run.

### Levelling

The wall fit is noisy per revolution but has **no drift**; the IMU is smooth
but drifts. So the lane heading is nudged a small step toward the fitted wall
direction once per revolution — and only when the fit is trustworthy: both
walls inside a corridor width, the car not badly yawed, no pillar steering it,
and the estimate close enough to the IMU that a disagreement means a bad fit
rather than drift.

## The corner

The corner, not the camera, is what loses a pillar on the next straight. A
plain arc fired the moment the turn side opens always ends with the car pinned
to the **outer** wall of the new straight. From there an outer-side pass is
free and an inner-side pass is unreachable, so the reach check correctly gives
up and the car deliberately goes by on the wrong side.

So the exit lane position is a commanded quantity, decided **before** the arc:

| `TURN_FRONT_MM` | 0.55 | 0.70 | 0.85 | 1.00 |
|---|---|---|---|---|
| 500 | +382 | +339 | +282 | +241 |
| 600 | +335 | +241 | +190 | +150 |
| 750 | +194 | +108 | +51 | −27 |
| 900 | +48 | −41 | −109 | −160 |

(simulated exit offset just after each corner, `+` = outer, lane limit ±398;
columns are `TURN_LOCK_FRACTION`)

Firing the arc while the wall ahead is still far, and tightening it, walks the
car from the outer wall to the inner one. Nothing the car does afterwards has
that authority — a pillar 200 mm past the corner leaves no room to cross.

> **Where the numbers in this section come from.** The exit-offset table, the
> finish rates and the wrong-side percentages were measured by the
> obstacle-round simulator that shipped with the firmware version this was
> ported from (`sim/fieldsim.py`, `regress.py`, `cornersweep.py` in that
> package). That simulator compiles the firmware with the state machine
> *inside* it and feeds it the old ASCII frame, so it cannot run against this
> repo as-is, and the figures have **not** been re-measured on the Pi-side
> port. What this repo does check is that the port makes the same decisions:
> the planner and FSM tests pin each rule to the firmware's behaviour, and
> `tests/test_firmware_sim.py` runs the Pi's FSM against the real compiled
> executor round a full lap. A regression sweep over pillar layouts on this
> architecture is the missing piece.

### Knowing which way to exit

The camera **does** see the next straight's first pillar, early, across the
corner. Those sightings have meaningless coordinates in the current lane
frame, and carrying the coordinates through the corner does not work: the car
travels most of a corridor width during the arc, so rotating an estimate 90°
about the car is wrong by hundreds of mm, and a confident wrong pillar is
worse than none.

But the coordinates are not what is needed. All the corner has to decide is
**which side to come out on**, and that needs only the colour — one bit, which
survives the corner perfectly.

Note the sign: the passing rule is about the **lane**, not the lap. Red is
passed on its right, which is the right-hand side of the lane, whichever way
round the car is going. Whether that is the inner or the outer side then
depends on the direction. Tying red to "inner" is right clockwise and exactly
backwards anticlockwise.

There are two sources for that bit, and `LanePlanner.planned_next_colour()`
prefers the first:

**The second pillar** (`USE_SECONDARY_CORNER`). The camera reports the two
largest accepted blobs, not one. The nearest is what the planner steers
around; the next one back, on the run up to a corner, is almost always the
first pillar of the next straight. Reporting only the largest made that
pillar invisible exactly when it mattered.

Distance alone does not qualify it — a straight with two pillars on it also
has a nearer and a further one, and shaping the corner from the second pillar
of the *current* straight is worse than not shaping it at all. So the second
pillar goes into the lane frame exactly as an ordinary sighting does and its
colour counts only when it lands **outside** this corridor
(`PILLAR_MAX_LAT_MM`), on the side the car is about to turn towards, and
further away than the pillar being passed. `SECONDARY_ZONE_MM` is how long,
in lane distance, that colour stays good.

One subtlety: before the floor colour has been read, the turn direction is
only a *guess*. A side test applied then throws away everything on one side
for the whole first straight, so `FSM.direction_known()` suspends it until a
colour has actually been seen.

**The cross-corner sighting** (`CARRY_TRACKS`) is the fallback, for a frame
in which only one pillar was visible at all: a *primary* pillar whose lane
coordinates put it outside this corridor.

### Three arc shapes

With the exit side known before the arc, `_plan_corner_exit()` picks one of
three settings rather than two:

| exit | trigger | lock | meaning |
|---|---|---|---|
| inner | `TURN_FRONT_INNER_MM` (1050) | `TURN_LOCK_INNER` (1.00) | turn early and tight — the short, near corner |
| outer | `TURN_FRONT_OUTER_MM` (600) | `TURN_LOCK_OUTER` (0.70) | run on, then turn on a looser arc |
| unknown | `TURN_FRONT_MM` (600) | `TURN_LOCK_FRACTION` (0.70) | the middle setting, which exits near the lane centre and leaves both sides reachable |

`TURN_FRONT_OUTER_MM` ships **equal to** `TURN_FRONT_MM`, i.e. the genuinely
wider arc is off. It is implemented, and the honest reason it is off is that
the sim says it does not pay: at 500 it buys an outer-side green about 60 mm
of clearance but exits at +339 against a limit of 398, and that cost 5
finished runs out of 24 in extra outer-wall contacts. 550 is worth trying on
the real mat, where the turning radii are not the sim's, but start with it
off.

### Enter wide, leave tight

A 90° arc throws the car **away** from the side it started on, so the way to
come out of a corner near the inner wall is to go into it near the outer one
— the ordinary racing line. Because the exit side is known *before* the
corner, the entry is simply its opposite, and `PRE_CORNER_SWING_MM` (250)
holds that line once the wall ahead is inside `PRE_CORNER_ZONE_MM` (900) and
no pillar is in play.

This is the half of the idea that pays: across all layouts it took wrong-side
passes from 21 % to 15 % and all but eliminated the recovery thrashing.

### Which trigger fires

1. The inner wall must have been **seen** (`SIDE_WALL_FRAMES` short readings)
   before an opening counts. A straight that starts inside a corner square —
   after a 3-point, or a late turn — sees "open" before the inner wall even
   begins, and that faked a second corner.
2. Then `SIDE_OPEN_FRAMES` consecutive readings past `SIDE_OPEN_MM`, and only
   while the car is within `TURN_TRIGGER_MAX_YAW` of the lane direction —
   mid-swerve the side beam is not sideways and fakes an open corner.
3. That **arms** the turn and takes the corner measurements. The arc itself
   starts later, when the wall ahead is about one turn radius away
   (`TURN_FRONT_MM`), with `TURN_DELAY_MM` as a minimum run-out and
   `TURN_ARM_MAX_MM` as the backstop for a missing front reading.

On the first corner the floor colour arms the gate and locks the direction:
orange = clockwise, blue = anticlockwise.

## The 3-point corner

On by default (`USE_CORNER_MANEUVER`), and gated. Clockwise only.

A red passed on its right — the inner side — just before the corner leaves the
car hugging the inner wall, and a plain arc from there lands it on the
**outer** side of the next straight. That is fatal only when the next
straight's first pillar needs the inner side too. So the 3-point fires when
both are true: an inner-side pass within `MNV_ZONE_MM` of the corner, **and**
`_plan_corner_exit()` has planned an inner exit. If the next pillar wants the
outer side, the plain arc is already doing the right thing, and stopping to
shuffle only costs clearance — in sim the outer-side greens went from
150–393 mm down to 79–92 mm when it fired on them for nothing.

Gated that way it is a clear win across every layout (finished 16/24 vs 15,
wrong-side passes 13 % vs 15 %), which is why it now ships on. On the case it
was written for — a red late on the straight and another just past the corner
— it turns crashes into passes.

The manoeuvre: swing toward the planned exit side, arc forward at full lock,
stop, reverse at the **opposite** lock (which keeps rotating the car the same
way), repeat if short.

Each forward leg must add `MNV_LEG_STEP_DEG` of **new** rotation, measured
from where that leg started. Measuring against the total turned since the old
lane does not work: reversing keeps rotating the car the same way, so by the
time a reverse leg ends the total already exceeds the next threshold, and
every later forward leg ended on its first pass without moving. The manoeuvre
silently collapsed into one forward arc and one reverse.

## Running it

```
python3 dashboard.py                      # the page, on :8080
python3 -m control.obstacle_round         # the same loop, headless
python3 -m control.obstacle_round --dry   # decide and print, send nothing
```

One process may hold the camera and the LiDAR at a time. `openRound.py` is the
open challenge and is deliberately untouched by any of this.

## Where the numbers live

`config.json`, one file:

- `params` — every Pi tunable, flat, by name. 147 of them, including the 73
  that used to be firmware parameters.
- `stm32` — the 20 the firmware still owns: servo geometry, the heading PID,
  the arc gains, the panic reflex.
- `camera_intrinsics`, `lidar`, `serial` — structured, not tunables.

Both halves are edited on the Tune tab. Pi values apply on the next tick;
STM32 values are queued and pushed a few per loop, so tuning can never stall
the 50 Hz feed. The firmware has no storage — it boots with compiled-in
defaults and a fresh `boot_id`, the Pi notices in the next TELEM frame and
pushes the saved set back, so a mid-run reset is back on your numbers within a
few hundred ms.
