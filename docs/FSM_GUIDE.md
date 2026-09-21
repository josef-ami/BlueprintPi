# The FSM, state by state

`control/fsm.py`. This file used to be a skeleton with TODOs in it; it is the
real state machine now, ported from `ObstacleRound.cpp`. Read
`docs/OBSTACLE_ROUND.md` first for the shape, and
`docs/PI_STM32_PROTOCOL.md` for the wire.

## The contract

```python
intent = fsm.step(ctx)      # once per tick, 50 Hz
```

`Ctx` is everything the FSM can see this tick: telemetry from the STM32
(heading, odometry, floor colour, the recovery flag), perception from the
camera and LiDAR, and the live parameters. `ActionIntent` is what it wants:
a mode, a heading or a steering angle, and a speed. Nothing in between touches
hardware.

A handler is a plain method. It must not sleep, block or loop — it is called
once per tick and must return fast.

## Odometry, and why there is no `zeroEncoder()`

The firmware used to zero the encoder at each corner and measure with
`absEnc(readEncoder())`. With the FSM up here that would be a race: the Pi
would ask for a zero and not know which frames were taken before it landed.
So `odo` free-runs on the STM32 and this file keeps its own baselines:

- `_corner_odo` — set at GO and at each corner. `_since_corner_mm()` is the
  direct replacement for `absEnc(readEncoder())`.
- `_dc_base` — set when `DRIVE_TO_CORNER` is entered, including on a restart.
- `_dc_arm_odo`, `_turn_start_odo`, `_mnv_base_odo` — their own spans.

`_dc_base` and `_corner_odo` are **not** the same: the safety-distance restart
resets the first and not the second, exactly as the firmware did.

## States

### `WAIT_START`

Armed, motor off. A Start that arrives *before* the state is entered is
discarded — so a Start sent while the car was still booting cannot launch it
the moment it arms. Start is refused if the LiDAR is stale or the STM32 is not
reporting.

On GO: lane heading = wherever the car is pointing, corner count to zero,
tracks cleared, baselines set.

### `DRIVE_TO_CORNER`

Holds `lane_heading + planner.lat_yaw_cmd` at `DRIVE_PWM`.

The planner is live from the **first tick** after the arc — a pillar right
after the corner has to be handled immediately. Only the *levelling* waits out
`POST_CORNER_LOCKOUT_CM`.

In order, every tick:

1. **Wall evidence.** This must run before the colour gate's early return. On
   the first corner the gate does not arm until the car crosses the orange
   line, which on this mat is already past the end of the inner wall —
   counting only after the gate therefore never saw the wall, the open count
   could never rise, and the turn never fired. On an empty mat the car drove
   into the far wall and sat in recovery forever. It only appeared to work
   when a pillar happened to sit to the turn side and stood in for the wall.
2. **Colour gate** — first corner, or the lidar-dead fallback.
3. **Arm the turn**, and take the corner measurements here, because this is
   where the corner really is.
4. **Start the arc** when the wall ahead is close enough.

Exits to `TURNING` or `CORNER_MANEUVER`.

### `TURNING`

Commands `ARC` toward `lane_heading ∓ 90°` at `turn_lock_cmd`. The STM32 runs
the arc's inner loop and reports `ARC_DONE` when it is within `TURN_STOP_DEG`;
the Pi enforces the `TURN_CAP_CM` odometry backstop itself, so a car that
never reaches the heading still leaves the arc.

Handing back early is deliberate: finishing the arc to a fraction of a degree
took ~600 mm of the next straight and left a pillar there no room. The heading
PID finishes the last degrees while the planner is already live.

On exit: lane heading rotates, the corner baseline moves, tracks are cleared —
anything seen mid-turn was in the old lane frame.

### `CORNER_MANEUVER`

The 3-point corner. Four phases: `SWING`, `ARC_FWD`, `STOP`, `ARC_REV`. See
`docs/OBSTACLE_ROUND.md`. On by default, but only entered when an
inner-side pass sits just before the corner AND the corner is planned to exit
inner - i.e. the next straight's first pillar needs the inner side too.

### `FINAL_STRAIGHT`

Drives `L − A` from the end of the last corner, with the full steering law —
pillars can sit in the start section. `A` is the first segment, measured at
the first corner's arm point; `L` is the full start straight, measured at
corners 4 and 8 and averaged.

### `RECOVER`

**The Pi does not drive this state.** The wall panic is the STM32's reflex:
it takes the car when the front reading is short, reverses, and reports
`RECOVERING`. The FSM observes that flag, saves the state it was in, suspends
the planner and the levelling, mutes the floor colour until the car has driven
back past where it started reversing, and restores the saved state when the
flag clears.

The intent it emits meanwhile is the heading it *wants* — ignored by the
firmware while recovering, so that the moment it hands back the car is already
pointing the right way.

### `FINISHED`

Stopped. A Start re-arms and runs a fresh three laps: the firmware's pattern
of `goState(WAIT_START); waitStartStep(); startRequested = true` is kept, so
it takes two ticks and cannot skip the armed state.

## Safety overrides

Checked before the handler runs:

- **Stop from the page** works in every state.
- **No TELEM** → `STOP`. Without it there is no heading and no odometry, and
  there is nothing useful a state machine can do blind.
- **`RECOVERING`** → the overlay above.

## Adding a state

1. Add it to `State`.
2. Write `_your_state(self, c: Ctx) -> ActionIntent`.
3. Add it to the dispatch dict in `step()`.
4. Decide whether the planner and the levelling should be on in it, and set
   `planner_enabled` / `level_enabled` on entry. Sightings taken in a state
   whose lane frame is wrong will poison the track table.
5. Add it to `snapshot()` if the page should show anything new.

## Testing it

`tests/test_fsm.py` drives a fake STM32: the FSM only ever learns the heading,
the odometry and the floor colour through telemetry, so a dataclass with those
three fields is a complete substitute and a test can put the car anywhere.

```
python3 tests/run.py fsm        # or: pytest tests/test_fsm.py
```
