# Pi <-> STM32 protocol — obstacle round

This is the contract. `control/link.py` and `firmware/ObstacleRound.cpp` are
both written against it; if you change one, change this file and the other.

## Who does what

The Pi owns the state machine, the lane planner, the pillar tracks and every
decision about where the car should be. The STM32 owns the loops that cannot
run over a 50 Hz link:

| Runs on the Pi, at frame rate (50 Hz) | Runs on the STM32, at its own rate |
|---|---|
| the seven FSM states | heading PID, IMU rate (~100 Hz) |
| lane offset from the two wall cones | servo slew limit, 2.5 deg/update |
| pillar tracks, pass planning, give-up | the 90 deg arc's inner loop |
| wall levelling of the lane heading | the wall-panic recovery reflex |
| corner trigger, corner-exit shaping | encoder/odometry, IMU, colour sensor |
| the 3-point corner's phase sequencing | servo and motor PWM |

The split follows what the old firmware already did. `updatePlanner()` was
gated on `if (!lidarNewFrame) return;` and `updateLevel()` on `lidarNewRev`,
so both already ran at the Pi's frame rate — moving them up costs no control
bandwidth at all. `updateDriveSteer()` was gated on `gImuFresh` and steps the
servo by at most `SERVO_SLEW` per IMU update; that one cannot move and does
not.

## The port

One serial link, `/dev/ttyACM0`, Blackpill native USB CDC. It carries four
things:

| Direction | What | Framing |
|---|---|---|
| Pi -> STM32 | DRIVE frame | binary, 19 bytes, sync `AA 55` |
| STM32 -> Pi | TELEM frame | binary, 21 bytes, sync `55 AA` |
| both | parameter protocol | ASCII lines, `\n` terminated |
| STM32 -> Pi | `#` log lines | ASCII, `\n` terminated |

The sync words are byte-reversed between directions so a frame can never be
mistaken for one travelling the other way. `0xAA` is not a valid ASCII byte,
so no log line and no parameter line can contain either sync pair: a reader
hunting for `AA` steps over text, and everything it steps over is text. Both
sides use that to demultiplex — see "Demultiplexing" below.

## DRIVE — Pi to STM32, 19 bytes, 50 Hz

Little-endian. The STM32's link clock is the arrival of this frame, not the
LiDAR: a frame with `LIDAR_OK` clear still refreshes the link.

```
 0-1   AA 55     sync
 2     seq       u8    +1 per frame, wraps. Dropped frames are visible.
 3     flags     u8    see below
 4-5   heading   i16   target ABSOLUTE heading, deci-degrees, IMU frame
 6-7   steer     i16   road-wheel command, deci-degrees, + = LEFT
 8     speed     u8    motor PWM magnitude, 0-255. Direction from REVERSE.
 9     arc_lock  u8    lock fraction x100 (ARC mode only), 20-100
10-11  left      u16   mm at 90 deg,  65535 = no return
12-13  front     u16   mm at 0 deg,   65535 = no return
14-15  right     u16   mm at 270 deg, 65535 = no return
16     rev       u8    LiDAR revolution counter, low byte
17     cmd       u8    0 = none, 1 = REBOOT
18     xor8      u8    XOR of bytes 2..17
```

### flags

| bit | name | meaning |
|---|---|---|
| 0x01 | `ENABLE` | 0 = motor off and steering centred, whatever else says |
| 0x02 | `LIDAR_OK` | the Pi's LiDAR is live. Clear = the ranges are not to be trusted; the panic reflex stands down |
| 0x04 | `CAM_OK` | the camera is live |
| 0x08 | `PILLAR_SEEN` | the camera has a pillar right now. Suppresses the panic reflex — a short front reading is that pillar, and the planner is already steering round it |
| 0x30 | `MODE` | 2 bits, see below |
| 0x40 | `REVERSE` | drive backwards at `speed` |
| 0x80 | reserved | must be 0 |

### mode (bits 4-5 of flags)

| value | name | what the STM32 does |
|---|---|---|
| 0 | `STOP` | motor off, steering to `SERVO_TRUE_STRAIGHT`, PID reset |
| 1 | `HEADING_HOLD` | close the heading PID on `heading` at IMU rate. `steer` ignored |
| 2 | `DIRECT` | apply `steer` through the trim and limits, no PID. For the 3-point corner's locked-over legs |
| 3 | `ARC` | the eased 90 deg arc toward `heading`, capped at `arc_lock` of each side's travel. Sets `ARC_DONE` when within `TURN_STOP_DEG` |

`HEADING_HOLD` is what `updateDriveSteer(laneHeading + latYawCmd, true)` used
to be: the Pi has already folded the planner's yaw command into `heading`.

## TELEM — STM32 to Pi, 21 bytes, 50 Hz

```
 0-1   55 AA     sync
 2     seq       u8    +1 per frame, wraps
 3     status    u8    see below
 4-5   heading   i16   deci-degrees, IMU frame, + = left (IMU_YAW_SIGN applied)
 6-7   yaw_rate  i16   deci-degrees/second. Diagnostic; the PID's D term is local
 8-11  odo       i32   encoder ticks, signed, forward counts up. NEVER zeroed —
                       the Pi keeps its own baselines
12-13  servo     i16   the angle actually commanded, deci-degrees
14     floor     u8    0 = none, 1 = orange, 2 = blue
15     tries     u8    consecutive recovery attempts that ended capped
16-19  boot_id   u32   changes on every reset. A new value means the parameter
                       table is back at compiled-in defaults
20     xor8      u8    XOR of bytes 2..19
```

### status

| bit | name | meaning |
|---|---|---|
| 0x01 | `ENABLED` | the motor is being driven |
| 0x02 | `IMU_OK` | a rotation-vector event has arrived recently |
| 0x04 | `COLOUR_OK` | the TCS34725 answered at boot |
| 0x08 | `ARC_DONE` | `ARC` mode has reached `TURN_STOP_DEG` of its target |
| 0x10 | `RECOVERING` | the panic reflex owns the car right now; DRIVE commands are held off |
| 0x20 | `RECOVER_CAPPED` | the last recovery ended on its distance cap, not on a clear reading |
| 0x40 | `LINK_STALE` | no DRIVE frame for `LINK_STALE_MS` |
| 0x80 | `PARAMS_PUSHED` | at least one parameter has been set since boot |

### Odometry is never zeroed

The old firmware called `zeroEncoder()` at each corner and measured with
`absEnc(readEncoder())`. With the FSM on the Pi that would be a race: the Pi
would ask for a zero and not know which frames were taken before it landed.
So `odo` free-runs and the Pi subtracts its own baselines. `TICKS_PER_CM`
lives on the STM32 (it is a hardware fact) and is mirrored up, so the Pi can
convert.

## The recovery reflex

The one behaviour the STM32 still initiates by itself. When `ENABLE` is set,
`LIDAR_OK` is set, `PILLAR_SEEN` is clear, `front <= WALL_PANIC_MM` and the
attempt count is under `RECOVER_MAX_TRIES`, it takes the car: motor off,
servo mirrored about straight, reverse at `speed` until `front >=
WALL_CLEAR_MM` or `RECOVER_MAX_CM` of travel. `RECOVERING` is set throughout.
DRIVE frames keep arriving and keep the link alive, but mode and steering are
ignored until it finishes.

The Pi sees `RECOVERING`, enters its own `RECOVER` state, suspends the planner
and the levelling exactly as the old `enterRecovery()` did, and resumes the
state it was in when the flag clears.

Why it stays local: the trigger is a distance reading the STM32 already has
and the response is one servo write plus one motor write. Routing that through
a 50 Hz link adds a frame of latency to the only reflex whose whole purpose is
to be faster than the crash.

## Parameter protocol — unchanged, table shrunk

Still ASCII, still exactly as it was, because it already works and
`params.py` already mirrors it. What changed is only which parameters exist:
the STM32 keeps 20 that describe the hardware and the fast loops, and
everything else moved to `config.json`, because the Pi is what runs that
logic now.

Of the firmware's 91 entries: 69 became Pi parameters under the same name,
18 stayed, and 4 became Pi parameters under a different one - the three
millisecond timeouts are seconds on the Pi (`COLOR_CONFIRM_MS` ->
`COLOR_CONFIRM_S`, `LIDAR_STALE_MS` -> `LIDAR_STALE_S`, `LIDAR_DEAD_MS` ->
`LIDAR_DEAD_S`), and `LIDAR_MAX_VALID_MM` split into the two range limits
the Pi's own LiDAR code uses, `CONE_MAX_RANGE_MM` and `PILLAR_MAX_MM`. The
twentieth firmware entry, `STEER_LOCK_DEG`, is new - see below.

```
Pi -> STM32
  N <name> <value>    set by name
  P<id> <value>       set by id
  ?P                  dump the table (streamed, PARAM_DUMP_PER_LOOP a pass)
  ?V                  report version / boot id
  C                   recompute derived values

STM32 -> Pi
  !V <ver> <count> <boot>
  !P <id> <name> <type> <val> <lo> <hi> <group>
  !p <id> <val>
  !E <what>
  !C
```

The firmware still has no storage. It boots with compiled-in defaults and a
fresh `boot_id`; the Pi notices the change in the very next TELEM frame — not
on a `?V` round trip, as before — re-reads the table and pushes the saved set
back. A mid-run reset is back on your numbers within a few hundred ms.

### What the STM32 still owns

Twenty entries, down from 91, in four groups:

| group | parameters |
|---|---|
| `drive` (12) | `TICKS_PER_CM`, `SERVO_TRUE_STRAIGHT`, `SERVO_MAX_LEFT`, `SERVO_MAX_RIGHT`, `STEER_LOCK_DEG`, `IMU_YAW_SIGN`, `HEAD_KP`, `HEAD_KI`, `HEAD_KD`, `YAW_FILT_ALPHA`, `SERVO_SLEW`, `INTEGRAL_CLAMP` |
| `turn` (3) | `TURN_KP`, `TURN_MIN_STEER`, `TURN_STOP_DEG` |
| `safety` (4) | `WALL_PANIC_MM`, `WALL_CLEAR_MM`, `RECOVER_MAX_CM`, `RECOVER_MAX_TRIES` |
| `link` (1) | `LINK_STALE_MS` |

`STEER_LOCK_DEG` is new to the table. The Pi sends `DIRECT` steering in
road-wheel degrees, so the firmware needs the lock angle to map that onto this
car's servo travel — the Pi stays in physical units and never learns the trim
or the stops. `COLOR_CONFIRM_MS` went the other way: the STM32 still
classifies the floor colour, but the debounce and the gate decision are the
Pi's, so it is `COLOR_CONFIRM_S` in `config.json` now.

Everything else — `CORRIDOR_MM`, `PASS_*`, `TRACK_*`, `LEVEL_*`, `MNV_*`,
`TURN_DELAY_MM`, `CORNER_EXIT_*`, `SIDE_OPEN_*`, `TARGET_CORNERS` and the
rest — is a Pi parameter in `config.json` now, tuned on the same page, applied
on the next tick with no serial round trip at all.

## Demultiplexing

Both sides read one byte stream carrying binary frames and ASCII lines.

**On the STM32.** A byte-wise state machine. `0xAA` while idle starts a
candidate frame; it then reads exactly 17 more bytes, checks the XOR, and on
failure pushes nothing and returns to hunting. Any other byte while idle is
appended to the ASCII line buffer until `\n`. `0xAA` cannot appear inside a
parameter line, so the two never interleave.

**On the Pi.** `control/link.py` hunts for `55 AA`, takes 21 bytes, checks the
XOR, and hands everything it skipped over to the log/parameter parser as text.
Same technique `percept_link.py` used, kept because it was proven.

## Timing and failure

| Condition | What happens |
|---|---|
| Pi stops sending | `LINK_STALE` after `LINK_STALE_MS`; the STM32 cuts the motor and centres the steering. **This is new** — the old obstacle firmware had no link-loss cut and a car whose Pi died carried on driving |
| LiDAR goes stale | the Pi clears `LIDAR_OK` and holds the last heading; the panic reflex stands down rather than acting on frozen ranges |
| Camera goes stale | the Pi clears `CAM_OK`; tracks age out on distance as they always did |
| TELEM goes stale | the Pi's FSM has no heading or odometry, so it commands `STOP`. There is nothing useful it can do blind |
| Bad XOR either way | the frame is dropped, not acted on. `seq` makes the gap visible |
| `boot_id` changes | the Pi re-reads and re-pushes the parameter table |

The link-loss cut is the one behavioural addition. With the FSM on the Pi, a
dead Pi means a car with no state machine at all, so carrying on is no longer
survivable the way it was when the firmware could drive itself.
