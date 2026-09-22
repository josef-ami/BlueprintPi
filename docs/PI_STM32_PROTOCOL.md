# Pi ↔ STM32 Link — Protocol Spec

**Scope:** the obstacle round only. The open round runs entirely on
`OpenRound.cpp` with the Pi electrically absent, and this protocol does not
apply to it.

**Transport:** Blackpill native USB CDC → `/dev/ttyACM0` on the Pi.
This is a *different namespace* from the lidar's `/dev/ttyUSB0` (a USB-UART
adapter), so the two devices can never collide or swap order at boot.
Baud is nominally 115200 but is ignored by native CDC — it matters only if
you later put a USB-UART bridge on PA9/PA10 instead.

**Who owns what**

| | Pi (`main_obstacle.py`) | STM32 (`ObstacleExecutor.cpp`) |
|---|---|---|
| Perception (camera, lidar, fusion) | ✅ | |
| State machine / strategy | ✅ | |
| Steering geometry (angle ↔ radius) | ✅ | |
| Servo trim, direction, end limits | | ✅ |
| Degrees → microseconds | | ✅ |
| Speed PID on the encoder | | ✅ |
| Heading-hold P loop | | ✅ |
| IMU, ToF, floor colour, button | | ✅ |
| Watchdog / failsafe | | ✅ |

The rule behind that table: anything needing a loop rate faster than the
30 Hz link, and anything that is a mechanical fact, lives on the STM32.
Everything that is a *decision* lives on the Pi.

---

## Frame 1 — DRIVE (Pi → STM32)

11 bytes, sent every Pi control tick (~30 Hz).

| Byte | Field | Type | Meaning |
|---|---|---|---|
| 0 | sync0 | `0xAA` | |
| 1 | sync1 | `0x55` | |
| 2 | `seq` | uint8 | increments per frame, wraps. Echoed in TELEM. |
| 3 | `flags` | uint8 | see below |
| 4–5 | `steer_ddeg` | int16 LE | road-wheel angle × 10, **+ = LEFT** |
| 6–7 | `speed_mmps` | int16 LE | target ground speed, **+ = FORWARD** |
| 8–9 | `heading_ddeg` | int16 LE | absolute target heading × 10 (HEADING_HOLD only) |
| 10 | `xor8` | uint8 | XOR of bytes 2–9 |

**flags byte**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `ENABLE` | 0 → motor off regardless of everything else |
| 1 | `CLOSED_LOOP` | 1 → `speed_mmps` is PID'd against the encoder. 0 → treated as open-loop duty (`speed × 255/700`) |
| 2–3 | `MODE` | 0 = DIRECT, 1 = HEADING_HOLD, 2 = STOP |

**Steer modes**

- **DIRECT** — the Pi computed a road-wheel angle. Used for arcs, avoidance
  swerves, parking shuffles: anything geometric.
- **HEADING_HOLD** — the Pi names a heading; the STM32 closes the loop on its
  own IMU at full rate. `steer_ddeg` is ignored. Used for straights.
  *This exists because a heading loop over a 30 Hz link would be sluggish,
  while the STM32 can run it at IMU rate.*
- **STOP** — motor off, steering centred. Overrides the other fields.

---

## Frame 2 — TELEM (STM32 → Pi)

22 bytes, sent at 50 Hz unconditionally (a steady heartbeat means the Pi can
detect silence, not just stale data).

| Byte | Field | Type | Meaning |
|---|---|---|---|
| 0 | sync0 | `0x55` | reversed vs DRIVE — a frame can't be mistaken for the other direction |
| 1 | sync1 | `0xAA` | |
| 2 | `seq_ack` | uint8 | last DRIVE `seq` accepted |
| 3 | `status` | uint8 | see below |
| 4–7 | `distance_mm` | int32 LE | cumulative signed odometry since boot |
| 8–9 | `speed_mmps` | int16 LE | **measured** ground speed |
| 10–11 | `heading_ddeg` | int16 LE | heading × 10, wrapped ±180, + = left of boot heading |
| 12–13 | `yaw_rate_ddps` | int16 LE | deg/s × 10 |
| 14–15 | `tof_front_mm` | uint16 LE | `0xFFFF` = invalid |
| 16–17 | `tof_left_mm` | uint16 LE | |
| 18–19 | `tof_right_mm` | uint16 LE | |
| 20 | `floor_colour` | uint8 | 0 none, 1 orange, 2 blue |
| 21 | `xor8` | uint8 | XOR of bytes 2–20 |

**status byte**

| Bit | Name | Meaning |
|---|---|---|
| 0 | `ENABLED` | motor is live |
| 1 | `WATCHDOG` | link went silent; motor was cut |
| 2 | `BUTTON` | start button pressed since boot (**latched**, never clears) |
| 3 | `CLOSED_LOOP` | PID engaged |
| 4 | `IMU_OK` | BNO085 initialised |
| 5 | `TOF_OK` | at least one VL53L1X alive |
| 6 | `COLOUR_OK` | TCS34725 found on the mux |

`0xFFFF` on a ToF field decodes to `float('inf')` on the Pi — never a fake
zero, matching the `worldstate.py` convention.

---

## Safety model

Three independent layers, each of which works if the others fail:

1. **STM32 watchdog.** No valid DRIVE frame for **250 ms** → `ENABLE` cleared,
   motor stopped, `WATCHDOG` bit set. This does not depend on the Pi being
   alive, which is the entire point.
2. **Pi-side staleness check.** `FSM.step()` returns a STOP intent if telemetry
   is older than 250 ms or the watchdog bit is set. Catches the reverse case.
3. **Checksum.** Any frame failing `xor8` is dropped silently in both
   directions. A corrupt frame is never acted on; the watchdog handles the
   resulting gap.

On shutdown `main_obstacle.py` sends an explicit all-stop frame
(`ENABLE = 0`) before closing the port, so the car stops on Ctrl-C rather
than coasting until the watchdog trips.

---

## Calibration constants that make the units real

These live in `ObstacleExecutor.cpp`. If they're wrong, the Pi's numbers are
meaningless — the FSM will ask for 400 mm/s and get something else.

| Constant | Value | Status |
|---|---|---|
| `TICKS_PER_CM` | 14.853 | ✅ measured on the current car |
| `SERVO_TRUE_STRAIGHT` | 69.0 | ⚠️ re-measure for the JX PS-1171MG |
| `SERVO_MAX_LEFT` / `RIGHT` | 5.0 / 115.0 | ⚠️ re-measure |
| `STEER_LOCK_DEG` | 35.0 | ✅ CAD verified, FINAL |
| `SPEED_KFF` | 255/700 | derived from top speed |
| `SPEED_KP / KI / KD` | 0.25 / 0.60 / 0 | starting point, tune on the mat |

### Odometry — resolved

**`TICKS_PER_CM = 14.853` is correct and measured** (248.8 ticks/rev over a
5.2 cm wheel, via `02_ticks_per_cm.cpp` on the current car). It is the same
constant `OpenRound.cpp` runs on.

The **31.933** still sitting in `ObstacleRound.cpp` and
`hardware_config.h` is stale — an earlier build with different gearing. Neither
value matches the 0.175 mm/count that `SPECSHEET.md` §4 derives from the
datasheet, which is why the derivation is flagged there as not holding for the
encoder as wired; the measured number wins.

Worth cleaning up in those two files so the repo has one answer, since this
constant scales every commanded speed, all `distance_mm` telemetry, and every
odometry-terminated leg (parking shuffle, avoidance return, corner backstops).

### Servo mapping

```
road-wheel deg (+ = left)  →  servo deg  →  microseconds

left  : servo = TRUE_STRAIGHT + deg × (MAX_LEFT  − TRUE_STRAIGHT) / (+35)
right : servo = TRUE_STRAIGHT + deg × (MAX_RIGHT − TRUE_STRAIGHT) / (−35)
µs    = (servo / 180) × 1000 + 1000
```

Both gains are negative on this linkage — servo angle *decreases* as the road
wheels go left. That's a mechanical fact, not a sign error. Left and right get
separate gains because the linkage is asymmetric about straight (64 servo-deg
of left travel vs 46 of right).

---

## Bring-up order

Do these in order. Each step isolates one failure mode; skipping ahead means
debugging two things at once.

1. **`python3 main_obstacle.py --dry`** — no serial at all. Watch the state
   line, confirm the FSM ticks and perception is sane. Nothing moves.
2. **Flash `ObstacleExecutor.cpp`, run without `--dry`, wheels off the
   ground.** Confirm `link=ok` and telemetry fields look right (heading
   responds when you rotate the car, `odo` counts when you spin a wheel).
3. **`--open-loop`, still on blocks.** Confirm steering direction: a positive
   `steer_ddeg` must turn the wheels **left**. If it's backwards, flip the sign
   of the servo gains — in the *firmware*, not the Pi.
4. **Tune the speed PID.** Command a constant `speed_mmps` and compare against
   the measured value in telemetry. Raise `KP` until it responds, add `KI`
   until steady-state error closes, leave `KD` at 0.
5. **Drop `--open-loop`** and drive.

---

## Debugging by symptom

| Symptom | Where to look |
|---|---|
| `link=STALE` on the Pi | STM32 not sending. Wrong port? Firmware crashed? Check `ls /dev/ttyACM*` |
| Car stops every ~250 ms | Pi tick is too slow or blocking — the watchdog is tripping between frames |
| Steering backwards | Servo gain signs in `setSteerDeg()`. Firmware only. |
| Car drives at the wrong speed | `TICKS_PER_CM` — see the conflict above |
| Speed oscillates | `SPEED_KP` too high, or `SPEED_KI` winding up. Check `S_CLOSED_LOOP` is actually set. |
| Heading drifts on straights | `HEAD_KP`, or the IMU didn't zero (car must be still at boot) |
| `rx_bad` climbing on `Link` | Electrical noise, or both sides disagree on frame length |
| Obstacle distance is `inf` | Lidar has no return at that bearing — a fusion/alignment problem, not a link problem |
