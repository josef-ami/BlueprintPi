# Obstacle round — lap only

Three laps, red/green pillar avoidance, no parking. This document is the whole
procedure: where the files go, what to measure, what to flash, and what to tune,
in an order where each step can only fail for one reason.

---

## 1. What was built

| File | Repo | Role |
|---|---|---|
| `src/obstacle_round/ObstacleLap.cpp` | WRO_TeamBluePrint | STM32 firmware. `OpenRound.cpp` + one state. |
| `control/solver.py` | BlueprintPi | The avoidance geometry. Pure functions. |
| `control/supervisor.py` | BlueprintPi | TRACK → COMMIT → IDLE lifecycle. |
| `control/percept_link.py` | BlueprintPi | The two binary frames + the link thread. |
| `obstacle_lap.py` | BlueprintPi | Entry point. Run this, not `main.py`. |
| `config.json` | BlueprintPi | Same file plus a new `avoid` block. |
| `tests/` | BlueprintPi | 38 tests, no hardware needed. |

`ObstacleExecutor.cpp` is **not** used. Its pin map (PB9/PB8 motor, TIM3
encoder, IMU on PB5/PB4/PB3) is from an earlier car and does not match this
hardware. Leave it alone or delete it, but do not flash it.

### Division of work

Anything with a deadline shorter than the 50 Hz link, or that is a mechanical
fact, is on the STM32. Anything that is a decision about the world is on the Pi.

| Pi | STM32 |
|---|---|
| Camera blobs → colour + bearing | The state machine |
| Lidar ranges, fusion, pillar tracking | Heading PID at IMU rate |
| **The avoidance solve** | The eased 90° arc |
| Side/front distances (feeds the corner trigger) | The corner trigger decision |
| | Odometry, servo trim, the watchdog |

### States (STM32)

```
BOOT ──Pi says lidar + camera are up──> 5 s countdown ──┐
                                                         v
   ┌──────────────────────────────> HEADING <────────────┘
   │                                  │
   │  leg done                        ├─ pillar solved ──> AVOID ──┐
   └──────────────────────────────────┤                            │
                                      ├─ side open ×3 revs ──> TURN90 ──> back
                                      └─ 12 corners + front == start ──> FINISH

   RECOVER overlays any moving state when the front beam drops under 200 mm.
```

---

## 2. The geometry, in one box

A pillar at bearing `b` and lidar range `r_face` defines a keep-out cone:

```
r_axis  = r_face + 25                       lidar sees the FACE, not the axis
d_clear = car_half_width + 25 + margin      = 57.1 + 25 + 40 = 122 mm
delta   = asin(d_clear / r_axis)            half-angle of the keep-out cone

already clear  ⟺  RED and b >= +delta       (pillar ends up on our LEFT)
                  GREEN and b <= -delta      (pillar ends up on our RIGHT)

theta = b + s·delta        s = +1 GREEN, -1 RED     heading offset, + = left
leg   = sqrt(r_axis² - d_clear²) + tail_clear
```

`delta` answers both questions — whether to act, and by how much — so there is
no separate skip test. `theta` is the minimum deviation: a pillar that is nearly
clear gets a small correction, not a full swerve.

| `r_axis` | `delta` | `leg` (before tail) | turn-in cost `R·delta` |
|---|---|---|---|
| 900 mm | 7.8° | 892 mm | 26 mm |
| 700 mm | 10.0° | 689 mm | 34 mm |
| 500 mm | 14.1° | 485 mm | 48 mm |
| 350 mm | 20.4° | 328 mm | 69 mm |
| 250 mm | 29.2° | 218 mm | 99 mm |

The tangent solve assumes the heading changes instantly; it does not, and the
right-hand column is what that costs. That is why the manoeuvre has two phases:

- **TRACK** (`r_axis > freeze_mm`): re-solve every tick, STM32 re-bases its
  odometry on each refresh. The lag corrects itself.
- **COMMIT** (`r_axis <= freeze_mm`, or the pillar left the frame): freeze the
  heading, run the leg out on odometry. This is what carries the car past a
  pillar the camera can no longer see, and it survives a lidar dropout.

---

## 3. Install

```bash
# --- Pi ---
cd ~/BlueprintPi
cp control/solver.py control/supervisor.py control/percept_link.py control/
cp obstacle_lap.py .
cp -r tests .
# config.json: either replace it, or paste the "avoid" block into yours
python3 -m pytest tests/ -q          # 38 passed, before you touch the car

# --- STM32 ---
# src/obstacle_round/ObstacleLap.cpp -> flash by DFU, same toolchain as OpenRound
```

`control/__init__.py` needs no change — the new modules are imported by full
path. Nothing in `openRound.py`, `main.py`, `dashboard.py` or `OpenRound.cpp` is
modified, so the open round still runs exactly as it does today.

---

## 4. Calibration — do these in this order

### 4.1 Lidar scan-plane height — **check this first**

The pillars are 100 mm tall. Measure from the mat to the middle of the C1's
optical window. **If it is at or above ~95 mm, nothing else in this document
matters** — the beam passes over every pillar and all fused ranges come back
`inf`. Shim the lidar down or raise the pillars' apparent height in your bench
test to confirm you have the margin.

Confirm at the extremes too: a pillar at 2 m must still return, and one at
300 mm must not be occluded by your own chassis.

### 4.2 Car half width

Measure extreme-to-extreme at the widest point of the car — tyres, mirrors of
the chassis, anything that sticks out — not the track width. CAD says 114.24 mm.

```json
"car_half_width_mm": <measured> / 2
```

This feeds `d_clear` directly. Getting it wrong by 10 mm moves every avoidance
path by 10 mm.

### 4.3 Camera ↔ lidar rotational alignment

The fusion attaches a lidar range to a camera bearing. If the two are rotated
relative to each other by more than `fusion.bearing_match_deg` (8°), fusion
silently attaches the **wall's** range to the pillar and the solver produces a
confident, wrong answer.

```bash
python3 obstacle_lap.py --dry
```

Put a pillar 1 m dead ahead of the lidar's 0°, on the centreline. Read the
`near_s` field in the status line (`R@+02/0985`, say).

- **Bearing should read ≈ 0.** If it reads `+6`, set
  `"camera_offset_deg": -6` in `config.json` and re-check.
- **Range should match your tape measure** to a few cm. If it reads the wall
  behind instead, the bearing is off or `fusion.gap_split_mm` is too large.

Repeat at ±30° and ±45°. The fisheye undistortion should hold the error flat
across the field; if it grows toward the edges, re-run
`calibration/capture_chessboard.py` + `calibrate_fisheye.py`.

### 4.4 HSV under venue light

Hyderabad's hall lighting is not your workshop's. Run `dashboard.py`, put a red
and a green pillar in frame at 0.3 m, 1 m and 2 m, and widen the `hsv` ranges
until both hold solid at all three. Then check `min_blob_area` (300) still
rejects the noise but keeps a pillar at 2 m — at 2 m a 50 mm pillar is small.

**Do this again on competition day, in the competition hall.** It is the single
most common reason a working obstacle round stops working at an event.

### 4.5 Floor colour thresholds

Unchanged from the open round (`pR > 52 && pB < 18` → orange,
`pB > 23 && pR < 40` → blue). If you re-seat the TCS34725 or change its height,
re-run `src/tools/calibration/08_floor_colour_thresholds.cpp`.

### 4.6 Carried over — verify, do not re-derive

These are already measured on this car and are used unchanged. Re-measure only
if that part of the car moved.

| Constant | Value | Re-measure with |
|---|---|---|
| `TICKS_PER_CM` | 14.853 | `02_ticks_per_cm.cpp` |
| `SERVO_TRUE_STRAIGHT` | 76.5 | `04_true_straight_servo.cpp` |
| `SERVO_MAX_LEFT / RIGHT` | 20 / 140 | steering end stops |
| `HEAD_KP` | 2.0 | `07_heading_gain.cpp` |
| turn arc (`TURN_KP/KV/...`) | as in the open round | `06_turn_90.cpp` |
| `SIDE_OPEN_MM` | 1500 | see 4.7 |

### 4.7 Side-open threshold

The corridor is 1000 mm wide, so a side beam can never read more than ~1000 mm
mid-straight; past the end of the inner wall it looks down the next straight,
well beyond 1500 mm. Verify by pushing the car along a straight with
`obstacle_lap.py --dry` running and watching `L` and `R` — the turn-side number
should jump from under 1000 to over 2000 at the corner, with nothing in between.

Note the trigger now counts **lidar revolutions** (`SIDE_OPEN_REVS = 3`), not
frames. Each bearing only gets one new sample per revolution (~10 Hz on the C1),
so counting frames at 50 Hz would "confirm" on a single measurement resent.

### 4.8 `margin_mm` — the only genuinely new number

Start at 40. It is the slack between the car's flank and the pillar at the
closest point of the pass.

- Car clips pillars → raise it.
- Car runs wide into the wall on pillars near a wall → lower it, and check the
  wall guard is on (`"wall_guard": true`).

At `margin_mm = 40`, `d_clear = 122 mm`. A pillar sitting 250 mm from a wall
leaves `250 − 25 − 57 = 168 mm` of room on the near side, so 122 fits with
46 mm to spare. A `d_clear` of 154 would leave 14 mm — which is why the
clearance is half the car width plus half the pillar, not the whole car width.

---

## 5. Bring-up ladder

Each rung isolates one failure. Do not skip.

**1 — Geometry, on a laptop.** `python3 -m pytest tests/ -q`. 38 passed. If the
sign conventions are wrong this is where you find out, not on the mat.

**2 — Perception, car stationary, no firmware.**
```bash
python3 obstacle_lap.py --dry
```
Walk a red pillar across the field of view. Watch:
- bearing goes **positive as it moves left** (the whole convention rests on this)
- range tracks your tape measure
- `TRCK` appears at ~900 mm and the commanded heading is **negative for RED**,
  positive for GREEN
- `CMIT` appears at ~350 mm and the heading stops changing
- hide the pillar → still `CMIT`, heading frozen

**3 — Link, wheels off the ground.** Flash `ObstacleLap.cpp`, run without
`--dry`. Expect `# colour CH4 READY`, `# zero yaw`, `# first PERCEPT frame
received`, `# Pi ready - 5s countdown`. The status line should show `ok`, the
heading should move when you rotate the car, `odo` should count when you spin a
wheel. `q` (lidar queue depth) must hover near 0.

**4 — Steering direction, still on blocks.** Hold a RED pillar in front. The
front wheels must turn **right**. GREEN → left. If they are backwards, the sign
is wrong in the **firmware** servo mapping, not in the solver — the tests in
step 1 already proved the solver.

**5 — One pillar on a straight, on the mat.** Motor on, one pillar, then the
same pillar in the other colour, then near each wall. Tune `margin_mm`.

**6 — One section, then a full lap, then three.** Watch `# TURN n/12` in the log
and that `# avoid released` always precedes the turn.

**7 — The finish.** Confirm the car stops within ~10 cm of where it started. The
log says which rule fired: `# FINISH by wall` (good) or `# FINISH by odo` (the
fallback — means the front beam was invalid, worth investigating).

---

## 6. Tuning reference

| Knob | Where | Default | Raise it when | Lower it when |
|---|---|---|---|---|
| `margin_mm` | config | 40 | clipping pillars | running wide into walls |
| `engage_mm` | config | 900 | reacting too late | chasing pillars in the next section |
| `freeze_mm` | config | 350 | heading twitches near the pillar | committing before the geometry settles |
| `tail_clear_mm` | config | 150 | the tail clips on the way out | the car overshoots past the pillar |
| `confirm_ticks` | config | 3 | false colour detections start manoeuvres | reacting too slowly |
| `refractory_mm` | config | 150 | it re-locks the pillar it just passed | it misses a genuine second pillar |
| `wall_margin_mm` | config | 90 | still touching walls | it refuses passes that were fine |
| `AVOID_SPEED` | .cpp | 60 | too slow to finish the leg | overshooting the solve |
| `POST_AVOID_LOCKOUT_CM` | .cpp | 30 | phantom turns after an avoid | real corners missed after an avoid |
| `SIDE_OPEN_MM` | .cpp | 1500 | phantom turns mid-straight | corners missed |

---

## 7. Debugging by symptom

| Symptom | Where to look |
|---|---|
| Car never starts, LED1 slow blink | The Pi is not sending. `ls /dev/ttyACM*`. Is `obstacle_lap.py` running? |
| `# WARN camera down` at start | Camera thread died — check the `[CameraThread] fatal:` line above it |
| Obstacle range is always `inf` | Lidar plane height (4.1) or camera↔lidar alignment (4.3). Not a link problem. |
| Steers the wrong way past pillars | Firmware servo mapping. The solver tests already cover the sign. |
| Phantom turn mid-straight | `SIDE_OPEN_MM`, or a turn fired during/just after an avoid — raise `POST_AVOID_LOCKOUT_CM` |
| Corner missed after a pillar | `POST_AVOID_LOCKOUT_CM` too large, or the avoid leg overran — check `# avoid capped` |
| Car swerves then straightens too early | `tail_clear_mm` |
| Car finishes in the wrong place | `# FINISH by odo` in the log means the front beam was invalid at the start |
| Heading drifts on straights | `HEAD_KP`, or the IMU did not zero — the car must be still at boot |
| `# avoid timed out` | The Pi held TRACK for 5 s: it is seeing a pillar it never gets close to. Check `engage_mm` vs the fused range. |
| Everything stops for ~250 ms at a time | The Pi tick is blocking. Check `q` — a climbing lidar queue means the consumer is falling behind. |

---

## 8. Protocol reference

`/dev/ttyACM0`, Blackpill native USB CDC. A different namespace from the lidar's
`/dev/ttyUSB0`, so they cannot collide or swap at boot. The firmware's `#` log
lines share the port; `0xAA` is not ASCII, so a log line can never contain the
TELEM sync word and the Pi separates the two cleanly.

### PERCEPT — Pi → STM32, 16 bytes, 50 Hz, sync `AA 55`

| Byte | Field | Notes |
|---|---|---|
| 2 | `seq` | echoed in TELEM |
| 3 | `flags` | b0 LIDAR_OK, b1 CAM_OK, b2–3 action (0 NONE / 1 TRACK / 2 COMMIT), b4 colour (0 RED / 1 GREEN), b5 HELLO |
| 4–9 | `left`, `front`, `right` | uint16 mm, `0xFFFF` = no return |
| 10 | `rev` | lidar revolution, low 8 bits |
| 11–12 | `target_heading` | int16, deg × 10, **absolute**, + = left |
| 13–14 | `leg_remaining` | uint16 mm |
| 15 | `xor8` | over bytes 2–14 |

### TELEM — STM32 → Pi, 22 bytes, 50 Hz, sync `55 AA`

| Byte | Field |
|---|---|
| 2 | `seq_ack` |
| 3 | `status` — b0 RUNNING, b1 LIDAR_STALE, b2 LIDAR_DEAD, b3 DIR_LOCKED, b4 CLOCKWISE, b5 IMU_OK, b6 COLOUR_OK, b7 RECOVERING |
| 4–7 | `odo_mm` int32, cumulative, **never reset** |
| 8–9 | `speed_mmps` int16 |
| 10–11 | `heading` int16, deg × 10 |
| 12–13 | `yaw_rate` int16, deg/s × 10 |
| 14–15 | `front_mm` uint16, `0xFFFF` invalid |
| 16 | `state` (BOOT 0 … RECOVER 5) |
| 17 | `corner_count` |
| 18–19 | `avoid_remaining_mm` uint16 |
| 20 | `floor_colour` (0 none, 1 orange, 2 blue) |
| 21 | `xor8` over bytes 2–20 |

`tests/test_wire.py` re-implements both C-side halves from the offsets written
in `ObstacleLap.cpp` and checks they agree with the Python codec. **If you change
a frame, change it in three places** — the `.cpp`, `percept_link.py`, and that
test — and run the tests.

### Safety layers (each works if the others fail)

1. **Lidar staleness** (STM32, 200 ms / 1000 ms). Stale stops the car acting on
   old distances; dead disables the lidar corner trigger and falls back to the
   colour gate. A frame carrying a committed leg but no ranges (`LIDAR_OK = 0`)
   cannot clear the stale flag — otherwise `0xFFFF` side readings would look
   like an open corner.
2. **Link staleness** (STM32, 250 ms). A stale link stops TRACK from re-basing,
   so a frozen heading expires on its own leg distance instead of running
   forever.
3. **Leg backstops** (STM32): 150 cm and 5 s, whichever comes first.
4. **Wall panic** (STM32): front under 200 mm → reverse on mirrored steering,
   bounded to 3 attempts.
5. **Checksum**, both directions. A corrupt frame is dropped, never acted on.

---

## 9. Known limits

- **Two pillars in a line.** The supervisor engages the nearest and holds a
  150 mm refractory afterwards. A second pillar closer than that behind the
  first will be picked up late. Raise `refractory_mm` only if you see re-locking.
- **A pillar tight against the wall on its legal side.** The wall guard reduces
  `theta` to fit, and refuses (holds the lane) if there is genuinely no room —
  taking the penalty rather than the crash. Watch for `no room on the legal
  side` in the log.
- **The final straight.** Avoidance is allowed there, so path length no longer
  equals straight-line distance; the finish uses the front-wall match for
  exactly that reason, with the `L − A` encoder arithmetic only as a fallback.
- **Parking is not implemented.** `FINISH` is a full stop, rule 9.24.2.
