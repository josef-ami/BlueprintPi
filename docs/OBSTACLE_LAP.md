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
| `control/supervisor.py` | BlueprintPi | NONE ↔ TRACK lifecycle. No leg, no timer — release is instant. |
| `control/percept_link.py` | BlueprintPi | The two binary frames + the link thread. |
| `obstacle_lap.py` | BlueprintPi | Entry point. Run this, not `main.py`. The loop is class `ObstacleLap`, which `dashboard.py` also runs. |
| `dashboard.py` | BlueprintPi | Calibration tab, plus an **Obstacle run** tab that runs `ObstacleLap` in-process (§10). |
| `config.json` | BlueprintPi | Same file plus a new `avoid` block. |
| `tests/` | BlueprintPi | 76 tests, no hardware needed (the loop's command and floor-filter tests need OpenCV). |

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
   │  pillar clear / out of view      ├─ pillar solved ──> AVOID ──┐
   └──────────────────────────────────┤                            │
                                      ├─ side open ×3 revs ──> TURN90 ──> back
                                      └─ 12 corners + front == start ──> FINISH

   RECOVER overlays any moving state when the front beam drops under 200 mm.
```

AVOID → HEADING is not on a distance or a timer: it fires the instant the Pi
sends `AVOID_NONE`, and the locked side's open-revolution count survives the
trip through AVOID (it is not re-zeroed on that particular re-entry), so a
corner sitting right after a pillar is not missed.

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
right-hand column is what that costs. That is why there are two regimes, both
sent as the same wire action (`TRACK` — there is no separate "committed" value
any more):

- **`r_axis > freeze_mm`**: re-solve every tick, STM32 re-bases its odometry on
  each refresh. The lag corrects itself.
- **`r_axis <= freeze_mm`**: the Pi stops adopting new headings (the geometry
  gets near-degenerate this close) and just holds the last one, but keeps
  sending fresh `TRACK` frames so the STM32 keeps re-basing.

Release back to `AVOID_NONE` happens the instant the pillar is confirmed clear,
or after it has been missing from the camera for a few ticks — never on a
distance or a timer. The old COMMIT phase that ran a leg out on odometry alone,
even with the pillar invisible, is gone: losing sight of it now releases the
manoeuvre instead of running it blind.

---

## 3. Install

```bash
# --- Pi ---
cd ~/BlueprintPi
cp control/solver.py control/supervisor.py control/percept_link.py control/
cp obstacle_lap.py .
cp -r tests .
# config.json: either replace it, or paste the "avoid" block into yours
python3 -m pytest tests/ -q          # 77 passed, before you touch the car

# --- STM32 ---
# src/obstacle_round/ObstacleLap.cpp -> flash by DFU, same toolchain as OpenRound
```

`control/__init__.py` needs no change — the new modules are imported by full
path. `openRound.py` and `OpenRound.cpp` are untouched, so the open round runs
exactly as it does today. `main.py` gained `fuse_with()` (what `fuse()` now calls)
and `sensors/camera.py`'s `detect_blobs()` optional `masks_out`, `floor`,
`rejected_out` and `context_out`. `fuse_with()` changes nothing for existing
callers. The camera thread now passes `floor=` from `config.json`, so **every**
caller's blobs go through the floor filter (4.4a); set `"floor": {"enabled":
false}` to get the old behaviour back.

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

Hyderabad's hall lighting is not your workshop's, and it is not even uniform
across the mat — the same pillar reads a different HSV from different spots, so
one threshold picked from one angle clips it elsewhere.

Use the **eyedropper** on the Calibration tab for this. Pick the colour
(Red / Green / Magenta), press **Pick from feed** (it switches to the Raw view
and stays armed), then click that colour's pillar in the feed **from several
positions on the mat** — near and far, and in the brightest and dimmest spots
you expect. Each click adds a sample (shown as a chip) and widens the mask to
span every sample so far, dropping the S/V floor to cover the dimmer views and
stretching the hue window across the warmer/cooler ones; gross outliers (a
stray floor pixel caught in a click) are trimmed so one bad click can't blow
the band open. **Undo last** / **Clear** manage the samples; **Esc** or the
button again stops. The sliders still show and fine-tune the result, and
**Save to config.json** writes it as before. Then confirm `min_blob_area`
(300) still rejects noise but keeps a pillar at 2 m — at 2 m a 50 mm pillar is
small.

**Do this again on competition day, in the competition hall.** It is the single
most common reason a working obstacle round stops working at an event — and the
multi-spot sampling is exactly what makes the mask survive the hall's uneven
light.

### 4.4a Floor / pillar filter — only red and green that stand on the mat

A red banner, a green exit sign or someone's shirt beyond the walls passes the
HSV test just like a pillar. `sensors/camera.detect_blobs` therefore keeps a
colour blob only if it **stands on our floor**:

1. **Contact strip.** Take the rows just under the blob (`strip_px` tall,
   central 60 % of its width). At least `min_white_frac` of those pixels must
   be white mat (saturation ≤ `white_s_max`, brightness ≥ `white_v_min`).
   A pillar whose base is below the bottom of the image is kept
   (*base below view*) — it is close, and it is certainly on the floor.
2. **Linked to the floor in front of the car** (`linked`). That white must be
   connected to the white floor at the bottom of the image without crossing a
   wall-dark pixel (brightness ≤ `dark_v_max`). A white card on the far side
   of the black wall passes test 1 but fails this one (*wall between*).
   Orange/blue corner lines are not dark, so they do not cut the floor.

`ignore_bottom_px` blanks the bottom rows out of the floor seed if the car's
own body or bumper shows at the bottom of the frame.

Everything that uses `detect_blobs` — the calibration views, `/api/worldstate`,
`obstacle_lap.py` and the dashboard's run tab — uses the filter, with the
`"floor"` block of `config.json`. Tune it on the calibration tab:

- **Floor filter** view: blue = white floor linked to the car, purple = white
  cut off by the wall, each blob's contact strip in green (pass) or red (fail),
  the ignored bottom rows hatched.
- **Blobs** view: kept pillars as before; rejected colour blobs as dashed grey
  boxes labelled *not on floor* or *wall between*.
- **Floor / pillar filter** section: on/off, the link check on/off, and the six
  sliders. *Save* writes them to `config.json` like the HSV bands.

Under venue light, first raise `white_v_min` until the **Floor filter** view
shows the mat solid blue with the walls not blue, then check every pillar on
the mat has a green strip at 0.3 m, 1 m and 2 m. The **Mask** view on the
calibration tab stays the raw HSV threshold (it is what the HSV sliders act
on); the run tab's RED/GREEN masks are pillar-only, next to a white-floor tile.

The filter also drops a far pillar whose base is hidden behind a wall, e.g.
seen over the inner wall across the corner. That one is in another section and
out of the lidar's reach anyway.

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

**1 — Geometry, on a laptop.** `python3 -m pytest tests/ -q`. 76 passed. If the
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
- inside ~350 mm the heading stops changing, but it is still `TRCK` on the
  status line — there is no separate committed state any more
- hide the pillar → after a few ticks (`LOST_GRACE_TICKS`, default 3) the line
  drops back to `----` and the note says "pillar out of view - released",
  instantly, with no distance run out

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
| `freeze_mm` | config | 350 | heading twitches near the pillar | holding a heading before the geometry has settled |
| `tail_clear_mm` | config | 150 | the tail clips on the way out | the car overshoots past the pillar |
| `confirm_ticks` | config | 3 | false colour detections start manoeuvres | reacting too slowly |
| `refractory_mm` | config | 150 | it re-locks the pillar it just passed | it misses a genuine second pillar |
| `wall_margin_mm` | config | 90 | still touching walls | it refuses passes that were fine |
| Speed (base) | dashboard | 70 | too slow to be competitive | overshooting solves / missing corners. Live slider on the Obstacle run tab; AVOID follows at 60/70 of it. `BASE_SPEED`/`AVOID_SPEED` in the .cpp are only the power-up defaults now. |
| `POST_AVOID_LOCKOUT_CM` | .cpp | 15 | phantom turns right after an avoid | — (corner progress from before the avoid is preserved automatically now; this only covers PID settle time, so it rarely needs raising) |
| `SIDE_OPEN_MM` | .cpp | 1500 | phantom turns mid-straight | corners missed |

---

## 7. Debugging by symptom

| Symptom | Where to look |
|---|---|
| Car never starts, LED1 slow blink | The Pi is not sending. `ls /dev/ttyACM*`. Is `obstacle_lap.py` running? |
| `# WARN camera down` at start | Camera thread died — check the `[CameraThread] fatal:` line above it |
| A pillar on the mat is not detected, but the Mask view shows it | Floor filter (4.4a): **Blobs** view says why (*not on floor* / *wall between*); **Floor filter** view shows whether the mat reads white |
| Obstacle range is always `inf` | Lidar plane height (4.1) or camera↔lidar alignment (4.3). Not a link problem. |
| Steers the wrong way past pillars | Firmware servo mapping. The solver tests already cover the sign. |
| Phantom turn mid-straight | `SIDE_OPEN_MM`, or a turn fired during/just after an avoid — raise `POST_AVOID_LOCKOUT_CM` |
| Corner missed after a pillar | Should not happen any more — the open-revolution count from before the avoid now survives it. If it still does, check `# avoid capped`/`# avoid timed out` (the Pi's TRACK frames stopped arriving fresh, so the backstop fired instead of an instant release) and consider lowering `POST_AVOID_LOCKOUT_CM` |
| Car swerves then straightens too early | `tail_clear_mm` |
| Car finishes in the wrong place | `# FINISH by odo` in the log means the front beam was invalid at the start |
| Heading drifts on straights | `HEAD_KP`, or the IMU did not zero — the car must be still at boot |
| `# avoid capped` / `# avoid timed out` | Should not fire in normal operation — release is the Pi's call (`AVOID_NONE`), not a distance or a timer. Seeing this means TRACK frames stopped arriving fresh (link stale, or the Pi crashed/froze) and the last-resort backstop caught it. Check the link, not `engage_mm`. |
| Everything stops for ~250 ms at a time | The Pi tick is blocking. Check `q` — a climbing lidar queue means the consumer is falling behind. |

---

## 8. Protocol reference

`/dev/ttyACM0`, Blackpill native USB CDC. A different namespace from the lidar's
`/dev/ttyUSB0`, so they cannot collide or swap at boot. The firmware's `#` log
lines share the port; `0xAA` is not ASCII, so a log line can never contain the
TELEM sync word and the Pi separates the two cleanly.

### PERCEPT — Pi → STM32, 17 bytes, 50 Hz, sync `AA 55`

| Byte | Field | Notes |
|---|---|---|
| 2 | `seq` | echoed in TELEM |
| 3 | `flags` | b0 LIDAR_OK, b1 CAM_OK, b2–3 action (0 NONE / 1 TRACK / 2 COMMIT — wire-legal, but the Pi never sends 2 any more), b4 colour (0 RED / 1 GREEN), b5 HELLO |
| 4–9 | `left`, `front`, `right` | uint16 mm, `0xFFFF` = no return |
| 10 | `rev` | lidar revolution, low 8 bits |
| 11–12 | `target_heading` | int16, deg × 10, **absolute**, + = left |
| 13–14 | `leg_remaining` | uint16 mm — a backstop cap now (`BACKSTOP_LEG_MM`, 1500), not a distance to reach; see §2 |
| 15 | `cmd` | 0 NONE, 1 RERUN, 2 STOP, 3 REBOOT |
| 16 | `base_speed` | uint8 PWM for the straights (0 = unset → firmware keeps its default/last). Clamped to `[SPEED_MIN, SPEED_MAX]` = [40, 150]; AVOID keeps the same fraction of it the defaults have (60/70). Set live from the dashboard's Speed slider. |
| 17 | `xor8` | over bytes 2–16 |

**CMD (byte 15)**

| Value | Taken when | Effect | The Pi holds it until |
|---|---|---|---|
| 1 RERUN | a 0 → 1 edge while FINISH or STOPPED | back to BOOT: HELLO, 5 s countdown, finish line re-measured at GO | TELEM shows BOOT |
| 2 STOP | 3 consecutive good frames, any state but FINISH/STOPPED | motor off, steering centred → STOPPED | TELEM shows STOPPED (or FINISH) |
| 3 REBOOT | 3 consecutive good frames | motor off, `# REBOOT from Pi`, then `NVIC_SystemReset()`; USB drops and re-enumerates, `setup()` re-zeroes the yaw | — (the Pi sends 6 and closes the port) |

A frame sent only to carry a command has `LIDAR_OK` clear and no ranges, so it
refreshes the STM32's link clock but never its lidar clock, and it repeats the
current `rev` so it cannot count as a new revolution.

### STATUS — STM32 → Pi, 61 bytes, 10 Hz, sync `55 A5`

Reporting only — nothing on either side decides anything from it. It carries the
FSM internals TELEM does not, for the dashboard's state-machine panel. Firmware
without it still works with everything else; `PerceptLink.status()` stays `None`.

| Byte | Field | Byte | Field |
|---|---|---|---|
| 2 | version (1) | 32–33 | lidar L mm (`0xFFFF` far) |
| 3 | state | 34–35 | lidar R mm (`0xFFFF` far) |
| 4 | flags: b0 FSM_STARTED, b1 BOOT_READY, b2 LIDAR_HOLD, b3 LINK_STALE, b4 BLIND, b5 RERUN_ARMED, b6 WALL_SEEN_L, b7 WALL_SEEN_R | 36–37 | straight mm (HEADING) |
| 5 | flags2: b0 REAL_OPEN_L, b1 REAL_OPEN_R, b2 LOCK_NEEDS_REAL_RETURN | 38–39 | post-corner lockout mm left |
| 6 | last PERCEPT: b0 HELLO, b1 LIDAR_OK, b2 CAM_OK, b3 GREEN, b4–5 action | 40–41 | post-avoid lockout mm left |
| 7 | last CMD byte | 42–43 | segment mm (since GO / last turn) |
| 8 | frames in a row carrying it | 44–45 | odo-fallback finish target mm |
| 9 | run number (GOs since power-up) | 46–47 | front at start mm (`0xFFFF`) |
| 10–11 | ms in state | 48–49 | phase mm: AVOID travelled / TURN90 arc / RECOVER backed |
| 12–13 | BOOT countdown ms left | 50–51 | AVOID leg mm (backstop cap now, not a target — see §2) |
| 14–15 | BOOT camera grace ms left | 52–53 | ms since last PERCEPT (`0xFFFF` never) |
| 16–17 | AVOID ms elapsed | 54–55 | ms since last LIDAR_OK frame (`0xFFFF` never) |
| 18–19 | lane heading, int16 deg × 10 | 56–59 | PERCEPT frames received, uint32 |
| 20–21 | target heading (HEADING lane / AVOID solved / TURN90 arc), int16 × 10 | 60 | `xor8` over bytes 2–59 |
| 22–23 | servo command, int16 servo-deg × 10 | | |
| 24–25 | motor PWM, int16 | | |
| 26–27 | open revolutions L, R | | |
| 28–29 | recover tries, state RECOVER returns to | | |
| 30 | finish reason: 0 –, 1 wall, 2 odo, 3 STOP command | 31 | reserved (0) |

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
| 16 | `state` (BOOT 0 … RECOVER 5, STOPPED 6) |
| 17 | `corner_count` |
| 18–19 | `avoid_remaining_mm` uint16 |
| 20 | `floor_colour` (always 0: this firmware does not read the colour sensor) |
| 21 | `xor8` over bytes 2–20 |

`tests/test_wire.py` re-implements the C-side halves (PERCEPT parsing and the
CMD rules, TELEM and STATUS packing) from the offsets written in
`ObstacleLap.cpp` and checks they agree with the Python codec. **If you change a
frame, change it in three places** — the `.cpp`, `percept_link.py`, and that
test — and run the tests.

### Safety layers (each works if the others fail)

1. **Lidar staleness** (STM32, 200 ms / 1000 ms). Stale stops the car acting on
   old distances; dead parks the car in HEADING until the lidar is back. A
   frame carrying a command but no ranges (`LIDAR_OK = 0`) cannot clear the
   stale flag — otherwise `0xFFFF` side readings would look like an open corner.
2. **Link staleness** (STM32, 250 ms). A stale link stops TRACK from re-basing
   its odometry, which is what lets the backstop cap/timeout below engage
   instead of holding a frozen heading with no one watching.
3. **AVOID backstops** (STM32): 150 cm of travel or 5 s, whichever comes
   first — last resort only. Normal release is entirely the Pi's call
   (`AVOID_NONE`, sent the instant the pillar is clear or out of view); these
   only fire if TRACK frames stop arriving fresh.
4. **Wall panic** (STM32): front under 200 mm → reverse on mirrored steering,
   bounded to 3 attempts.
5. **Checksum**, both directions. A corrupt frame is dropped, never acted on;
   STOP and REBOOT additionally need 3 identical frames in a row.
6. **STOP** (Pi → STM32, CMD 2). There is **no link-loss motor cut**: if the Pi
   just goes quiet, the car carries on under its own logic. The dashboard's
   *Stop car* and *End session* (and stopping `robodash.service`) send STOP and
   wait for STOPPED; the CLI's Ctrl-C only closes the port.

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
- **A pillar sitting right before a corner.** Fixed: the locked side's
  open-revolution count now survives a trip through AVOID (see §1's FSM note),
  so the corner that was building up before the pillar interrupted is not lost.
  `POST_AVOID_LOCKOUT_CM` (15 cm by default — re-tune on the mat) still guards
  the moment right after release so a still-yawed heading can't misfire the
  corner trigger.

---

## 10. Running it from the dashboard

`dashboard.py` → **Obstacle run** tab (`http://<pi>:8080/#run`). It runs the
same `ObstacleLap` loop the CLI runs, in the dashboard process, on the camera
and lidar the dashboard already owns, so there is nothing to stop first.

| Control | What it does |
|---|---|
| **Start** | Reads `config.json` from disk (unsaved slider changes are *not* used, as with the CLI), opens the port and starts the loop. `--dry`, `--no-avoid` and `--port` are the CLI's options. |
| **Stop car** | Holds CMD STOP until TELEM shows STOPPED. The loop keeps running. |
| **Rerun** | FINISH or STOPPED only: holds CMD RERUN until TELEM shows BOOT. |
| **End session** | STOP first (1 s to be acknowledged), then the loop stops and the port is released. The car is left STOPPED: next time, Start then Rerun, or Reboot. |
| **Reboot STM32** | Click twice. STOP, end the session, 6 × CMD REBOOT, close the port, then watch the USB device drop and come back. The yaw re-zeroes at boot, so keep the car still. |
| **Speed** | The straight-line PWM (40–150), sent in every frame (PERCEPT byte 16) and applied **live** — drag it mid-run and the firmware re-asserts it on the next tick, no restart. AVOID speed follows as a fixed fraction (60/70 of base); TURN90 and RECOVER are unaffected. Held in the session only (not written to `config.json`); each Start begins at the default 70. |

The page shows the camera with the run's own detections (rejected blobs dashed
grey, see 4.4a), its pillar-only RED/GREEN masks and the white-floor mask, the same lidar/obstacle/fusion views as the calibration tab (computed
with the run's config), the STM32 state machine (from TELEM + STATUS) with
every condition the current state is waiting on, the Pi avoidance supervisor,
the CLI's status line as live fields, and a console with everything the CLI
would print (the status line once a second, as with `--quiet`).

The open-round stream toggle on the calibration tab and a run cannot hold
`/dev/ttyACM0` at the same time; the dashboard refuses the second one.
