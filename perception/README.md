# perception — map-based localization + semantic belief + live virtual mat

A LiDAR-first perception layer for WRO FE 2026, built on the existing `nav/`
stack. The arena is fully known from the rulebook, so this is **map-based
localization** (correlative scan-matching against a known occupancy matrix) plus
a **semantic layer** (which seats hold pillars, what colour, where the parking
bay is) — not SLAM.

## The idea

1. **The matrix is known.** `nav/geom.py` holds the walls (two concentric
   squares). `nav/arena.py` adds the semantic map: the finite **seat grid** where
   traffic signs may stand, and the **parking bay** candidates.
2. **Superimpose the scan on the matrix → pose.** `nav/localize.py`
   (`DistanceField` + `ScanMatcher`) matches a LiDAR scan against the wall
   likelihood field and returns the car's pose. Pose is estimated from the
   **LiDAR (and camera) alone** — deliberately *without* IMU/encoder. It is
   rough along a straight (the corridor is featureless lengthways) and sharp at
   corners and wherever a mapped pillar is in view. The encoder + IMU are meant
   to **sharpen** this later, not to be required for it (`WorldBelief.predict`).
3. **Search the designated places.** `nav/pillarmap.extract` subtracts the known
   walls from the scan; whatever is left and snaps to a **seat** is a pillar.
   `perception/state.py` keeps a per-seat occupancy belief (present / empty /
   colour) that sharpens every scan.
4. **Camera gives colour.** The camera only has to say *what colour* the thing on
   a given bearing is; the LiDAR already gave its position. (`PillarMap.label`.)
5. **Parking.** `nav/parking.py` finds the two magenta blocks hugging an outer
   wall and returns the bay.

`perception/state.py::WorldBelief` folds all of this into one `snapshot()` — the
object the live dashboard streams and a planner will later read.

## Run it

Everything runs in `--sim` (synthetic RPLidar, no hardware) or `--live` (real
RPLidar C1 on the Pi).

```bash
# Live virtual-mat dashboard (open http://<host>:8080/)
python -m perception.mat_view --sim              # simulated car driving a lap
python -m perception.mat_view --sim --static     # simulated car parked
python -m perception.mat_view --live             # real RPLidar on the Pi

# One-shot static perception report (the on-mat "can it perceive?" test)
python -m perception.static_perceive --sim --guess=-1000,-1000,0
python -m perception.static_perceive --replay scan.npz
python -m perception.static_perceive --live --guess=-1000,-1000,0

# Capture real scans on the Pi for replay/regression
python -m tools.record_scan --out scan.npz --revs 10

# Tests (no pytest needed — uses the repo's tests/run.py)
python tests/run.py arena
python tests/run.py localize
python tests/run.py perception
```

`--guess x,y,deg` seeds which of the four (symmetric) corridors the car is in;
anywhere within ~150 mm is enough.

## Frames & conventions (inherited from the existing stack)

- Mat frame: origin at field centre, **x east, y north, millimetres**.
- LiDAR scan: 360-long array, index = integer degree, 0 = sensor forward,
  increasing anticlockwise; `inf` = no return. World angle of ray *i* at heading
  θ is `radians(i) + θ`.

## Status / next

- ✅ Static + driving perception proven in sim (pose error ~0 mm static; ~mm-level
  while driving where along-track is observable). Live virtual-mat dashboard.
- ⚠️ **Seat grid coordinates in `nav/arena.py` are parametric and must be verified
  against the official field drawing.** Perception never hard-depends on them
  (detections snap to the nearest seat, else are reported free), so refining them
  only sharpens the "which seat" label.
- ⬜ Later: fold in encoder + IMU (`predict`) to sharpen along-track; camera-based
  bearing fusion for pose; then the non-deterministic planner; then STM32.
