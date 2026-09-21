# BluePrint Pi

Everything that lives on the Pi 5 of Team Blueprint's bot for WRO 2026 APAC,
plus the STM32 firmware it talks to.

## Layout

```
dashboard.py              the page: run, tune and calibrate, on :8080
openRound.py              the open challenge - a LiDAR feeder, nothing else
params.py                 what is tunable, and the only writer of config.json
worldstate.py             the conventions, and what the sensor threads publish
config.json               one file: params, stm32, intrinsics, device paths

sensors/camera.py         pillar detection and bearings
sensors/lidar.py          beams, wall cones, pillar range, candidates

control/fsm.py            the obstacle round's state machine
control/planner.py        lane position, pillar tracks, the pass planner
control/intent.py         ActionIntent - what the car should do
control/mapper.py         ActionIntent -> physically legal
control/link.py           the DRIVE/TELEM wire
control/obstacle_round.py the 50 Hz loop

firmware/ObstacleRound.cpp  the STM32 executor
firmware/sim/             stub Arduino headers: builds the firmware for a PC
calibration/              fisheye calibration, and the Lab colour fit

docs/OBSTACLE_ROUND.md    how the car gets round
docs/FSM_GUIDE.md         the state machine, state by state
docs/PI_STM32_PROTOCOL.md the wire, and who owns what
```

## Setup

```bash
sudo apt install -y python3-libcamera python3-kms++ python3-picamera2
python3 -m venv --system-site-packages env
source env/bin/activate
pip install -r requirements.txt
```

## Running

One process may hold the camera and the LiDAR at a time.

```bash
python3 dashboard.py                      # the page, http://<pi>:8080
python3 -m control.obstacle_round         # the same loop, headless
python3 -m control.obstacle_round --dry   # decide and print, send nothing
python3 openRound.py                      # the open challenge
```

### As a daemon

```bash
sudo cp robodash.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now robodash.service
```

## The page

Three tabs, one process — so you can calibrate while watching what the
detector makes of it, which the three separate pages this replaced could not
do.

**Run** — the camera with the detector's boxes on it, Start/Stop, the FSM's
state, the lane and the pillars it is tracking, the STM32's telemetry, the
console. Grey dashed boxes are rejected blobs, labelled with which test said
no: `A` too flat, `S` ragged, `F` not on the mat, `L` mat not linked to the
floor in front, `C` too little colour.

**Tune** — every tunable, live. Pi values apply on the next tick; STM32 values
are queued and pushed a few per loop, so tuning can never stall the 50 Hz
feed. Save writes `config.json`.

**Calibrate** — click the mat and the pillars to fit Lab colour ranges. Lab
because HSV gates on saturation, and a matte pillar under dim light falls
under the threshold and vanishes — green first, since red has two hue bands
and survives longer. Also where `AREA_K` is measured.

## Where the driving happens

The Pi owns the state machine, the lane planner and every decision about where
the car should be. The STM32 owns the loops a 50 Hz link cannot close: the
heading PID at IMU rate, the 90° arc's inner loop, the wall-panic reflex,
odometry and the sensors.

This is a change: the firmware used to own the FSM and the planner, and the
Pi was a sensor pipe. The logic was ported rather than rewritten, and the
split follows what the old firmware had already chosen — its planner and its
levelling were both gated on the Pi's frame arriving, so they were running at
50 Hz already.

**One behavioural addition:** the firmware now cuts the motor if the Pi goes
quiet. The old one deliberately did not, because a car whose Pi died could
carry on under its own state machine. It has no state machine now.

## Tests

```bash
python3 tests/run.py            # no dependencies beyond numpy and cv2
python3 tests/run.py planner    # or a subset
pytest -q                       # if pytest is installed
```

299 tests covering the detector, the LiDAR geometry, the planner, the FSM,
both wire frames, the mapper and the parameter store. They run with no robot
attached — the Pi-only libraries are stubbed and nothing under test calls into
them, which is the point of keeping the geometry and the state machine free of
hardware.

`tests/test_firmware_sim.py` compiles `firmware/ObstacleRound.cpp`
**unmodified** with g++ against the stub headers in `firmware/sim/`, and
drives it over its real wire: DRIVE frames from `control.link.pack_drive`,
TELEM decoded by `control.link`, parameter lines parsed by
`params.StmParams`. The last two tests close the loop — the Pi's FSM driving
the real firmware round twelve corners of a kinematic lap, both directions,
with the firmware's servo and motor moving the car and its TELEM as the FSM's
only view of it. That proves the logic and both halves of the protocol
together. It says nothing about the hardware: the IMU, encoder, colour
sensor, servo and motor are stubs that do exactly what they are told.
Skipped if there is no g++.

```bash
sh firmware/sim/build.sh      # just the build, -Wall, into firmware/sim/
```
