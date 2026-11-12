// Lidar-triggered corners with IMU heading hold - ESP32 DevKit V1, RPLIDAR C1,
// SparkFun BNO08x (SPI), 2x BTS7960. Non-blocking state machine. Single file.
//
//   State 1  S_FORWARD     drive forward holding the lane heading, watch the lidar.
//                          First open reading on the inner side -> slow to CORNER_SPEED and remember the
//                          front reading (= width of the cross corridor).
//            S_APPROACH    corner confirmed: creep on at CORNER_SPEED until the middle of the cross corridor
//   State 2  S_TURN_RIGHT  90 degree PD turn right, turn counter + 1, back to State 1
//   State 3  S_TURN_LEFT   90 degree PD turn left,  turn counter + 1, back to State 1
//   State 4  S_FINAL       drive until the front (270 deg) reading matches the one taken at the start,
//                          creep forward/back to align, then stop and wait for the button
//
//   State 1 -> APPROACH : lidar 0 deg (right) / 180 deg (left) > WALL_MM + OPEN_MARGIN_MM for OPEN_READINGS
//                         readings in a row, on a side that has first been seen as a wall on this straight
//   APPROACH -> 2 / 3   : front <= half the cross corridor width (+ CORNER_CENTER_MARGIN_MM), then a
//                         short brake (BTS7960 both inputs HIGH, never reverses). If the pivot point is already reached when the corner is
//                         confirmed, it stops at once.
//   State 1 -> 4        : turn counter reached TARGET_TURNS (3 laps x 4 turns)
//
// The lidar is mounted rotated 90 degrees on the robot: 0 deg = robot's RIGHT, 180 deg = LEFT,
// 270 deg = FRONT. Angles increase clockwise, so 270..290 is front-right and 250..270 is front-left.
//
// ON THE STRAIGHTS the heading target is laneHeading + offset (offset > 0 = to the right), and the IMU
// heading hold steers to it. The offset comes from, in priority order:
//   1. SWERVE: wide front reading < FRONT_STOP_MM for FRONT_CLOSE_READINGS revolutions -> +/-SWERVE_DEG toward
//      the front sector with the larger average. Ends after FRONT_CLOSE_READINGS clear readings in a row, or
//      after SWERVE_MAX_MS. Front < FRONT_DANGER_MM -> back up (RECOVER) instead.
//   2. WALL CENTERING: both side walls seen -> CENTER_K * (right - left), capped at CENTER_MAX_DEG.
//      A side inside the lidar blind zone (reads 0) while the other side is a wall -> full offset away from it.
//
// SAFETY
//   - stuck watchdog: front reading not changing while driving -> RECOVER (back up, retry), RECOVER_MAX_TRIES
//   - IMU watchdog: no yaw report for IMU_TIMEOUT_MS -> pause
//   - lidar watchdog: no revolution for LIDAR_TIMEOUT_MS -> pause
//   - turn stall: no yaw progress -> duty boosted step by step, then pause
//
// BUTTON (PB14)
//   waiting / finished -> press: start a fresh run (robot still for START_SETTLE_MS first)
//   running            -> press: PAUSE (motors off)
//   paused             -> press: RESUME where it stopped (laps kept). Failures also pause instead of ending.
//
// Logging: USB serial only, 115200 baud.
//
// IMU wiring:   SCK 18, MISO 19, MOSI 23, CS 22, INT 21, RST 13
// Motors:       left RPWM/LPWM 26/25, right 27/33
// Lidar (C1):   lidar TX -> GPIO16 (RX2), lidar RX -> GPIO17 (TX2), 5 V, GND
// Button 14, LED 4.

#include <Arduino.h>
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

#define PIN_SCK 18
#define PIN_MISO 19
#define PIN_MOSI 23
#define PIN_CS 22
#define PIN_INT 21
#define PIN_RST 13

const int PIN_BTN = 14;   // start button to GND (internal pull-up, pressed = LOW)
const int PIN_LED = 4;    // LED: blinks = ready, solid = running, fast blink = paused

const int L_FWD = 26, L_BACK = 25;
const int R_FWD = 27, R_BACK = 33;

// State types live up here, before any function, so the Arduino IDE's auto-generated prototypes can use them.
enum State {
  S_WAIT_BTN,       // blinking, waiting for a press
  S_SETTLE_START,   // robot still for START_SETTLE_MS (start front measured), then State 1
  S_FORWARD,        // State 1
  S_APPROACH,       // corner seen: drive on to the middle of the cross corridor
  S_STOP_PAUSE,     // stopped before a turn
  S_TURN_RIGHT,     // State 2
  S_TURN_LEFT,      // State 3
  S_POST_TURN,      // stopped after a turn
  S_FINAL,          // State 4
  S_RECOVER,        // backing up after being stuck / too close
  S_PAUSED          // button or failure: motors off, next press resumes
};

// ============================================================
// TUNING
// ============================================================

// ---- driving and turning. All duties are 3x your earlier values.
const int DRIVE_SPEED = 90;            // 0..255 PWM duty on a straight (was 30)
const int CORNER_SPEED = 60;           // duty from the first open reading until the pivot (20 x 3)
const bool YAW_RISES_TURNING_RIGHT = false;  // if the first turn runs away, the sketch aborts and tells you to flip this
const float TURN_DEG = 90.0f;
const float KP = 9.0f;                 // turn: duty per degree of error (was 3)
const float KD = 1.05f;                // turn: duty per (degree/second) (was 0.35)
const int TURN_MAX = 60;               // was 20
const int TURN_MIN = 30;               // was 10
const float DEADBAND_DEG = 10.0f;
const float STOP_RATE_DPS = 20.0f;
const uint32_t TURN_TIMEOUT_MS = 8000;

// ---- turn stall boost: if the pivot is not moving, add duty step by step (your TURN_MIN/MAX stay the base)
const uint32_t TURN_PROGRESS_MS = 500;  // check the yaw progress this often
const float    TURN_PROGRESS_DEG = 4.0f;// less than this in TURN_PROGRESS_MS (8 deg/s) = too slow: boost
const int      TURN_BOOST_STEP = 15;    // duty added per check that saw no progress (was 5)
const int      TURN_BOOST_MAX = 150;    // most duty the boost may add on top of TURN_MAX (was 50)
const uint32_t TURN_STALL_MS = 1500;    // still no progress this long at full boost -> pause

// ---- heading hold on the straights
const float HEAD_KP = 3.0f;            // duty per degree of heading error. NOT scaled with speed: the turn rate
                                       //   a duty difference gives does not depend on forward speed (x3 oscillated)
const float HEAD_CORR_MAX = 120.0f;    // largest left/right duty difference the hold may add (was 40)
const float HEAD_SLEW = 5.0f;          // largest change of the correction per NEW IMU sample (not scaled, see above)

// ---- corner detection (State 1 -> APPROACH)
const uint16_t WALL_MM = 500;          // a side reading above this (+ margin) means the wall has opened
const uint16_t OPEN_MARGIN_MM = 100;   // "C": extra margin added to WALL_MM before a side counts as open
const uint8_t  OPEN_READINGS = 2;      // "N": readings in a row above the threshold before it counts as open.
                                       //      (was 3; the robot is already at CORNER_SPEED from the first one)
                                       //      One reading = one lidar revolution (about 100 ms).
const uint8_t  WALL_SEEN_READINGS = 3; // a side must first read as a wall this many revolutions in a row on the
                                       //   current straight before it may "open" (stops false corners)
const uint32_t LOCKOUT_BASE_MS = 300;  // minimum forward time after a turn before a corner may fire.
const uint32_t LOCKOUT_EXTRA_MS = 0;   //   Backstop only: the wall-seen rule above does the real work, at any speed.
const uint32_t LOCKOUT_MS = LOCKOUT_BASE_MS + LOCKOUT_EXTRA_MS;

// ---- robot geometry
const uint16_t NOSE_MM = 70;                 // lidar centre to the front of the robot
const uint16_t REAR_STOP_MM = 150;           // !!! SET FOR YOUR ROBOT: lidar centre to the back + margin.
                                             //     Reversing stops when the rear (90 deg) reads closer.

// ---- braking before a pivot: BTS7960 short brake (both inputs HIGH -> both high-side switches on,
//      the motor terminals are shorted together). It stops the wheels but can never drive them backward.
//      Needs R_EN / L_EN held HIGH (enabled). Afterwards both inputs go LOW, which also brakes (low side).
const uint32_t BRAKE_MS = 150;               // how long the short brake is held (0 = no explicit brake)

// ---- corner seen ahead in the scan (speed-independent)
// Every revolution, the inner wall is followed through the front quadrant of the scan (points whose lateral
// distance is within CORNER_WALL_TOL_MM of the side reading). Where it stops is the end of the inner wall:
//   cornerAhead = how far ahead of the lidar the inner wall ends
//   cross width = front - cornerAhead  (the cross corridor lies between the wall end and the wall ahead)
const uint16_t CORNER_WALL_TOL_MM = 40;      // a scan point is on the inner wall if its lateral distance is this close
const uint16_t CORNER_SLOW_AHEAD_MM = 250;   // slow to CORNER_SPEED when the inner wall ends closer than this ahead
const uint16_t CROSS_MIN_MM = 150, CROSS_MAX_MM = 2000;   // plausible cross corridor widths

// ---- corner approach (pivot in the middle of the cross corridor)
const int16_t  CORNER_CENTER_MARGIN_MM = 0;  // pivot when front <= corridor width / 2 + this
const uint32_t APPROACH_MAX_MS = 4000;       // pivot anyway after this long

// ---- run length
const int TARGET_TURNS = 12;           // 3 laps x 4 turns, then State 4

// ---- State 4: stop where the run started (front reading match)
const uint16_t FRONT_DEADBAND_MM = 15;       // done when |front - start front| <= this
const uint16_t FRONT_SLOW_MM = 150;          // stop driving and start aligning this far before the target
const int      ALIGN_PWM = 75;               // creep duty, forward and back (was 25)
const uint32_t ALIGN_SETTLE_MS = 300;        // stopped this long before a reading counts
const uint32_t ALIGN_PULSE_FIRST_MS = 40;    // align moves in timed pulses; the first one is this long,
const uint32_t ALIGN_PULSE_MIN_MS = 15;      //   later ones are sized from how far the previous pulse moved
const uint32_t ALIGN_PULSE_MAX_MS = 200;
const uint32_t ALIGN_TIMEOUT_MS = 8000;      // give up aligning, stop where it is
const uint32_t FINAL_MAX_MS = 20000;         // State 4 drive backstop: start aligning after this long anyway
const uint32_t FINAL_MS = 1000;              // only if no start front was recorded: drive this long and stop

// ---- collision avoidance: front swerve
const uint16_t FRONT_ANG = 2700;              // front direction, 0.1 degree units (270 deg)
const float    FRONT_HALF_DEG = 2.0f;         // narrow front (start/stop position, corner approach, stuck check)
const float    FRONT_WIDE_HALF_DEG = 10.0f;   // wide front (swerve / danger): nearest point within +/- this
const uint16_t FRONT_STOP_MM = 200;           // wide front closer than this = too close
const uint16_t FRONT_DANGER_MM = NOSE_MM + 100; // wide front closer than this (nose 100 mm from the wall):
                                              //   at a corner -> pivot now, elsewhere -> back up
const uint8_t  FRONT_CLOSE_READINGS = 2;      // revolutions in a row: close to start a swerve, clear to end it
const float    SECTOR_DEG = 20.0f;            // width of each front sector: right = 270..290, left = 250..270
const float    SWERVE_DEG = 5.0f;             // swerve: heading offset from the lane heading
const uint32_t SWERVE_HOLDOFF_MS = 1000;      // a new swerve cannot start sooner than this after the last one started
const uint32_t SWERVE_MAX_MS = 1500;          // a swerve ends after this long even if the front is still close

// ---- wall centering on the straights
const float   CENTER_K = 0.06f;               // degrees of heading offset per mm of (right - left) difference
const float   CENTER_MAX_DEG = 10.0f;         // cap on the centering offset
const uint8_t SIDE_ZERO_READINGS = 2;         // side reads 0 (blind zone) this many revolutions -> steer away
const float   CENTER_SLEW_DEG = 1.0f;         // largest change of the centering offset per lidar revolution
const float   CENTER_VALID_DEG = 15.0f;       // side beams only trusted while within this of the lane heading

// ---- stuck watchdog + recovery
const uint32_t STUCK_MS = 1500;               // front changed less than STUCK_MM for this long while driving = stuck
const uint16_t STUCK_MM = 15;
const int      RECOVER_PWM = 90;              // reverse duty (was 30)
const uint32_t RECOVER_MS = 800;              // how long to back up
const int      RECOVER_MAX_TRIES = 3;         // in a row without completing a turn -> pause

// ---- watchdogs
const uint32_t LIDAR_STALE_MS = 250;          // no new lidar revolution for this long while driving: hold still
const uint32_t LIDAR_TIMEOUT_MS = 1000;       // no new lidar revolution for this long while running: pause
const uint32_t IMU_TIMEOUT_MS = 200;          // no new yaw report for this long while running: pause

// ---- start position (front reading) - the hand on the button must not be measured
const uint16_t START_STABLE_MM = 30;          // while waiting: front (median of 5) steady within this ...
const uint8_t  START_STABLE_READINGS = 10;    //   ... for this many revolutions (1 s) = the start front
const uint16_t START_CLEAR_MM = 30;           // after the press: front back within this of the start front = clear
const uint32_t START_CLEAR_TIMEOUT_MS = 6000; // not clear for this long -> do not start

// ---- yaw rate
const uint32_t YAW_RATE_WINDOW_MS = 40;       // rate measured over at least this long (IMU reports arrive in bursts)

// ---- timings
const uint32_t BTN_DEBOUNCE_MS = 25;
const uint32_t BTN_LOCKOUT_MS = 400;   // ignore a second press this soon after one
const uint32_t START_SETTLE_MS = 1000; // after the press: front clear and robot still this long, then go
const uint32_t STOP_PAUSE_MS = 400;    // stopped before a turn
const uint32_t TURN_FINISH_MS = 150;   // motors off this long before a turn counts as done
const uint32_t POST_TURN_MS = 300;     // stopped after a turn, before State 1 again
const uint32_t TURN_PRINT_MS = 250;    // while turning, print yaw this often

// +1: yaw rises when turning right.  -1: yaw falls when turning right (clockwise negative).
const float SIGN = YAW_RISES_TURNING_RIGHT ? 1.0f : -1.0f;

// ============================================================
// LIDAR  (RPLIDAR C1 on Serial2 - from the supplied snippet)
// ============================================================
HardwareSerial& L = Serial2;
const int LIDAR_RX = 16, LIDAR_TX = 17;
const uint32_t LIDAR_BAUD = 460800;

const int MAX_PTS = 900;                 // the C1 gives about 510 per revolution at 10 rev/s
struct Pt { uint16_t ang, dist; };       // ang in 0.1 degree, dist in mm (0 = no return)
Pt pts[MAX_PTS];
int nPts = 0;

uint8_t pk[5]; int pn = 0;
bool scanning = false;
uint32_t lastData = 0, lastTry = 0;

// latest result, one per lidar revolution (0 = no valid return in the window)
uint16_t lidarRightMm = 0, lidarLeftMm = 0;   // 0 deg = right, 180 deg = left
uint16_t lidarFrontMm = 0;                    // 270 deg, narrow window (FRONT_HALF_DEG)
uint16_t lidarFrontWideMm = 0;                // 270 deg, wide window (FRONT_WIDE_HALF_DEG)
uint16_t lidarRearMm = 0;                     // 90 deg = back of the robot
float    secRightAvg = 0, secLeftAvg = 0;     // average distance in the front-right / front-left sectors
bool     secRightAny = false, secLeftAny = false;   // false = no valid return in that sector
bool     lidarNewRev = false;                 // true only on the loop a revolution finished
bool     haveRev = false;
uint32_t lidarRevMs = 0;

uint16_t frontHist[5] = {0, 0, 0, 0, 0};      // last 5 narrow front readings (0 = no return)

// inner wall end seen in the last full scan, per side (computed when the revolution completes)
bool     scanCornerR = false, scanCornerL = false;
int16_t  scanAheadR = 0, scanAheadL = 0;      // the wall ends this far ahead of the lidar (mm)
uint16_t scanCrossR = 0, scanCrossL = 0;      // cross corridor width (mm)
bool cornerFromScan(bool rightSide, uint16_t sideMm, uint16_t frontMm, int16_t &aheadMm, uint16_t &crossMm);
uint8_t  frontHistIdx = 0;

void sendCmd(uint8_t cmd) { uint8_t p[2] = {0xA5, cmd}; L.write(p, 2); L.flush(); }
void flushRx() { while (L.available()) L.read(); }

bool startScan() {
  sendCmd(0x25); delay(40); flushRx();    // STOP first, in case a scan is already running
  sendCmd(0x20);                          // SCAN
  uint8_t d[7]; int i = 0; uint32_t t = millis();
  while (i < 7 && millis() - t < 2000) {  // hunt for the response descriptor A5 5A 05 00 00 40 81
    if (!L.available()) continue;
    uint8_t b = L.read();
    if (i == 0 && b != 0xA5) continue;
    if (i == 1 && b != 0x5A) { i = (b == 0xA5) ? 1 : 0; continue; }
    d[i++] = b;
  }
  if (i < 7 || d[2] != 0x05 || d[6] != 0x81) return false;
  nPts = 0; pn = 0;
  lastData = millis();
  return true;
}

// Distance (mm) of the closest valid point within +/-halfWidth (0.1 degree units) of targetAng.
// Returns 0 if there is no valid return in that window.
uint16_t distAt(uint16_t targetAng, uint16_t halfWidth = 20) {   // default +/-2.0 degrees
  uint16_t best = 0;
  for (int i = 0; i < nPts; i++) {
    if (!pts[i].dist) continue;
    int diff = abs((int)pts[i].ang - (int)targetAng);
    if (diff > 1800) diff = 3600 - diff;                         // wrap-around at 0/360
    if (diff <= halfWidth && (best == 0 || pts[i].dist < best)) best = pts[i].dist;
  }
  return best;
}

// Average distance (mm) of the valid points with angle in (lo, hi] when rightSide, [lo, hi) otherwise
// (0.1 degree units, no wrap needed around 270). Returns false if the sector has no valid point.
bool sectorAvg(uint16_t lo, uint16_t hi, bool rightSide, float &avg) {
  uint32_t sum = 0; int n = 0;
  for (int i = 0; i < nPts; i++) {
    if (!pts[i].dist) continue;
    uint16_t a = pts[i].ang;
    bool in = rightSide ? (a > lo && a <= hi) : (a >= lo && a < hi);
    if (in) { sum += pts[i].dist; n++; }
  }
  if (!n) return false;
  avg = (float)sum / n;
  return true;
}

void publishRevolution() {
  lidarRightMm = distAt(0);        // 0 deg   = robot's right
  lidarLeftMm  = distAt(1800);     // 180 deg = robot's left
  lidarFrontMm = distAt(FRONT_ANG, (uint16_t)(FRONT_HALF_DEG * 10));
  lidarFrontWideMm = distAt(FRONT_ANG, (uint16_t)(FRONT_WIDE_HALF_DEG * 10));
  lidarRearMm = distAt(900, 100);  // 90 deg = back, nearest within +/-10 deg
  uint16_t sec = (uint16_t)(SECTOR_DEG * 10);
  secRightAny = sectorAvg(FRONT_ANG, FRONT_ANG + sec, true,  secRightAvg);   // 270..290 = front-right
  secLeftAny  = sectorAvg(FRONT_ANG - sec, FRONT_ANG, false, secLeftAvg);    // 250..270 = front-left
  uint16_t openMm = WALL_MM + OPEN_MARGIN_MM;     // corner geometry needs the full scan: do it now
  scanCornerR = cornerFromScan(true,  lidarRightMm <= openMm ? lidarRightMm : 0, lidarFrontMm, scanAheadR, scanCrossR);
  scanCornerL = cornerFromScan(false, lidarLeftMm  <= openMm ? lidarLeftMm  : 0, lidarFrontMm, scanAheadL, scanCrossL);
  frontHist[frontHistIdx] = lidarFrontMm;
  frontHistIdx = (frontHistIdx + 1) % 5;
  lidarNewRev = true;
  haveRev = true;
  lidarRevMs = millis();
}

// Median of the valid readings among the last 5 narrow front readings; 0 if fewer than 3 are valid.
uint16_t frontMedian() {
  uint16_t v[5]; int n = 0;
  for (int i = 0; i < 5; i++) if (frontHist[i]) v[n++] = frontHist[i];
  if (n < 3) return 0;
  for (int i = 1; i < n; i++)
    for (int j = i; j > 0 && v[j] < v[j - 1]; j--) { uint16_t t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
  return v[n / 2];
}

// End of the inner wall on one side, from the current scan. The wall is followed point by point from beside
// the robot (90 deg from the front) toward the front; the first point that jumps away from the wall line marks
// its end. rightSide: lidar 360 -> 275 (front-right), else lidar 180 -> 265 (front-left).
// sideMm = distance to that wall (0 deg / 180 deg reading). Fills aheadMm (how far ahead the wall ends) and
// crossMm (front - aheadMm) and returns true if a plausible corner was found.
bool cornerFromScan(bool rightSide, uint16_t sideMm, uint16_t frontMm, int16_t &aheadMm, uint16_t &crossMm) {
  if (!sideMm || !frontMm) return false;
  bool onWall = false, ended = false;
  float lastFwd = 0;
  for (int k = 0; k < nPts; k++) {
    int i = rightSide ? (nPts - 1 - k) : k;                      // scan order: angles increase through the array
    if (!pts[i].dist) continue;
    int a = pts[i].ang;                                          // 0.1 deg
    float th;                                                    // angle from the front toward the side, deg
    if (rightSide) { if (a < 2750 || a > 3600) continue; th = (a - 2700) * 0.1f; }
    else           { if (a > 2650 || a < 1800) continue; th = (2700 - a) * 0.1f; }
    float r = th * (PI / 180.0f);
    float lat = pts[i].dist * sinf(r), fwd = pts[i].dist * cosf(r);
    if (fabsf(lat - sideMm) <= CORNER_WALL_TOL_MM) { onWall = true; lastFwd = fwd; }
    else if (onWall && fwd > lastFwd) { ended = true; break; }   // left the wall line further ahead: the wall ended
  }
  if (!onWall || !ended) return false;
  int ahead = (int)lastFwd;
  int cross = (int)frontMm - ahead;
  if (cross < (int)CROSS_MIN_MM || cross > (int)CROSS_MAX_MM) return false;
  aheadMm = (int16_t)ahead;
  crossMm = (uint16_t)cross;
  return true;
}

void serviceLidar() {
  lidarNewRev = false;
  if (!scanning) {
    if (millis() - lastTry > 3000) {
      lastTry = millis();
      scanning = startScan();
      if (!scanning) Serial.println("lidar not responding, retrying...");
    }
    return;
  }

  while (L.available()) {
    pk[pn++] = L.read();
    if (pn < 5) continue;
    bool S = pk[0] & 1, nS = (pk[0] >> 1) & 1, C = pk[1] & 1;
    if (S != nS && C) {                                          // valid 5-byte sample
      uint8_t quality = pk[0] >> 2;
      uint16_t angQ6 = (((uint16_t)pk[2] << 8) | pk[1]) >> 1;    // degrees * 64
      uint16_t distQ2 = ((uint16_t)pk[4] << 8) | pk[3];          // mm * 4
      if (S) {                                                   // start of a new revolution
        if (nPts >= 20) publishRevolution();                     // skip the partial first one
        nPts = 0;
      }
      if (nPts < MAX_PTS) {
        pts[nPts].ang = (uint16_t)((angQ6 * 10UL + 32) / 64);
        pts[nPts].dist = (quality > 0) ? (distQ2 + 2) / 4 : 0;
        nPts++;
      }
      lastData = millis();
      pn = 0;
    } else {                                                     // lost sync: slide by one byte
      memmove(pk, pk + 1, 4); pn = 4;
    }
  }

  if (millis() - lastData > 3000 && millis() - lastTry > 3000) { // data stopped: restart the scan
    lastTry = millis();
    scanning = startScan();
  }
}

bool lidarOk() { return haveRev && millis() - lidarRevMs <= LIDAR_TIMEOUT_MS; }

// ============================================================
// IMU
// ============================================================
BNO08x imu;
float yaw = 0;                         // degrees, game rotation vector (no magnetometer, drifts slowly)
bool haveYaw = false;
bool yawFresh = false;                 // true only on the loop a NEW rotation-vector report arrived
float yawRate = 0;                     // deg/s, filtered
uint32_t yawT = 0;                     // time of the last yaw report
float    rateRefYaw = 0;               // yaw rate is measured from here ...
uint32_t rateRefT = 0;                 //   ... once at least YAW_RATE_WINDOW_MS has passed
float wrap(float a);

void updateYaw() {
  yawFresh = false;
  if (imu.wasReset()) { Serial.println("IMU reset, re-enabling reports"); imu.enableGameRotationVector(20); }
  for (int n = 0; n < 10 && imu.getSensorEvent(); n++) {
    if (imu.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      float qi = imu.getQuatI(), qj = imu.getQuatJ(), qk = imu.getQuatK(), qr = imu.getQuatReal();
      uint32_t nowT = millis();
      yaw = atan2f(2.0f * (qr * qk + qi * qj), 1.0f - 2.0f * (qj * qj + qk * qk)) * RAD_TO_DEG;
      if (!haveYaw) { rateRefYaw = yaw; rateRefT = nowT; }
      else if (nowT - rateRefT >= YAW_RATE_WINDOW_MS) {      // reports 1-2 ms apart would inflate the rate
        float inst = wrap(yaw - rateRefYaw) * 1000.0f / (nowT - rateRefT);
        yawRate = 0.5f * yawRate + 0.5f * inst;
        rateRefYaw = yaw; rateRefT = nowT;
      }
      yawT = nowT; haveYaw = true;
      yawFresh = true;
    }
  }
}

bool imuOk() { return haveYaw && millis() - yawT <= IMU_TIMEOUT_MS; }

float wrap(float a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

// ============================================================
// MOTORS
// ============================================================
void side(int fwdPin, int backPin, int v) {
  if (v >= 0) { analogWrite(backPin, 0); analogWrite(fwdPin, v); }
  else        { analogWrite(fwdPin, 0);  analogWrite(backPin, -v); }
}
void motors(int l, int r) { side(L_FWD, L_BACK, l); side(R_FWD, R_BACK, r); }

// Short brake: both inputs of both drivers HIGH. No direction, so it cannot reverse the robot.
void motorsBrake() {
  analogWrite(L_FWD, 255); analogWrite(L_BACK, 255);
  analogWrite(R_FWD, 255); analogWrite(R_BACK, 255);
}

// ============================================================
// BUTTON  (non-blocking debounce; true once per press)
// btnStable starts as LOW ("pressed") on purpose: the first debounced edge is then a RELEASE,
// so a button held at power-up can never start the robot.
// ============================================================
bool     btnLastRaw = LOW, btnStable = LOW;
uint32_t btnChangeMs = 0, btnLastPressMs = 0;
bool     btnPressedOnce = false;

bool buttonPressed() {
  bool raw = digitalRead(PIN_BTN);
  if (raw != btnLastRaw) { btnLastRaw = raw; btnChangeMs = millis(); }
  if (raw != btnStable && millis() - btnChangeMs >= BTN_DEBOUNCE_MS) {
    btnStable = raw;
    if (btnStable == LOW) {
      if (btnPressedOnce && millis() - btnLastPressMs < BTN_LOCKOUT_MS) return false;
      btnPressedOnce = true;
      btnLastPressMs = millis();
      return true;
    }
  }
  return false;
}

// ============================================================
// STATE MACHINE
// ============================================================

const char* stateName(State s) {
  switch (s) {
    case S_WAIT_BTN: return "wait"; case S_SETTLE_START: return "settle"; case S_FORWARD: return "State 1";
    case S_APPROACH: return "approach"; case S_STOP_PAUSE: return "stop before turn";
    case S_TURN_RIGHT: return "State 2 (turn right)"; case S_TURN_LEFT: return "State 3 (turn left)";
    case S_POST_TURN: return "after turn"; case S_FINAL: return "State 4"; case S_RECOVER: return "recover";
    case S_PAUSED: return "paused";
  }
  return "?";
}

State    state = S_WAIT_BTN;
uint32_t stateT0 = 0;

float laneHeading = 0.0f;              // ideal heading of the current leg
float corr = 0.0f;                     // current left/right duty difference (slew-limited)

int      turnCount = 0;
int      pendingDir = 0;               // +1 right, -1 left: the turn that follows the approach
bool     sideLocked = false;           // after the first turn only that side is watched
bool     lockedRight = true;
uint32_t forwardT0 = 0;                // when the current State 1 leg began

// corner detection, per side
uint8_t  openCntR = 0, openCntL = 0;   // readings in a row above the open threshold
uint8_t  wallCntR = 0, wallCntL = 0;   // readings in a row as a wall
bool     armedR = false, armedL = false;   // side has been seen as a wall on this straight
const int WIDTH_HIST = 20;              // corridor width = smallest right + left sum over the last 20 revolutions (2 s)
uint16_t widthHist[WIDTH_HIST];        //   (the readings just before a corner opens are inflated, never smaller)
uint8_t  widthIdx = 0;

void widthReset() { for (int i = 0; i < WIDTH_HIST; i++) widthHist[i] = 0; widthIdx = 0; }

// Record right + left when both are walls; returns the smallest recent sum (0 = none yet).
uint16_t corridorWidth() {
  uint16_t best = 0;
  for (int i = 0; i < WIDTH_HIST; i++) if (widthHist[i] && (!best || widthHist[i] < best)) best = widthHist[i];
  return best;
}

// corner approach
uint16_t approachTargetMm = 0;         // pivot when the front reaches this (0 = pivot at once)
uint32_t approachT0 = 0;
bool     cornerSlow = false;           // corner close / side opening: CORNER_SPEED, no swerve
uint16_t crossWidthMm = 0;             // cross corridor width measured from the scan (0 = unknown)
int16_t  cornerAheadMm = -1;           // inner wall ends this far ahead (-1 = not seen this revolution)
uint16_t crossMem[4] = {0, 0, 0, 0};   // cross width measured at each of the 4 corners (reused on later laps)
bool     braking = false;              // reverse brake pulse running (start of S_STOP_PAUSE)

// start position
uint16_t stableFront = 0;              // front that stayed steady while waiting (0 = none)
uint16_t candFront = 0;
uint8_t  candCnt = 0;
uint32_t settleOkSince = 0;            // front clear since (settle)

// swerve + centering
int      swerveDir = 0;                // 0 = not swerving, +1 right, -1 left
uint32_t swerveT0 = 0;                 // when the last swerve started
bool     swerveEver = false;           // a swerve has started this run (for the holdoff)
uint8_t  frontCloseCnt = 0, frontClearCnt = 0;
uint8_t  zeroCntR = 0, zeroCntL = 0;   // side reads 0 (blind zone / no return) this many revolutions in a row
float    centerOffset = 0.0f;          // degrees, > 0 = to the right

// stuck watchdog + recovery
uint16_t stuckRefMm = 0;
uint32_t stuckRefMs = 0;
State    recoverFrom = S_FORWARD;
int      recoverTries = 0;

// pause
State    pausedFrom = S_FORWARD;

// turn
uint32_t turnT0 = 0, turnSettleT = 0, lastTurnPrint = 0;
bool     turnSettling = false;
int      turnBoost = 0;
float    progRefYaw = 0;
uint32_t progRefMs = 0, stallSince = 0;

// State 4
enum FinalPhase { FS_TIMED, FS_DRIVE, FS_ALIGN };
FinalPhase fsPhase = FS_DRIVE;
uint16_t startFrontMm = 0;             // narrow front at the start (0 = unknown -> timed State 4)
uint32_t fsT0 = 0, fsAlignT0 = 0, fsStillSince = 0;
int      fsDir = 0;                    // align: +1 forward, -1 back, 0 stopped (settling)
uint32_t fsPulseT0 = 0, fsPulseMs = 0;  // current align pulse
float    fsMsPerMm = 0;                // learned: pulse ms per mm moved (0 = not learned yet)
int      fsLastErr = 0;                // error before the last pulse (0 = no pulse yet)

uint32_t lastYawPrint = 0;

bool running() {
  return state == S_FORWARD || state == S_APPROACH || state == S_STOP_PAUSE || state == S_TURN_RIGHT ||
         state == S_TURN_LEFT || state == S_POST_TURN || state == S_FINAL || state == S_RECOVER;
}

void enterWait() {
  motors(0, 0);
  digitalWrite(PIN_LED, LOW);
  state = S_WAIT_BTN;
  stateT0 = millis();
  Serial.println("ready: press the start button (keep the robot still)");
}

void resetStraightAids() {
  swerveDir = 0;
  frontCloseCnt = frontClearCnt = 0;
  zeroCntR = zeroCntL = 0;
  centerOffset = 0.0f;
}

void stuckReset() { stuckRefMm = 0; stuckRefMs = millis(); }

// Heading hold toward target at base duty (negative base = reversing). Runs only on a NEW IMU sample.
// For a differential drive the same correction sign works forward and backward.
void holdHeading(float target, int base) {
  if (!yawFresh) return;
  float r = wrap(target - yaw) * SIGN;                 // > 0: nose must go right
  float want = constrain(HEAD_KP * r, -HEAD_CORR_MAX, HEAD_CORR_MAX);
  corr += constrain(want - corr, -HEAD_SLEW, HEAD_SLEW);
  int l  = (int)lroundf(base + corr), rr = (int)lroundf(base - corr);
  if (base >= 0) { l = constrain(l, 0, 255);  rr = constrain(rr, 0, 255); }    // never reverses a wheel
  else           { l = constrain(l, -255, 0); rr = constrain(rr, -255, 0); }   // never drives a wheel forward
  motors(l, rr);
}

// ---------------- pause / resume ----------------
void enterForward();
void startFinal();
void beginTurnTimers();

void pauseRun(const char* why) {
  pausedFrom = (state == S_RECOVER) ? recoverFrom : state;
  motors(0, 0);
  state = S_PAUSED;
  Serial.printf("PAUSED in %s: %s - press the button to resume (turns %d/%d)\n",
             stateName(pausedFrom), why, turnCount, TARGET_TURNS);
}

void resumeState(State s) {
  corr = 0.0f;
  switch (s) {
    case S_APPROACH:
      stuckReset(); approachT0 = millis();
      motors(DRIVE_SPEED, DRIVE_SPEED); state = S_APPROACH;
      break;
    case S_STOP_PAUSE: case S_POST_TURN:
      motors(0, 0); braking = false; state = s; stateT0 = millis();
      break;
    case S_TURN_RIGHT: case S_TURN_LEFT:
      beginTurnTimers(); state = s;                    // same target, the turn counter is not counted again
      break;
    case S_FINAL:
      startFinal();
      break;
    default:
      enterForward();
      break;
  }
}

void resumeRun() {
  if (!lidarOk()) { Serial.println("can't resume: lidar not ready"); return; }
  if (!imuOk())   { Serial.println("can't resume: IMU not reporting"); return; }
  Serial.printf("RESUMED (%s)\n", stateName(pausedFrom));
  resumeState(pausedFrom);
}

// ---------------- recovery ----------------
void enterRecover(State from, const char* why) {
  if (++recoverTries > RECOVER_MAX_TRIES) {
    recoverTries = 0;
    pauseRun("stuck: too many recoveries in a row");
    return;
  }
  recoverFrom = from;
  corr = 0.0f;
  motors(-RECOVER_PWM, -RECOVER_PWM);
  state = S_RECOVER;
  stateT0 = millis();
  Serial.printf("RECOVER %d/%d: %s, backing up %lu ms (front %u mm)\n", recoverTries, RECOVER_MAX_TRIES, why,
             (unsigned long)RECOVER_MS, lidarFrontMm);
}

void recoverStep() {
  holdHeading(laneHeading, -RECOVER_PWM);
  if (lidarNewRev && lidarRearMm && lidarRearMm < REAR_STOP_MM) {
    motors(0, 0);
    Serial.printf("recover: rear %u mm, stopped reversing\n", lidarRearMm);
    resumeState(recoverFrom);
    return;
  }
  if (millis() - stateT0 >= RECOVER_MS) {
    motors(0, 0);
    resumeState(recoverFrom);
  }
}

// Stuck: the narrow front reading has not changed by STUCK_MM for STUCK_MS while driving forward.
// Without a valid front reading nothing can be judged, so the window restarts.
bool stuckStep() {
  if (!lidarNewRev) return false;
  uint16_t f = lidarFrontMm;
  if (!f) { stuckReset(); return false; }
  if (!stuckRefMm || abs((int)f - (int)stuckRefMm) >= STUCK_MM) { stuckRefMm = f; stuckRefMs = millis(); return false; }
  return millis() - stuckRefMs >= STUCK_MS;
}

// ---------------- straights: swerve + centering ----------------
// Returns true when the wide front is inside FRONT_DANGER_MM (caller backs up).
bool straightAidsStep(bool allowSwerve) {
  if (swerveDir != 0 && millis() - swerveT0 >= SWERVE_MAX_MS) {
    swerveDir = 0;
    frontCloseCnt = frontClearCnt = 0;
    Serial.printf("swerve end: time limit %lu ms, front %u mm\n", (unsigned long)SWERVE_MAX_MS, lidarFrontWideMm);
  }
  if (!lidarNewRev) return false;

  uint16_t w = lidarFrontWideMm;                       // 0 = no return: neither close nor clear
  bool danger = w && w < FRONT_DANGER_MM;
  if (allowSwerve) {
    if (w) {
      if (w < FRONT_STOP_MM) { if (frontCloseCnt < 250) frontCloseCnt++; frontClearCnt = 0; }
      else                   { if (frontClearCnt < 250) frontClearCnt++; frontCloseCnt = 0; }
    }
    if (swerveDir != 0 && frontClearCnt >= FRONT_CLOSE_READINGS) {
      swerveDir = 0;
      Serial.printf("swerve end: front %u mm\n", w);
    }
    if (swerveDir == 0 && frontCloseCnt >= FRONT_CLOSE_READINGS &&
        (!swerveEver || millis() - swerveT0 >= SWERVE_HOLDOFF_MS)) {
      // More room on the right (larger average) -> swerve right. A sector with no valid return counts
      // as the most room. Equal -> right.
      float rAvg = secRightAny ? secRightAvg : 1e9f;
      float lAvg = secLeftAny  ? secLeftAvg  : 1e9f;
      swerveDir = (rAvg >= lAvg) ? +1 : -1;
      swerveT0 = millis();
      swerveEver = true;
      Serial.printf("swerve %s: front %u mm, front-right avg %.0f, front-left avg %.0f\n",
                 swerveDir > 0 ? "right" : "left", w,
                 secRightAny ? secRightAvg : -1.0f, secLeftAny ? secLeftAvg : -1.0f);
    }
  } else if (swerveDir != 0) {
    swerveDir = 0;
    frontCloseCnt = frontClearCnt = 0;
  }

  // ---- wall centering
  uint16_t open = WALL_MM + OPEN_MARGIN_MM;
  zeroCntR = lidarRightMm ? 0 : (zeroCntR < 250 ? zeroCntR + 1 : zeroCntR);
  zeroCntL = lidarLeftMm  ? 0 : (zeroCntL < 250 ? zeroCntL + 1 : zeroCntL);
  bool wallR = lidarRightMm && lidarRightMm <= open;
  bool wallL = lidarLeftMm  && lidarLeftMm  <= open;
  // Far off the lane heading the side beams hit the walls at a slant: keep the last offset instead.
  if (fabsf(wrap(yaw - laneHeading)) > CENTER_VALID_DEG) return danger;
  float want;
  if (wallR && wallL)
    want = constrain(CENTER_K * ((float)lidarRightMm - (float)lidarLeftMm), -CENTER_MAX_DEG, CENTER_MAX_DEG);
  else if (zeroCntR >= SIDE_ZERO_READINGS && wallL) want = -CENTER_MAX_DEG;   // right wall in the blind zone
  else if (zeroCntL >= SIDE_ZERO_READINGS && wallR) want = +CENTER_MAX_DEG;   // left wall in the blind zone
  else want = 0.0f;
  centerOffset += constrain(want - centerOffset, -CENTER_SLEW_DEG, CENTER_SLEW_DEG);
  return danger;
}

// Heading target for the straights: lane heading plus the swerve or the centering offset.
float straightTarget() {
  float off = (swerveDir != 0) ? swerveDir * SWERVE_DEG : centerOffset;   // swerve wins
  return wrap(laneHeading + SIGN * off);                                  // off > 0 = right
}

// ---------------- State 1 ----------------
void enterForward() {
  if (turnCount >= TARGET_TURNS) { startFinal(); return; }
  corr = 0.0f;
  openCntR = openCntL = 0;
  wallCntR = wallCntL = 0;
  armedR = armedL = false;
  widthReset();
  cornerSlow = false;
  crossWidthMm = 0;
  resetStraightAids();
  stuckReset();
  forwardT0 = millis();
  motors(DRIVE_SPEED, DRIVE_SPEED);
  state = S_FORWARD;
  Serial.printf("State 1: forward (turns %d/%d)\n", turnCount, TARGET_TURNS);
}

// One lidar reading for one side. 0 = no valid return: skipped (neither counts nor resets).
void sideUpdate(uint16_t mm, bool watched, uint8_t &wallCnt, bool &armed, uint8_t &openCnt) {
  uint16_t open = WALL_MM + OPEN_MARGIN_MM;
  if (!watched) { openCnt = 0; return; }
  if (mm == 0) return;
  if (mm <= open) {
    openCnt = 0;
    if (wallCnt < 250) wallCnt++;
    if (wallCnt >= WALL_SEEN_READINGS) armed = true;
  } else {
    wallCnt = 0;
    if (armed && openCnt < 250) openCnt++;           // only a side seen as a wall can open
  }
}

void stopForPivot(const char* why);

// Corner confirmed on side dir (+1 right, -1 left): pivot point = middle of the cross corridor.
void cornerConfirmed(int dir) {
  if (!sideLocked) {
    sideLocked = true;
    lockedRight = (dir > 0);
    Serial.printf("locked onto the %s side\n", lockedRight ? "right (0 deg)" : "left (180 deg)");
  }
  uint16_t cross = crossWidthMm;
  int ci = turnCount % 4;                              // this corner's index around the lap
  if (cross) crossMem[ci] = cross;
  else if (crossMem[ci]) { cross = crossMem[ci]; Serial.println("cross corridor not measured: using this corner's width from an earlier lap"); }
  else { cross = corridorWidth(); Serial.println("cross corridor not measured: using this corridor's width"); }
  int tgt = (int)(cross / 2) + CORNER_CENTER_MARGIN_MM;
  approachTargetMm = (cross && tgt > 0) ? (uint16_t)tgt : 0;
  Serial.printf("corner: right %u mm, left %u mm | cross corridor %u mm -> pivot at front %u mm\n",
                lidarRightMm, lidarLeftMm, cross, approachTargetMm);
  pendingDir = dir;
  resetStraightAids();                                 // no swerve during the approach
  stuckReset();
  uint16_t f = lidarFrontMm ? lidarFrontMm : lidarFrontWideMm;
  if (!approachTargetMm) { stopForPivot("corridor width unknown"); return; }
  if (f && f <= approachTargetMm) { stopForPivot("pivot point already reached"); return; }
  approachT0 = millis();
  state = S_APPROACH;
}

// State 1 -> APPROACH. Evaluated once per lidar revolution. Returns true if the state changed.
bool forwardStep() {
  if (!lidarNewRev) return false;

  bool watchR = !sideLocked || lockedRight;
  bool watchL = !sideLocked || !lockedRight;
  sideUpdate(lidarRightMm, watchR, wallCntR, armedR, openCntR);   // 0 deg   = right
  sideUpdate(lidarLeftMm,  watchL, wallCntL, armedL, openCntL);   // 180 deg = left
  uint16_t open = WALL_MM + OPEN_MARGIN_MM;
  if (lidarRightMm && lidarLeftMm && lidarRightMm <= open && lidarLeftMm <= open) {
    widthHist[widthIdx] = lidarRightMm + lidarLeftMm;
    widthIdx = (widthIdx + 1) % WIDTH_HIST;
  }

  bool inLockout = turnCount > 0 && millis() - forwardT0 < LOCKOUT_MS;     // short backstop after a turn

  // corner ahead, from the scan geometry (only on an armed side: its wall has been seen on this straight)
  cornerAheadMm = -1;
  if (watchR && armedR && scanCornerR) {
    cornerAheadMm = scanAheadR; if (scanAheadR > 30) crossWidthMm = scanCrossR;   // measured while still ahead
  } else if (watchL && armedL && scanCornerL) {
    cornerAheadMm = scanAheadL; if (scanAheadL > 30) crossWidthMm = scanCrossL;
  }

  bool opening = (watchR && openCntR > 0) || (watchL && openCntL > 0);
  bool close = cornerAheadMm >= 0 && cornerAheadMm <= (int)CORNER_SLOW_AHEAD_MM;
  if ((opening || close) && !cornerSlow && !inLockout) {
    cornerSlow = true;
    Serial.printf("corner %s: slowing to %d, cross corridor %u mm\n",
                  close ? "ahead" : "opening", CORNER_SPEED, crossWidthMm);
  } else if (!opening && !close && cornerSlow && cornerAheadMm < 0 && !crossWidthMm) {
    cornerSlow = false;
    Serial.println("corner was a false alarm, back to full speed");
  }
  if (inLockout) return false;

  int dir = 0;                                         // both at once is not expected; right is checked first
  if (openCntR >= OPEN_READINGS) dir = +1;
  else if (openCntL >= OPEN_READINGS) dir = -1;
  if (!dir) return false;
  cornerConfirmed(dir);
  return true;
}

// Stop for a pivot: short brake, then S_STOP_PAUSE runs out the pause and starts the turn.
void stopForPivot(const char* why) {
  Serial.printf("pivot point (%s): front %u mm, braking\n", why, lidarFrontMm);
  if (BRAKE_MS > 0) motorsBrake(); else motors(0, 0);
  braking = true;
  state = S_STOP_PAUSE;
  stateT0 = millis();
}

// APPROACH -> stop -> turn, when the front reaches the middle of the cross corridor.
void approachStep() {
  holdHeading(laneHeading, CORNER_SPEED);
  bool go = false;
  const char* why = "";
  if (approachTargetMm == 0) { go = true; why = "corridor width unknown, turning here"; }
  else if (millis() - approachT0 >= APPROACH_MAX_MS) { go = true; why = "approach time limit"; }
  else if (lidarNewRev) {
    uint16_t f = lidarFrontMm ? lidarFrontMm : lidarFrontWideMm;
    if (f && f <= approachTargetMm) { go = true; why = "centre reached"; }
    else if (lidarFrontWideMm && lidarFrontWideMm < FRONT_DANGER_MM) { go = true; why = "front too close"; }
  }
  if (!go) {
    if (stuckStep()) enterRecover(S_APPROACH, "stuck during approach");
    return;
  }
  stopForPivot(why);
}

// ---------------- turns ----------------
void beginTurnTimers() {
  turnT0 = millis();
  lastTurnPrint = turnT0;
  turnSettling = false;
  turnBoost = 0;
  progRefYaw = yaw;
  progRefMs = turnT0;
  stallSince = 0;
}

void startTurn(int dir) {              // dir: +1 right, -1 left
  turnCount++;
  laneHeading = wrap(laneHeading + SIGN * dir * TURN_DEG);   // exact next lane, not "yaw +/- 90"
  beginTurnTimers();
  Serial.printf("State %d: turn %s #%d: yaw %.1f -> target %.1f\n",
             dir > 0 ? 2 : 3, dir > 0 ? "right" : "left", turnCount, yaw, laneHeading);
  state = (dir > 0) ? S_TURN_RIGHT : S_TURN_LEFT;
}

// PD turn to laneHeading. dir: +1 right, -1 left.
void turnStep(int dir) {
  uint32_t el = millis() - turnT0;
  float e = wrap(laneHeading - yaw);
  float r = dir * e * SIGN;                             // > 0: still need to turn in direction dir
  float rate = dir * yawRate * SIGN;                    // > 0: turning in direction dir now

  if (millis() - lastTurnPrint >= TURN_PRINT_MS) {      // progress line: time, yaw, degrees left, turn rate
    lastTurnPrint = millis();
    Serial.printf("  t=%lu ms yaw %.1f left %.1f rate %.1f boost %d\n", (unsigned long)el, yaw, r, rate, turnBoost);
  }

  if (el > 800 && fabsf(e) > fabsf(TURN_DEG) + 25) {
    Serial.println("ABORT: the error is growing. Flip YAW_RISES_TURNING_RIGHT and run again.");
    Serial.println("stopped");
    enterWait();
    return;
  }
  if (el >= TURN_TIMEOUT_MS) {
    Serial.printf("turn timed out after %lu ms, yaw %.1f\n", (unsigned long)el, yaw);
    pauseRun("turn timed out");
    return;
  }

  bool crossed = (r <= 0);                              // reached or passed the target
  if (fabsf(r) < DEADBAND_DEG || crossed) {
    motors(0, 0);                                       // stop driving; no more corrections, so it cannot hunt
    progRefYaw = yaw; progRefMs = millis(); stallSince = 0;   // coasting is not a stall
    if (fabsf(rate) < STOP_RATE_DPS) {
      if (!turnSettling) { turnSettling = true; turnSettleT = millis(); }
      else if (millis() - turnSettleT >= TURN_FINISH_MS) {
        Serial.printf("turn done in %lu ms, yaw %.1f (error %.1f)\n", (unsigned long)(millis() - turnT0), yaw,
                   wrap(laneHeading - yaw));
        recoverTries = 0;                               // progress made
        state = S_POST_TURN;
        stateT0 = millis();
      }
    } else {
      turnSettling = false;
    }
    return;
  }

  turnSettling = false;
  // stall check: no yaw progress -> boost the duty step by step, then give up
  if (millis() - progRefMs >= TURN_PROGRESS_MS) {
    if (fabsf(wrap(yaw - progRefYaw)) < TURN_PROGRESS_DEG) {
      if (turnBoost < TURN_BOOST_MAX) {
        turnBoost = constrain(turnBoost + TURN_BOOST_STEP, 0, TURN_BOOST_MAX);
        Serial.printf("turn not moving: boost +%d duty\n", turnBoost);
      } else if (!stallSince) stallSince = millis();
      else if (millis() - stallSince >= TURN_STALL_MS) {
        pauseRun("turn stalled at full boost (raise TURN_MIN / TURN_MAX)");
        return;
      }
    } else stallSince = 0;
    progRefYaw = yaw; progRefMs = millis();
  }
  float u = KP * r - KD * rate;                         // PD: the D term brakes as it speeds up
  int v = constrain((int)fabsf(u), TURN_MIN, TURN_MAX) + turnBoost;
  v = constrain(v, 0, 255);
  if ((u > 0) == (dir > 0)) motors(v, -v);              // pivot right
  else                      motors(-v, v);              // pivot left
}

// ---------------- State 4 ----------------
void finishFinal(const char* why) {
  motors(0, 0);
  Serial.printf("done (%s): front %u mm, start front %u mm, yaw %.1f (error %.1f)\n", why, lidarFrontMm,
             startFrontMm, yaw, wrap(laneHeading - yaw));
  enterWait();
}

void startFinal() {
  corr = 0.0f;
  resetStraightAids();
  stuckReset();
  fsT0 = millis();
  fsDir = 0;
  state = S_FINAL;
  if (startFrontMm) {
    fsPhase = FS_DRIVE;
    Serial.printf("State 4: drive to the start position (front %u mm)\n", startFrontMm);
  } else {
    fsPhase = FS_TIMED;
    Serial.printf("State 4: no start front recorded, final forward %lu ms\n", (unsigned long)FINAL_MS);
  }
  motors(DRIVE_SPEED, DRIVE_SPEED);
}

void finalStep() {
  if (fsPhase == FS_TIMED) {
    if (straightAidsStep(true)) { enterRecover(S_FINAL, "front too close"); return; }
    if (millis() - fsT0 >= FINAL_MS) { finishFinal("timed"); return; }
    holdHeading(straightTarget(), DRIVE_SPEED);
    return;
  }

  if (fsPhase == FS_DRIVE) {
    straightAidsStep(false);                            // the target is a known safe distance: no swerve here
    if (stuckStep()) { enterRecover(S_FINAL, "stuck in State 4"); return; }
    holdHeading(straightTarget(), DRIVE_SPEED);
    bool near = lidarNewRev && lidarFrontMm && lidarFrontMm <= startFrontMm + FRONT_SLOW_MM;
    bool late = millis() - fsT0 >= FINAL_MAX_MS;
    if (near || late) {
      motors(0, 0);
      if (late && !near) Serial.println("State 4: drive time limit, aligning on the front beam");
      fsPhase = FS_ALIGN;
      fsDir = 0;
      fsMsPerMm = 0;
      fsLastErr = 0;
      fsAlignT0 = fsStillSince = millis();
      Serial.printf("align: front %u mm, target %u mm\n", lidarFrontMm, startFrontMm);
    }
    return;
  }

  // ---- FS_ALIGN: timed pulses forward / back until the front matches the start reading.
  // At ALIGN_PWM the robot moves further between two lidar readings than the deadband, so it cannot
  // stop on a reading. Instead: pulse, stop, settle, measure, and size the next pulse from how far
  // the last one moved (no speed needs to be known).
  if (millis() - fsAlignT0 > ALIGN_TIMEOUT_MS) { finishFinal("align timeout"); return; }
  if (fsDir != 0) {                                     // a pulse is running
    holdHeading(laneHeading, fsDir * ALIGN_PWM);
    if (millis() - fsPulseT0 >= fsPulseMs) {
      motors(0, 0);
      fsDir = 0;
      fsStillSince = millis();
    }
    return;
  }
  motors(0, 0);                                         // stopped: wait, then judge a fresh reading
  if (millis() - fsStillSince < ALIGN_SETTLE_MS || !lidarNewRev || !lidarFrontMm) return;
  int err = (int)lidarFrontMm - (int)startFrontMm;      // > 0: still too far from the wall ahead -> forward
  if (abs(err) <= (int)FRONT_DEADBAND_MM) { finishFinal("front matched"); return; }
  if (fsLastErr != 0) {                                 // learn from the previous pulse
    int moved = abs(fsLastErr - err);
    if (moved >= 3) fsMsPerMm = (float)fsPulseMs / moved;
  }
  uint32_t ms = fsMsPerMm > 0 ? (uint32_t)(abs(err) * fsMsPerMm * 0.8f) : ALIGN_PULSE_FIRST_MS;
  fsPulseMs = constrain(ms, ALIGN_PULSE_MIN_MS, ALIGN_PULSE_MAX_MS);
  fsLastErr = err;
  fsDir = (err > 0) ? 1 : -1;
  fsPulseT0 = millis();
  corr = 0.0f;
  motors(fsDir * ALIGN_PWM, fsDir * ALIGN_PWM);
  Serial.printf("align: front %u mm (target %u), pulse %s %lu ms\n", lidarFrontMm, startFrontMm,
                fsDir > 0 ? "forward" : "back", (unsigned long)fsPulseMs);
}

// ============================================================
// SETUP / LOOP
// ============================================================
void setup() {
  Serial.begin(115200);
  int pins[] = {L_FWD, L_BACK, R_FWD, R_BACK};
  for (int p : pins) analogWrite(p, 0);
  delay(1000);

  Serial.println("\n=== lidar_fsm ===");

  L.setRxBufferSize(4096);                // must come before begin()
  L.begin(LIDAR_BAUD, SERIAL_8N1, LIDAR_RX, LIDAR_TX);
  delay(1500);
  scanning = startScan();
  if (!scanning) Serial.println("lidar not responding, retrying...");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, -1);
  Serial.println("step 1: beginSPI");
  if (!imu.beginSPI(PIN_CS, PIN_INT, PIN_RST, 3000000, SPI)) {
    Serial.println("FAIL: BNO08x not found over SPI, stopping.");
    while (true) delay(1000);
  }
  Serial.println("step 2: beginSPI ok");
  delay(500);
  imu.enableGameRotationVector(20);
  delay(100);
  imu.getSensorEvent();
  uint32_t t = millis();
  while (!haveYaw && millis() - t < 3000) { updateYaw(); delay(5); }
  if (!haveYaw) { Serial.println("FAIL: no yaw from the IMU in 3 s, stopping."); while (true) delay(1000); }
  Serial.printf("step 3: yaw %.1f, IMU running\n", yaw);

  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
  enterWait();
}

void loop() {
  serviceLidar();
  updateYaw();

  // ---- button: start (waiting), pause (running), resume (paused), cancel (settling)
  if (buttonPressed()) {
    if (state == S_WAIT_BTN) {
      digitalWrite(PIN_LED, HIGH);
      state = S_SETTLE_START;
      stateT0 = settleOkSince = millis();
      if (stableFront) Serial.printf("start: keep the robot still, hands clear of the front (expect %u mm)\n", stableFront);
      else Serial.println("start: keep the robot still (no steady front reading)");
    } else if (state == S_SETTLE_START) {
      Serial.println("start cancelled");
      enterWait();
    } else if (state == S_PAUSED) {
      resumeRun();
    } else if (running()) {
      pauseRun("button");
    }
  }

  // ---- watchdogs: losing a sensor pauses the run (laps kept)
  if (running() && !lidarOk()) pauseRun("lidar lost");
  if (running() && !imuOk())   pauseRun("IMU lost");

  // ---- LED
  if (state == S_WAIT_BTN)  digitalWrite(PIN_LED, (millis() / 300) % 2);
  else if (state == S_PAUSED) digitalWrite(PIN_LED, (millis() / 100) % 2);
  else digitalWrite(PIN_LED, HIGH);

  // Lidar a little late: do not drive blind. Motors off until the next revolution (the step functions
  // set them again on the next IMU sample). Turns are IMU-only and carry on.
  bool stale = haveRev && millis() - lidarRevMs > LIDAR_STALE_MS;
  if (stale && (state == S_FORWARD || state == S_APPROACH || state == S_FINAL || state == S_RECOVER)) {
    motors(0, 0);
    corr = 0.0f;
    return;
  }

  switch (state) {
    case S_WAIT_BTN:
      if (lidarNewRev) {                                // start front = the front reading that held steady
        uint16_t f = frontMedian();
        if (f && candFront && abs((int)f - (int)candFront) <= (int)START_STABLE_MM) {
          if (candCnt < 250) candCnt++;
          if (candCnt >= START_STABLE_READINGS) stableFront = candFront;
        } else { candFront = f; candCnt = 0; }
      }
      if (millis() - lastYawPrint > 500) {
        lastYawPrint = millis();
        Serial.printf("yaw %.1f | lidar 0deg %u mm, 180deg %u mm, 270deg %u mm%s\n", yaw, lidarRightMm, lidarLeftMm,
                   lidarFrontMm, lidarOk() ? "" : " (lidar NOT ok)");
      }
      break;

    case S_SETTLE_START: {
      // the front must be back at the steady value (hand gone) for START_SETTLE_MS
      uint16_t fm = frontMedian();
      bool clear = !stableFront || (fm && abs((int)fm - (int)stableFront) <= (int)START_CLEAR_MM);
      if (!clear) settleOkSince = millis();
      if (millis() - stateT0 >= START_CLEAR_TIMEOUT_MS) {
        Serial.printf("front not clear (%u mm, expected %u mm) - something in front of the lidar? not starting\n",
                      fm, stableFront);
        enterWait();
        break;
      }
      if (millis() - settleOkSince >= START_SETTLE_MS) {
        if (!lidarOk()) { Serial.println("lidar not ready, not starting"); enterWait(); break; }
        if (!imuOk())   { Serial.println("IMU not reporting, not starting"); enterWait(); break; }
        laneHeading = yaw;                              // lane 1 = the way the robot faces now
        turnCount = 0;
        sideLocked = false;
        for (int i = 0; i < 4; i++) crossMem[i] = 0;
        swerveEver = false;
        recoverTries = 0;
        startFrontMm = stableFront ? stableFront : frontMedian();   // where State 4 must stop (0 = unknown)
        if (startFrontMm) Serial.printf("lane heading %.1f | start front %u mm\n", laneHeading, startFrontMm);
        else Serial.printf("lane heading %.1f | start front unknown: State 4 will be timed (%lu ms)\n", laneHeading,
                        (unsigned long)FINAL_MS);
        enterForward();
      }
      break;
    }

    case S_FORWARD: {                                   // State 1
      if (forwardStep()) break;                         // corner -> approach / pivot
      if (straightAidsStep(!cornerSlow)) {              // front inside FRONT_DANGER_MM
        if (cornerSlow) {                               // the side is opening: this is the corner, pivot now
          bool r = (openCntR > 0) && (!sideLocked || lockedRight);
          Serial.println("front too close while the side opens: pivot now");
          cornerConfirmed(r ? +1 : -1);
          if (state == S_APPROACH) stopForPivot("front too close");
        } else enterRecover(S_FORWARD, "front too close");
        break;
      }
      if (stuckStep()) { enterRecover(S_FORWARD, "stuck"); break; }
      holdHeading(straightTarget(), cornerSlow ? CORNER_SPEED : DRIVE_SPEED);
      break;
    }

    case S_APPROACH:
      approachStep();
      break;

    case S_STOP_PAUSE:
      if (braking && millis() - stateT0 >= BRAKE_MS) { motors(0, 0); braking = false; }
      if (millis() - stateT0 >= STOP_PAUSE_MS) { motors(0, 0); braking = false; startTurn(pendingDir); }
      break;

    case S_TURN_RIGHT:                                  // State 2
      turnStep(+1);
      break;

    case S_TURN_LEFT:                                   // State 3
      turnStep(-1);
      break;

    case S_POST_TURN:
      if (millis() - stateT0 >= POST_TURN_MS) enterForward();   // State 1, or State 4 at TARGET_TURNS
      break;

    case S_FINAL:                                       // State 4
      finalStep();
      break;

    case S_RECOVER:
      recoverStep();
      break;

    case S_PAUSED:
      motors(0, 0);
      break;
  }

  delay(1);
}
