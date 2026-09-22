#include <Arduino.h>
#include <Servo.h>
#include <SPI.h>
#include <Wire.h>
#include <SparkFun_BNO08x_Arduino_Library.h>
#include <Adafruit_TCS34725.h>

#ifndef DEG_TO_RAD
#define DEG_TO_RAD 0.017453292519943295
#endif

// ============================================================
// OBSTACLE ROUND - NON-BLOCKING FIRMWARE
//
// Built from OpenRound.cpp (same FSM, pins, calibration, turn trigger,
// recovery, final straight) with the old obstacle-round logic
// (main.ino + tracker.py) folded into the driving:
//
//   old main.ino                      this file
//   ---------------------------------------------------------------
//   wall PD on ToF (right - left)  -> wall-centre PD on LiDAR R - L,
//                                     output is a heading offset that the
//                                     IMU heading PID tracks
//   vision PD, red -100 / green    -> same gains, same targets, same
//     +100 px, area blend 1000-1800   area blend
//   servo = 97 - blended (67..127) -> servo = 76.5 + blended (20..140)
//                                     sign flipped: on this linkage a
//                                     LOWER servo angle steers LEFT
//   sonar <50 cm hard steer        -> dropped
//   sonar <30 cm reverse (blocking)-> non-blocking RECOVER at 200 mm,
//                                     suppressed while the camera sees a
//                                     pillar (the pillar PD owns that)
//   magenta                        -> ignored (Pi never sends it)
//
// SERIAL FRAME (Pi -> STM32), one line per send, ~50 Hz:
//     left,front,right,rev,color,err,area,vseq,coneL,coneR,wallAng,pX,pY,uX,uY\n
//   left/front/right  mm, 65535 = no return (single beams, used for the
//                     corner trigger and the front panic)
//   rev               lidar revolution counter - cone values only change
//                     once per rev, so the wall PD and levelling run per rev
//   coneL / coneR     PERPENDICULAR distance to the left / right wall, from a
//                     line fitted to every LiDAR point in a 45 deg cone
//                     centred on 90 / 270 deg (pillars rejected as outliers).
//                     65535 = no wall fitted on that side
//   wallAng           car yaw relative to the walls, deci-degrees,
//                     + = car pointing LEFT of the wall direction.
//                     32767 = no valid fit
//   pX / pY           chosen pillar's position in mm from the LiDAR,
//                     x forward, y left (camera bearing + LiDAR range).
//                     32767 = no pillar / not located
//   uX / uY           nearest LiDAR object inside the corridor that is not
//                     the camera's pillar - colour unknown. 32767 = none
//   color             1 = red, 0 = green, 2 = none
//   err               pillar centre x - 320 (640 px frame), + = right
//   area              blob area in 320x240 detection pixels
//   vseq              camera frame counter - the vision PD only updates
//                     when this changes, so a camera frame resent in
//                     several serial lines can't spike the D term
// A 4-field open-round line (left,front,right,rev) still parses; vision
// then reads as "none".
//
// COMMANDS (Pi -> STM32), sent by the Start / Stop buttons on the Pi page:
//     S\n   START - only accepted while armed in WAIT_START (or FINISHED,
//            which re-arms and starts a fresh run) and the LiDAR is live
//     X\n   STOP  - motor off, steering centred, straight to FINISHED
//
// START SEQUENCE (no physical button)
//   power on -> wait for the first Pi frame (LED1 slow blink)
//            -> armed in WAIT_START (LED1 medium blink)
//            -> START from the page -> run
//
// PER CORNER
//   1. drive on IMU heading + cone wall centring + pillar PD; the lane
//      heading is levelled slowly to the fitted wall direction so IMU
//      drift can't build up over 3 laps
//   2. FIRST corner only: floor colour line arms the gate and sets
//      direction (orange = CW, blue = CCW)
//   3. turn fires when the turn-side distance reads above SIDE_OPEN_MM
//      for SIDE_OPEN_FRAMES consecutive NEW frames
//   4. 90 deg arc at TURN_LOCK_FRACTION of full lock (wider than the open
//      round), vision OFF during the arc, back to DRIVE right after
//      (vision ON again immediately, including the post-corner lockout)
//
// BENCH-VERIFIED (inherited from OpenRound.cpp)
//   motor    PA2 forward, PA3 reverse
//   encoder  TIM5 PA0/PA1, negated so forward counts up
//   IMU      BNO08x SPI1 ~100 Hz, clockwise = negative yaw
//   colour   TCS34725 CH4, white pR 47 / orange pR 69 / blue pB 27
//   servo    500-2500 us, straight 76.5, left stop 20, right stop 140
//            turning radius at full lock: left 27 cm, right 25 cm
// ============================================================

enum BlockColor { COLOR_NONE, COLOR_ORANGE, COLOR_BLUE };

enum RobotState {
  STATE_WAIT_START,
  STATE_DRIVE_TO_CORNER,
  STATE_TURNING,
  STATE_FINAL_STRAIGHT,
  STATE_RECOVER,
  STATE_FINISHED,
  STATE_CORNER_MANEUVER      // 3-point corner after an inner-side pillar (see below)
};

// ============================================================
// HARDWARE PINS & OBJECTS
// ============================================================
const int MOT_RPWM_PIN = PA2;     // forward  (TIM2_CH3; TIM5 is the encoder)
const int MOT_LPWM_PIN = PA3;     // reverse  (TIM2_CH4)
const int SERVO_PIN    = PA8;

const int IMU_CS_PIN  = PA4;
const int IMU_INT_PIN = PB0;
const int IMU_RST_PIN = PB1;

const int LED1_PIN = PB12;        // slow blink (500 ms) = waiting for Pi; medium blink (250 ms) = armed, waiting for START; solid = running; fast blink = lidar stale
const int LED2_PIN = PB13;        // lit while ORANGE is under the sensor
const int LED3_PIN = PB14;        // lit while BLUE is under the sensor

#define I2C_SCL     PB6
#define I2C_SDA     PB7
#define TCA_RST_PIN PB8
#define TCA_ADDR    0x70
#define TCS_CH      4             // TCS34725

// ---- calibration ----
       float TICKS_PER_CM        = 14.853;

const int   SERVO_MIN_PULSE_US  = 500;
const int   SERVO_MAX_PULSE_US  = 2500;
       float SERVO_TRUE_STRAIGHT = 76.5;
       float SERVO_MAX_LEFT      = 20.0;   // left hard stop  (below straight steers LEFT)
       float SERVO_MAX_RIGHT     = 140.0;  // right hard stop (above straight steers RIGHT)
       float IMU_YAW_SIGN        = 1.0;    // clockwise reads negative

// Obstacle round: one constant PWM for driving, turning and reversing.
       int DRIVE_PWM = 60;


SPIClass SPI_IMU(PA7, PA6, PA5);  // MOSI, MISO, SCLK
Servo steeringServo;
BNO08x myIMU;
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_16X);

bool  tcsOk = false;
float initialYawOffset = 0.0;

// ============================================================
// PI LINK   "left,front,right,rev,color,err,area,vseq\n"
// ============================================================
       unsigned long LIDAR_STALE_MS      = 200;
       unsigned long LIDAR_DEAD_MS       = 1000;
       uint16_t      LIDAR_MAX_VALID_MM  = 3500;   // mat diagonal
const uint16_t      LIDAR_FAR           = 9999;   // internal "nothing there"

// A beam with no return must read FAR, never near.
uint16_t lidarSanitize(long v) {
  if (v <= 0 || v > (long)LIDAR_MAX_VALID_MM) return LIDAR_FAR;
  return (uint16_t)v;
}

uint16_t      lidarL = LIDAR_FAR, lidarF = LIDAR_FAR, lidarR = LIDAR_FAR;
unsigned long lidarLastMs = 0;
bool          lidarStale  = true;
bool          lidarDead   = true;
uint32_t      lidarFrames = 0;
bool          lidarNewFrame = false;   // true only on the loop a frame was parsed

// vision fields, from the same line
const int VIS_RED   = 1;
const int VIS_GREEN = 0;
const int VIS_NONE  = 2;
int      visColor = VIS_NONE;
int      visErr   = 0;
long     visArea  = 0;
uint32_t visSeq   = 0;

// 45 deg cone wall fits, from the same line
const long  CONE_NONE     = 65535;
const long  ANG_NONE      = 32767;
uint16_t coneL = LIDAR_FAR, coneR = LIDAR_FAR;   // perpendicular mm, FAR = no fit
bool     wallAngValid = false;
float    wallAngDeg   = 0.0;                     // + = car pointing left of the walls
uint32_t lidarRev     = 0;
bool     lidarNewRev  = false;                   // true only on the loop rev changed

// pillar position in the car frame (Pi: camera bearing + LiDAR range)
const long PXY_NONE = 32767;
bool  pillarXYValid = false;
float pillarX = 0.0, pillarY = 0.0;              // mm from the LiDAR, x fwd, y left
// nearest LiDAR object in the corridor whose colour is not known yet
// SECONDARY PILLAR - the second-largest accepted blob, i.e. a pillar that is
// further away but still seen. Approaching a corner the nearest pillar is the
// one on THIS straight; anything still visible behind it is almost always the
// first pillar of the NEXT straight, and its colour is all the corner needs to
// know which way to come out. See planCornerExit().
int   secColor    = VIS_NONE;
bool  secXYValid  = false;
float secX = 0.0, secY = 0.0;

bool  unknownXYValid = false;
float unknownX = 0.0, unknownY = 0.0;

// ============================================================
// NAV LINK  (fields 18..22)  -  the map-based path tracker on the Pi
// ============================================================
// The Pi now localises against the KNOWN arena (3000 mm outer, 1000 mm
// internal block, both fixed by the rules), builds a map of the traffic
// signs in MAT coordinates, and plans one line for the whole lap. All of
// that stays on the Pi. What arrives here is three numbers and a validity
// bit - this firmware needs no map, no plan and no geometry.
//
//   navOk     1 = the Pi's pose is trustworthy AND a plan exists
//   navCross  cross-track error to the planned line, mm, + = car LEFT of it
//   navHdg    heading error to the line tangent, deci-deg, + = steer LEFT
//   navCurv   planned curvature a short preview ahead, 1e-6 / mm, signed
//   navDone   1 = three laps of arc length completed
//
// WHY THIS IS SAFE TO SHIP. navOk is checked every loop. The moment it goes
// false - Pi crash, LiDAR stall, scan matching rejected too long, USB
// unplugged - the car falls straight back to the wall-relative planner
// below, continuously, with no mode to get stuck in. Setting
// USE_NAV_TRACK = 0 from the Pi page reverts to exactly the old behaviour
// without a re-flash.
       bool     USE_NAV_TRACK  = false;   // OFF by default. Turn on from the page.
       float    NAV_KP_HEAD    = 0.9;
       float    NAV_KP_CROSS   = 0.0035;  // rad per mm, through an atan
       float    NAV_MAX_STEER  = 26.0;    // deg of front wheel, <= the 26.7 lock
       unsigned long NAV_STALE_MS = 250;

bool     navOk        = false;
int      navCross     = 0;
float    navHdgDeg    = 0.0;
float    navCurv      = 0.0;              // 1/mm
bool     navDone      = false;
unsigned long navLastMs = 0;

// Wheelbase, CAD measured (Front Bumper.step): rear axle to front axle.
const float NAV_WHEELBASE_MM = 135.85;

bool navValid() {
  return USE_NAV_TRACK && navOk && (millis() - navLastMs) < NAV_STALE_MS;
}

char    lidarBuf[256];               // 23-field frame is ~150 chars
uint8_t lidarLen = 0;

bool startRequested = false;
bool stopRequested  = false;

bool parseTuning(char *s);           // PARAMETER TABLE, near the bottom

void parseLine() {
  // one-letter commands from the Pi page
  if (lidarBuf[0] == 'S' && lidarBuf[1] == '\0') { startRequested = true; return; }
  if (lidarBuf[0] == 'X' && lidarBuf[1] == '\0') { stopRequested  = true; return; }

  // tuning lines share this port; they never look like a sensor frame
  // (a frame always starts with a digit or '-')
  if (parseTuning(lidarBuf)) return;

  long f[23];
  uint8_t n = 0;
  char *p = lidarBuf;
  while (n < 23) {
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) break;             // no digits - malformed
    f[n++] = v;
    if (*end != ',') break;
    p = end + 1;
  }
  if (n < 3) return;                 // not even a lidar frame - drop

  lidarL = lidarSanitize(f[0]);
  lidarF = lidarSanitize(f[1]);
  lidarR = lidarSanitize(f[2]);

  if (n >= 8) {
    visColor = (f[4] == VIS_RED || f[4] == VIS_GREEN) ? (int)f[4] : VIS_NONE;
    visErr   = (int)f[5];
    visArea  = f[6];
    visSeq   = (uint32_t)f[7];
  } else {
    visColor = VIS_NONE;             // open-round feed: no camera
  }

  if (n >= 4 && (uint32_t)f[3] != lidarRev) { lidarRev = (uint32_t)f[3]; lidarNewRev = true; }

  if (n >= 11) {
    coneL = (f[8] == CONE_NONE) ? LIDAR_FAR : lidarSanitize(f[8]);
    coneR = (f[9] == CONE_NONE) ? LIDAR_FAR : lidarSanitize(f[9]);
    wallAngValid = (f[10] != ANG_NONE);
    wallAngDeg   = wallAngValid ? f[10] / 10.0f : 0.0f;
  } else {                           // older feed: fall back to the 90/270 beams
    coneL = lidarL; coneR = lidarR; wallAngValid = false;
  }

  if (n >= 13 && f[11] != PXY_NONE && f[12] != PXY_NONE) {
    pillarXYValid = true; pillarX = (float)f[11]; pillarY = (float)f[12];
  } else {
    pillarXYValid = false;
  }
  if (n >= 15 && f[13] != PXY_NONE && f[14] != PXY_NONE) {
    unknownXYValid = true; unknownX = (float)f[13]; unknownY = (float)f[14];
  } else {
    unknownXYValid = false;
  }

  if (n >= 18) {
    secColor   = (f[15] == VIS_RED || f[15] == VIS_GREEN) ? (int)f[15] : VIS_NONE;
    secXYValid = (f[16] != PXY_NONE && f[17] != PXY_NONE);
    if (secXYValid) { secX = (float)f[16]; secY = (float)f[17]; }
  } else {
    secColor = VIS_NONE; secXYValid = false;      // older feed: no second pillar
  }

  // nav fields. A feed without them simply never sets navOk, so an old Pi
  // build and this firmware still work together.
  if (n >= 23) {
    navOk     = (f[18] != 0);
    navCross  = (int)f[19];
    navHdgDeg = f[20] / 10.0f;
    navCurv   = f[21] / 1000000.0f;
    navDone   = (f[22] != 0);
    navLastMs = millis();
  } else {
    navOk = false;
  }

  lidarLastMs   = millis();
  lidarStale    = false;
  lidarDead     = false;
  lidarFrames++;
  lidarNewFrame = true;
}

void serviceLidar() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lidarLen > 0) {
        lidarBuf[lidarLen] = '\0';
        parseLine();
        lidarLen = 0;
      }
    } else if (lidarLen < sizeof(lidarBuf) - 1) {
      lidarBuf[lidarLen++] = c;
    } else {
      lidarLen = 0;                 // overflow - drop and resync on newline
    }
  }
  unsigned long age = millis() - lidarLastMs;
  if (age > LIDAR_STALE_MS) lidarStale = true;
  if (age > LIDAR_DEAD_MS)  lidarDead  = true;
}

// ---- turn trigger tuning ----
       uint16_t SIDE_OPEN_MM     = 1500;  // turn side above this = inner wall gone
       uint8_t  SIDE_OPEN_FRAMES = 3;     // consecutive NEW frames before believing it
       float    TURN_TRIGGER_MAX_YAW = 25.0;  // only while the car is this close to the lane
                                             //   direction: mid-swerve (up to 75 deg) the "side"
                                             //   beam isn't sideways and fakes an open corner (sim)

uint8_t sideOpenCount = 0;
uint8_t sideWallCount = 0;              // frames the inner wall has been SEEN this straight
       uint8_t SIDE_WALL_FRAMES = 3;     // must see it this many times before "open" can count:
                                        //   a straight that starts inside a corner square (after a
                                        //   3-point, or a late turn) sees "open" before the inner
                                        //   wall even begins - that faked a second corner in sim

// ---- wall recovery ----
       uint16_t WALL_PANIC_MM     = 200;
       uint16_t WALL_CLEAR_MM     = 350;
#define        RECOVER_PWM         DRIVE_PWM
       float    RECOVER_MAX_CM    = 30.0;
       int      RECOVER_MAX_TRIES = 3;

// ============================================================
// RUN CONSTANTS
// ============================================================
       int   TARGET_CORNERS      = 12;
       float   FINAL_STRAIGHT_CM   = 100;
       float SEARCH_SAFETY_CM    = 400.0;
       float POST_CORNER_LOCKOUT_CM = 50.0;

// ---- CORNER EXIT SHAPING ----------------------------------------------
// The sim showed the corner, not the camera, is what loses an after-corner
// pillar. A plain arc fired the moment the turn side opens always ends with
// the car pinned to the OUTER wall of the new straight (lane offset ~ +365 mm
// against a LANE_LIMIT of 398). From there an outer-side pass is free and an
// inner-side pass is unreachable, so latReach() correctly gives up and the car
// deliberately goes by on the wrong side.
//
// So the exit lane position stops being a by-product and becomes a commanded,
// tunable quantity:
//
//   TURN_DELAY_MM        drive this much further after the turn side opens
//                        before starting the arc. Later trigger = the arc
//                        finishes deeper into the new straight = exit nearer
//                        the inner side.
//   CORNER_EXIT_MM       lane offset to aim for as the arc hands over
//                        (+ = left of centre). 0 = centred, which keeps BOTH
//                        passing sides reachable - the safe default when the
//                        next straight cannot be seen yet.
//   POST_CORNER_YAW_MAX  steering authority for the first POST_CORNER_BOOST_MM
//   POST_CORNER_BOOST_MM of a straight. Plain centring is capped at
//                        CENTRE_YAW_MAX (20 deg), which is far too gentle to
//                        recover a whole lane width before a pillar 250 mm in.
//   PRE_CORNER_SWING_MM  lane offset to hold once the corner is within
//   PRE_CORNER_ZONE_MM   PRE_CORNER_ZONE_MM and no pillar is in play. Entering
//                        the corner from a chosen side is the cheapest way to
//                        leave it on a chosen side.
//   CARRY_TRACKS         keep sighting pillars through the arc and rotate the
//                        survivors into the new lane frame, so a pillar after
//                        the corner is already confirmed when DRIVE resumes
//                        instead of needing TRACK_CONFIRM fresh sightings.
      float TURN_DELAY_MM        = 0.0;    // minimum run-out after the corner is seen
      float TURN_FRONT_MM        = 600.0;  // start the arc when the wall ahead is this close (0 = off)
      float TURN_ARM_MAX_MM      = 1600.0;  // backstop: arc anyway this far after arming
      bool  CORNER_ARC_ADAPT     = true;   // shape the arc to the planned exit side
      bool  USE_SECONDARY_CORNER = true;   // use the 2nd pillar's colour to shape the corner
      float SECONDARY_ZONE_MM    = 1500.0; // ... if it was seen this recently
      float TURN_FRONT_OUTER_MM  = 600.0;  // WIDE corner: run on further, then turn.
                                           // Default is the SAME as TURN_FRONT_MM, i.e.
                                           // the wide ARC is off. In sim it does buy an
                                           // outer-side green ~60 mm of clearance, but
                                           // it also exits at +339 against a limit of
                                           // 398, and across all layouts that cost 5
                                           // finished runs out of 24 in extra outer-wall
                                           // contacts. The "enter wide" half of the idea
                                           // (PRE_CORNER_SWING_MM) delivers most of the
                                           // benefit for none of that. Try 550 or 500 on
                                           // the real mat, where your radii differ.
      float TURN_LOCK_OUTER      = 0.70;
      float TURN_FRONT_INNER_MM  = 1050.0;  // ... start the arc earlier when exiting INNER
      float TURN_LOCK_INNER      = 1.00;   // ... and tighter
      float CORNER_EXIT_MM       = 0.0;
      float CORNER_EXIT_BIAS_MM  = 200.0;  // exit offset when the next straight's colour IS known
      float POST_CORNER_YAW_MAX  = 45.0;
      float POST_CORNER_BOOST_MM = 600.0;
      float PRE_CORNER_ZONE_MM   = 900.0;  // wall ahead closer than this = corner soon
                                           // (0 turns the pre-corner swing off)
      float PRE_CORNER_SWING_MM  = 250.0;  // how far to the entry side
      bool  CARRY_TRACKS         = true;

float firstSegmentCm      = 0.0;
float fullStartStraightCm = 0.0;
bool  haveFullStraight    = false;
float finalDistanceCm     = FINAL_STRAIGHT_CM;

// ============================================================
// FSM DATA
// ============================================================
bool fsmStarted = false;

RobotState    currentState = STATE_WAIT_START;
bool          entered = false;
unsigned long phaseT0 = 0;

BlockColor lockedColor   = COLOR_NONE;
bool       clockwiseMode = true;
int        cornerCount   = 0;

float targetHeading = 0.0;
float laneHeading   = 0.0;

BlockColor lastFirstColor = COLOR_NONE;

// cached IMU, refreshed every loop
bool          gImuFresh = false;
float         gHeading  = 0.0;
float         gYawRate  = 0.0;
float         gPrevH    = 0.0;
unsigned long gPrevHT   = 0;

// cached colour, refreshed every loop
BlockColor gRawColor = COLOR_NONE;

// ============================================================
// HELPERS
// ============================================================
float wrapDeg(float angle) {
  while (angle > 180.0)  angle -= 360.0;
  while (angle < -180.0) angle += 360.0;
  return angle;
}

void tcaselect(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void resetTCA() {
  pinMode(TCA_RST_PIN, OUTPUT);
  digitalWrite(TCA_RST_PIN, LOW);
  delay(10);
  digitalWrite(TCA_RST_PIN, HIGH);
  delay(10);
}

void setMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0)      { analogWrite(MOT_RPWM_PIN, speed); analogWrite(MOT_LPWM_PIN, 0); }
  else if (speed < 0) { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, -speed); }
  else                { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, 0); }
}

float lastServoCmd = SERVO_TRUE_STRAIGHT;

void setServoAngle(float angleDeg) {
  angleDeg = constrain(angleDeg, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  lastServoCmd = angleDeg;
  int pulse = (int)((angleDeg / 180.0) * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) + SERVO_MIN_PULSE_US;
  steeringServo.writeMicroseconds(pulse);
}

// TIM5 encoder, negated so driving forward counts up
void zeroEncoder() { TIM5->CNT = 0; }
long readEncoder() { return -(int32_t)TIM5->CNT; }
long absEnc(long v) { return v < 0 ? -v : v; }

bool turnIsClockwise() {
  return (lockedColor == COLOR_NONE) ? (lastFirstColor == COLOR_ORANGE) : clockwiseMode;
}
// Before the floor colour has been read, turnIsClockwise() is a GUESS (it
// returns false, i.e. anticlockwise). Anything that uses the turn direction to
// reject evidence has to know that, or on the first straight it throws away
// everything on one side for no reason.
bool directionKnown() {
  return lockedColor != COLOR_NONE || lastFirstColor != COLOR_NONE;
}
uint16_t turnSideMm() { return turnIsClockwise() ? lidarR : lidarL; }

// ---- IMU ----
float readYaw() {
  float qI = myIMU.getQuatI(), qJ = myIMU.getQuatJ();
  float qK = myIMU.getQuatK(), qReal = myIMU.getQuatReal();
  if (qI == 0.0f && qJ == 0.0f && qK == 0.0f && qReal == 0.0f) return 0.0f;
  float yawRadians = atan2(2.0f * (qI * qJ + qReal * qK),
                           (qReal * qReal + qI * qI - qJ * qJ - qK * qK));
  return yawRadians * (180.0 / PI);
}

float readHeading() {
  float h = fmod(readYaw() - initialYawOffset + 540.0, 360.0) - 180.0;
  return IMU_YAW_SIGN * h;
}

void zeroYaw() {
  Serial.println(F("# zeroing yaw"));
  unsigned long t = millis();
  while (millis() - t < 3000) {
    if (myIMU.wasReset()) myIMU.enableGameRotationVector();
    if (myIMU.getSensorEvent() && myIMU.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      initialYawOffset = readYaw();
      Serial.print(F("# zero yaw ")); Serial.println(initialYawOffset);
      return;
    }
    delay(10);
  }
  Serial.println(F("# ERROR no IMU event to zero"));
}

// ---- floor colour ----
void readColor(uint16_t &r, uint16_t &g, uint16_t &b, uint16_t &c) {
  if (!tcsOk) { r = 0; g = 0; b = 0; c = 0; return; }
  tcaselect(TCS_CH);
  Wire.beginTransmission(0x29);
  Wire.write(0x80 | 0x20 | 0x14);      // command | auto-increment | CDATAL
  Wire.endTransmission();
  Wire.requestFrom((uint8_t)0x29, (uint8_t)8);
  if (Wire.available() < 8) { r = 0; g = 0; b = 0; c = 0; return; }
  c  = (uint16_t)Wire.read();  c |= (uint16_t)Wire.read() << 8;
  r  = (uint16_t)Wire.read();  r |= (uint16_t)Wire.read() << 8;
  g  = (uint16_t)Wire.read();  g |= (uint16_t)Wire.read() << 8;
  b  = (uint16_t)Wire.read();  b |= (uint16_t)Wire.read() << 8;
}

BlockColor classifyColor() {
  uint16_t r, g, b, c;
  readColor(r, g, b, c);
  float total = (float)r + (float)g + (float)b;
  if (total < 100.0f) return COLOR_NONE;
  float pR = (r / total) * 100.0f;
  float pB = (b / total) * 100.0f;
  if (pR > 52.0f && pB < 18.0f) return COLOR_ORANGE;
  if (pB > 23.0f && pR < 40.0f) return COLOR_BLUE;
  return COLOR_NONE;
}

       unsigned long COLOR_CONFIRM_MS = 6;
BlockColor    pendingColor = COLOR_NONE;
unsigned long pendingStart = 0;

void resetColorDetector() { pendingColor = COLOR_NONE; pendingStart = 0; }

bool colorMuted    = false;
long colorMuteFrom = 0;

BlockColor detectColor(BlockColor wantColor) {
  if (colorMuted) { resetColorDetector(); return COLOR_NONE; }
  BlockColor rawColor = gRawColor;
  if (wantColor != COLOR_NONE && rawColor != wantColor) rawColor = COLOR_NONE;

  if (rawColor == COLOR_NONE)   { resetColorDetector(); return COLOR_NONE; }
  if (rawColor != pendingColor) { pendingColor = rawColor; pendingStart = millis(); return COLOR_NONE; }
  if (millis() - pendingStart >= COLOR_CONFIRM_MS) { resetColorDetector(); return rawColor; }
  return COLOR_NONE;
}

// ============================================================
// LANE POSITION + PILLAR PASS PLANNER
// ============================================================
// Replaces the old main.ino laws (wall PD on R-L, vision PD holding the
// pillar at +/-150 px). The simulator showed the pixel law cannot do the
// far case: holding a fixed image offset at close range makes the car
// ORBIT the pillar and hit it, and with the 160 deg lens a pillar is only
// big enough to react to in the last ~50 cm.
//
// Now everything is a LATERAL POSITION in the lane (mm, + = left of the
// lane centre):
//   car      laneOffMm   from the two cone wall fits (perpendicular mm)
//   pillar   track.lat   car lane position + the pillar's position in the
//                        car frame (Pi: camera bearing + LiDAR range),
//                        rotated by the car's yaw to the lane
// Red must be passed on its right -> car lat <= pillar lat - PASS_CLEAR
// Green on its left               -> car lat >= pillar lat + PASS_CLEAR
// The car aims for the lane position closest to the centre that satisfies
// every pillar it is approaching or still alongside, and steers there with
// a yaw command (heading = laneHeading + yaw) that the IMU PID tracks. The
// yaw points at a PASS POINT - the target lane position PASS_LEAD_MM before
// the pillar - so it sharpens as the pillar nears and arrives in time (a
// proportional law eases off near the target and arrived too late in sim).
// With no pillar in play the target is the lane centre - that IS the wall
// centring now.
       float    CORRIDOR_MM       = 1000.0;
       float    CAR_HALF_W_MM     = 57.0;
       float    PILLAR_HALF_MM    = 25.0;
       float    PASS_MARGIN_MM    = 80.0;    // air gap car side <-> pillar face
      float    PASS_CLEAR_MM     = 162.0;   // derived, see recomputeDerived()
       float    WALL_MARGIN_MM    = 45.0;
      float    LANE_LIMIT_MM     = 398.0;   // derived, see recomputeDerived()
       float    CENTRE_AIM_MM     = 600.0;   // centring: aim at the lane centre this far ahead
       float    PASS_LEAD_MM      = 160.0;   // be at the pass position this far BEFORE the pillar
                                            //   (half car length + pillar half + margin)
       float    PASS_AIM_MIN_MM   = 120.0;   // shortest aim distance (sharpest swerve)
       float    HOLD_AIM_MM       = 250.0;   // alongside a pillar: gentle hold
       float    OFF_JUMP_MM       = 120.0;   // lane offset can't move this much in one scan
       float    CENTRE_YAW_MAX    = 20.0;    // deg, plain centring
       float    PASS_YAW_MAX      = 75.0;    // deg, while a pillar is in play (sim: 45 too little
                                            //   for a far-side pillar; 75 reaches the most layouts)
       float    UNK_COMMIT_MM     = 400.0;   // unknown colour this close -> dodge to the roomier side
       float    TURN_RADIUS_MM    = 270.0;   // full-lock radius (worse side) for the reach estimate
       float    AVOID_MARGIN_MM   = 15.0;   // bare miss distance when no correct pass is reachable
       float    GIVEUP_LEAD_MM    = 60.0;    // reach is judged to this far before the pillar
       bool     ALLOW_GIVE_UP     = true;    // correct side unreachable -> pass on the other side
                                            //   rather than hit it (check your rulebook penalty)
       float    PLAN_MAX_AHEAD_MM = 1600.0;  // ignore sightings farther ahead
       float    PILLAR_MAX_LAT_MM = 420.0;   // seats are well inside the corridor; anything
                                            //   farther out is a pillar of ANOTHER straight seen
                                            //   across the corner (sim: one at 483 caused a crash)
       float    PASS_HOLD_MM      = 250.0;   // keep a pillar's side until it is this far behind the
                                            //   car CENTRE (tail ~82 mm + pillar half 25 + the tail's
                                            //   swing when the car turns back; 130 clipped the tail)
       float    TRACK_MATCH_MM    = 200.0;   // same pillar if within this (along and lateral)
       uint8_t  TRACK_CONFIRM     = 2;       // sightings before a pillar steers the car
       float    TRACK_FORGET_MM   = 300.0;   // unconfirmed and not seen for this far -> dropped
       uint16_t LANE_VALID_MAX_MM = 1100;    // cone farther than this = no wall on that side
      float    TICKS_PER_MM      = 1.4853;  // derived, see recomputeDerived()

struct PillarTrack { bool used; int color; float lat; float along; uint8_t hits; float lastSeen; bool flipped; };
const int   MAX_TRACKS = 4;
PillarTrack tracks[MAX_TRACKS];

float laneOffMm   = 0.0;     // + = car left of lane centre
bool  laneOffOk   = false;
float laneAlongMm = 0.0;     // distance along the lane since the last corner
long  laneAlongEnc = 0;
float latYawCmd   = 0.0;     // + = yaw left of laneHeading
float latTarget   = 0.0;
bool  passActive  = false;
bool  plannerEnabled = false;  // DRIVE / FINAL only - sightings mid-turn are in the wrong lane frame

void clearTracks() { for (int i = 0; i < MAX_TRACKS; i++) tracks[i].used = false; passActive = false; }

// ---- CROSS-CORNER SIGHTING --------------------------------------------
//
// The camera DOES see the next straight's first pillar - it just sees it early,
// from the current straight, across the corner (sim: the green after corner 1
// is in view from 590 mm out, several frames before the turn fires). Those
// sightings are thrown away today because their lane coordinates are nonsense
// in the current frame, and PILLAR_MAX_LAT_MM rejects them.
//
// Carrying the COORDINATES through the corner does not work: the car travels
// most of a corridor width during the arc, so rotating the estimate 90 deg
// about the car is wrong by hundreds of mm, and a confident wrong pillar is
// worse than none.
//
// But the coordinates are not what is needed. All the corner has to decide is
// WHICH SIDE to come out on, and that only needs the COLOUR:
//     red   must be passed on its right -> exit on the INNER side
//     green must be passed on its left  -> exit on the OUTER side
// which is one bit, and it survives the corner perfectly.
const float CROSS_MAX_LAT_MM = 1300.0;   // beyond this it is not the next straight

int   nextStraightColor = VIS_NONE;      // colour waiting after the next corner

// ---- SECONDARY PILLAR ------------------------------------------------------
//
// The camera sends TWO pillars now: the largest accepted blob, and the next
// largest. The second one is by definition further away, and on the run up to a
// corner "further away" almost always means "in the next straight" - the
// pillars of this straight are beside or behind the car by then.
//
// That single bit of colour is what decides the shape of the corner:
//
//   secondary GREEN -> it must be passed on the LEFT of its lane
//   secondary RED   -> it must be passed on the RIGHT of its lane
//
// and the corner is shaped to come out on that side. Clockwise, the right of
// the lane is the INNER side, so red wants the short, early, tight corner and
// green wants the wide one - go further forward, then turn. Anticlockwise both
// swap, because the inner wall is on the other hand.
//
// This is more reliable than inferring it from a sighting's lane coordinates:
// the camera only ever sent ONE pillar before, so a far pillar was invisible
// whenever a nearer one was bigger - which, on the run up to a corner, is
// exactly when it matters.
int   secSeenColor = VIS_NONE;           // last secondary colour worth believing
float secSeenAt    = -1e9;               // laneAlongMm when it was seen

// VIS_RED / VIS_GREEN / VIS_NONE - what is waiting on the next straight, from
// the second pillar if it has been seen recently enough, else from a
// cross-corner sighting. Used both to shape the corner and, before that, to
// decide which side to ENTER it from.
int plannedNextColour() {
  if (USE_SECONDARY_CORNER && secSeenColor != VIS_NONE &&
      laneAlongMm - secSeenAt <= SECONDARY_ZONE_MM) return secSeenColor;
  return nextStraightColor;
}

void clearSecondary() { secSeenColor = VIS_NONE; secSeenAt = -1e9; }

// Called once per new frame, from updatePlanner.
//
// "Further away" alone is NOT enough to mean "in the next straight". A straight
// with two pillars on it (and the rulebook allows several) also presents a
// nearer and a further one, and shaping the corner from the second pillar of
// the CURRENT straight would be worse than not shaping it at all.
//
// So the colour is only believed when the geometry agrees: the second pillar is
// put into this lane's frame exactly as addSighting() does, and it counts only
// if it lands OUTSIDE this corridor (|lat| > PILLAR_MAX_LAT_MM) on the side the
// car is about to turn towards. Anything inside the corridor is just another
// pillar on this straight, and the normal planner deals with it.
void trackSecondary() {
  if (!USE_SECONDARY_CORNER || secColor == VIS_NONE) return;
  if (lidarStale || !secXYValid || !laneOffOk) return;
  if (secX <= 0.0f) return;                        // behind the car

  if (pillarXYValid) {                             // must be the further of the two
    if (secX * secX + secY * secY <= pillarX * pillarX + pillarY * pillarY) return;
  }

  float yaw   = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
  float along = secX * cosf(yaw) - secY * sinf(yaw);
  float lat   = laneOffMm + secX * sinf(yaw) + secY * cosf(yaw);
  if (along <= 0.0f || fabs(lat) > CROSS_MAX_LAT_MM) return;
  if (fabs(lat) <= PILLAR_MAX_LAT_MM) return;      // still inside this corridor
  if (directionKnown()) {                          // else we cannot say which side
    bool towardTurn = turnIsClockwise() ? (lat < 0.0f) : (lat > 0.0f);
    if (!towardTurn) return;                       // the straight behind, not ahead
  }

  secSeenColor = secColor;
  secSeenAt    = laneAlongMm;
}
float cornerExitCmd     = 0.0;           // lane offset to hold for the first
                                         // POST_CORNER_BOOST_MM of the next
                                         // straight (+ = left, already signed)

void clearCrossCorner() { nextStraightColor = VIS_NONE; }

void planCornerExit();   // defined with the turn law, which owns the arc constants

// last pillar this straight that the car passed on the INNER side (CW: red,
// CCW: green) - decides the 3-point corner
bool  haveInnerPass   = false;
float lastInnerPassAt = 0.0;

void resetLaneAlong() { laneAlongMm = 0.0; laneAlongEnc = readEncoder();
                        haveInnerPass = false; clearCrossCorner(); clearSecondary(); }

// A pillar is "seen" when the latest line carries red/green and it is fresh.
bool pillarSeen() { return !lidarStale && visColor != VIS_NONE; }

bool laneOffsetRaw(float &off) {
  bool l = coneL <= LANE_VALID_MAX_MM, r = coneR <= LANE_VALID_MAX_MM;
  if (l && r) {
    float both = 0.5f * ((float)coneR - (float)coneL);
    float sum  = (float)coneL + (float)coneR;
    if (fabs(sum - CORRIDOR_MM) < 150.0f || !laneOffOk) { off = both; return true; }
    // walls don't add up to the corridor: one cone is fitted to something
    // else (a pillar beside the car). Keep the side that agrees with before.
    float fromL = CORRIDOR_MM / 2 - coneL, fromR = coneR - CORRIDOR_MM / 2;
    off = (fabs(fromL - laneOffMm) < fabs(fromR - laneOffMm)) ? fromL : fromR;
    return true;
  }
  if (l)      { off = CORRIDOR_MM / 2 - coneL;               return true; }
  if (r)      { off = coneR - CORRIDOR_MM / 2;               return true; }
  return false;
}

uint8_t offJumps = 0;
// The car moves < 40 mm sideways per LiDAR rev; a bigger jump is a bad fit.
// Accept it only if it persists for 3 revs (then it's real, e.g. after a corner).
bool laneOffset(float &off) {
  float o;
  if (!laneOffsetRaw(o)) { offJumps = 0; return false; }
  if (laneOffOk && fabs(o - off) > OFF_JUMP_MM && offJumps < 3) {
    if (lidarNewRev) offJumps++;
    return true;                                           // keep the previous value
  }
  offJumps = 0;
  off = o;
  return true;
}

void addSighting() {
  if (!pillarSeen() || !pillarXYValid || !laneOffOk) return;
  float yaw = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
  float along  = pillarX * cosf(yaw) - pillarY * sinf(yaw);
  float latRel = pillarX * sinf(yaw) + pillarY * cosf(yaw);
  float lat    = laneOffMm + latRel;
  if (along < -50.0f || along > PLAN_MAX_AHEAD_MM) return;
  if (fabs(lat) > PILLAR_MAX_LAT_MM) {
    // A pillar of ANOTHER straight, seen across the corner. Its position in
    // this lane frame is meaningless, so it must not steer the car here - but
    // if it lies on the side the car is about to turn towards, its colour says
    // which side the corner has to be exited on. See CROSS-CORNER SIGHTING.
    bool towardTurn = !directionKnown() ||
                      (turnIsClockwise() ? (lat < 0.0f) : (lat > 0.0f));
    if (towardTurn && along > 0.0f && fabs(lat) < CROSS_MAX_LAT_MM)
      nextStraightColor = visColor;
    return;
  }
  float at = laneAlongMm + along;

  int slot = -1, freeSlot = -1, oldest = 0;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used) { if (freeSlot < 0) freeSlot = i; continue; }
    if (tracks[i].color == visColor &&
        fabs(tracks[i].along - at) < TRACK_MATCH_MM && fabs(tracks[i].lat - lat) < TRACK_MATCH_MM) { slot = i; break; }
    if (tracks[i].along < tracks[oldest].along) oldest = i;
  }
  if (slot >= 0) {                                         // refine
    tracks[slot].lat   += 0.5f * (lat - tracks[slot].lat);
    tracks[slot].along += 0.5f * (at  - tracks[slot].along);
    tracks[slot].lastSeen = laneAlongMm;
    if (tracks[slot].hits < 255) tracks[slot].hits++;
    if (tracks[slot].hits == TRACK_CONFIRM) {
      Serial.print(F("# pillar ")); Serial.print(visColor == VIS_RED ? F("RED") : F("GREEN"));
      Serial.print(F(" lat=")); Serial.print((int)tracks[slot].lat);
      Serial.print(F(" at=")); Serial.println((int)tracks[slot].along);
    }
    return;
  }
  slot = (freeSlot >= 0) ? freeSlot : oldest;
  tracks[slot].used = true; tracks[slot].color = visColor;
  tracks[slot].lat = lat;   tracks[slot].along = at;
  tracks[slot].hits = 1;    tracks[slot].lastSeen = laneAlongMm; tracks[slot].flipped = false;
}

// Most sideways travel the car can make in s mm of lane, starting at yaw
// psi0 (rad, + = already angled TOWARD the target): arc at full lock up to
// PASS_YAW_MAX, then straight at that angle.
float latReach(float s, float psi0) {
  if (s <= 0) return 0;
  float R = TURN_RADIUS_MM, phi = PASS_YAW_MAX * DEG_TO_RAD;
  psi0 = constrain(psi0, -phi, phi);
  float aArc = R * (sinf(phi) - sinf(psi0));            // lane distance used by the arc
  if (s <= aArc) {                                      // still on the arc when we get there
    float sp = asinf(constrain(sinf(psi0) + s / R, -1.0f, 1.0f));
    return R * (cosf(psi0) - cosf(sp));
  }
  return R * (cosf(psi0) - cosf(phi)) + (s - aArc) * tanf(phi);
}

void updatePlanner() {
  // lane distance: encoder projected onto the lane direction
  long enc = readEncoder();
  float dmm = (enc - laneAlongEnc) / TICKS_PER_MM;
  laneAlongEnc = enc;
  laneAlongMm += dmm * cosf(wrapDeg(gHeading - laneHeading) * DEG_TO_RAD);

  if (!lidarNewFrame) return;
  laneOffOk = laneOffset(laneOffMm);
  if (!plannerEnabled) { latYawCmd = 0.0f; passActive = false; return; }
  addSighting();
  trackSecondary();

  // tracks in play: everything still within PASS_HOLD behind, up to and
  // including the NEAREST pillar ahead (farther ones wait their turn)
  float nearestAhead = 1e9;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel < -PASS_HOLD_MM) {                                        // passed
      if (tracks[i].hits >= TRACK_CONFIRM) {
        bool passRight = (tracks[i].color == VIS_RED) != tracks[i].flipped;
        if (passRight == clockwiseMode) {        // car went by on the INNER side
          haveInnerPass = true; lastInnerPassAt = tracks[i].along;
        }
      }
      tracks[i].used = false; continue;
    }
    if (tracks[i].hits < TRACK_CONFIRM) {                            // not trusted yet
      if (laneAlongMm - tracks[i].lastSeen > TRACK_FORGET_MM) tracks[i].used = false;
      continue;
    }
    if (rel > 0 && rel < nearestAhead) nearestAhead = rel;
  }
  float lo = -LANE_LIMIT_MM, hi = LANE_LIMIT_MM;
  float urgentRel = 1e9, urgentBound = 0; bool any = false;
  float aimMm = CENTRE_AIM_MM;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used || tracks[i].hits < TRACK_CONFIRM) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel > nearestAhead + 1.0f) continue;
    any = true;
    float bound;
    bool passRight = (tracks[i].color == VIS_RED) != tracks[i].flipped;   // car goes to the pillar's right
    if (passRight) { bound = tracks[i].lat - PASS_CLEAR_MM; if (bound < hi) hi = bound; }
    else           { bound = tracks[i].lat + PASS_CLEAR_MM; if (bound > lo) lo = bound; }
    if (rel < urgentRel) { urgentRel = rel; urgentBound = bound; }
  }
  passActive = any;

  // Default lane target with no pillar in play. Normally the lane centre, but
  // just after a corner it is CORNER_EXIT_MM and just before one it is
  // PRE_CORNER_SWING_MM. Both are signed "+ = OUTER side of the lap", so one
  // number works for clockwise and anticlockwise.
  float target = 0.0f;
  if (!any) {
    if (cornerCount > 0 && laneAlongMm < POST_CORNER_BOOST_MM)
      target = cornerExitCmd;
    else if (PRE_CORNER_ZONE_MM > 0.0f && !lidarStale && lidarF < PRE_CORNER_ZONE_MM) {
      // Enter wide to leave tight. A 90 deg arc throws the car AWAY from the
      // side it started on, so the way to come out of a corner near the inner
      // wall is to go into it near the outer one - the ordinary racing line.
      // The exit side is already known here (the second pillar told us before
      // the corner), so the entry is simply its opposite.
      int nxt = plannedNextColour();
      float exitSide = (nxt == VIS_NONE) ? (clockwiseMode ? 1.0f : -1.0f)
                                         : (nxt == VIS_RED ? -1.0f : 1.0f);
      target = -exitSide * PRE_CORNER_SWING_MM;
    }
  }
  if (lo > hi) target = urgentBound;                     // conflict: most urgent pillar wins
  else         target = constrain(target, lo, hi);
  if (any) {
    // aim at the pass point: target lane position, PASS_LEAD before the
    // nearest pillar ahead - so the swerve gets sharper as it gets closer
    // and arrives in time. Alongside / past it: gentle hold.
    if (nearestAhead < 1e8 && nearestAhead > PASS_LEAD_MM)
      aimMm = fmaxf(nearestAhead - PASS_LEAD_MM, PASS_AIM_MIN_MM);
    else
      aimMm = HOLD_AIM_MM;

    // Reach check: can the car still get to the pass position before the
    // nearest pillar? If not, commit to its other side instead of hitting it.
    if (ALLOW_GIVE_UP && laneOffOk && nearestAhead < 1e8 && nearestAhead > 0) {
      float need = fabs(target - laneOffMm);
      float sAvail = nearestAhead - GIVEUP_LEAD_MM;
      float yawNow = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;       // + = left
      float psi0 = (target > laneOffMm) ? yawNow : -yawNow;               // + = toward target
      if (need > 60.0f && need > latReach(sAvail, psi0) + 30.0f) {
        bool flipped = false;
        for (int i = 0; i < MAX_TRACKS; i++) {
          if (!tracks[i].used || tracks[i].hits < TRACK_CONFIRM || tracks[i].flipped) continue;
          if (fabs((tracks[i].along - laneAlongMm) - nearestAhead) > 1.0f) continue;
          bool passRight = (tracks[i].color == VIS_RED);
          float other = passRight ? tracks[i].lat + PASS_CLEAR_MM : tracks[i].lat - PASS_CLEAR_MM;
          float needO = fabs(other - laneOffMm);
          float psiO  = (other > laneOffMm) ? yawNow : -yawNow;
          // switch only if the other side can really be reached from here -
          // late in the swerve it can't, and swinging back is worse
          if (fabs(other) <= LANE_LIMIT_MM && needO < need && needO <= latReach(sAvail, psiO)) {
            tracks[i].flipped = true;
            flipped = true;
            target = other;
            Serial.print(F("# give up: ")); Serial.print(tracks[i].color == VIS_RED ? F("RED") : F("GREEN"));
            Serial.print(F(" unreachable, need ")); Serial.print((int)need);
            Serial.print(F(" in ")); Serial.println((int)sAvail);
          }
        }

        // NEITHER side reachable with full clearance. Holding the unreachable
        // target here drives the car straight into the pillar - the sim did
        // exactly that on a red 160 mm past a corner. A wrong-side pass costs
        // points; a collision costs the run. So fall back to the bare
        // geometric miss distance (no PASS_MARGIN) and take whichever side is
        // closer to reach, correct or not.
        if (!flipped) {
          float avoidClear = PILLAR_HALF_MM + CAR_HALF_W_MM + AVOID_MARGIN_MM;
          for (int i = 0; i < MAX_TRACKS; i++) {
            if (!tracks[i].used || tracks[i].hits < TRACK_CONFIRM) continue;
            if (fabs((tracks[i].along - laneAlongMm) - nearestAhead) > 1.0f) continue;
            float a = tracks[i].lat - avoidClear, b = tracks[i].lat + avoidClear;
            float na = fabs(a - laneOffMm),       nb = fabs(b - laneOffMm);
            float pick = (na <= nb) ? a : b;
            if (fabs(pick) > LANE_LIMIT_MM) pick = (na <= nb) ? b : a;   // that side is wall
            if (fabs(pick - laneOffMm) < need) {
              target = pick;
              Serial.print(F("# AVOID only: pillar at ")); Serial.print((int)nearestAhead);
              Serial.print(F(" mm, aiming ")); Serial.println((int)pick);
            }
          }
        }
      }
    }
  }

  // ---- LiDAR object whose colour is still unknown ----
  // The camera only covers ~+/-48 deg, so a pillar near the far wall can be
  // out of view until too late. Line up with it (it comes into view and both
  // passing sides stay open); if it is still unnamed at UNK_COMMIT_MM, dodge
  // to the side with more room so we never drive into it.
  if (unknownXYValid && laneOffOk) {
    float yaw = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
    float uAlong = unknownX * cosf(yaw) - unknownY * sinf(yaw);
    float uLat   = laneOffMm + unknownX * sinf(yaw) + unknownY * cosf(yaw);
    bool known = false;                                  // same place as a named pillar?
    for (int i = 0; i < MAX_TRACKS; i++)
      if (tracks[i].used && tracks[i].hits >= TRACK_CONFIRM &&
          fabs(tracks[i].along - (laneAlongMm + uAlong)) < TRACK_MATCH_MM &&
          fabs(tracks[i].lat - uLat) < TRACK_MATCH_MM) known = true;
    if (!known && uAlong > 100.0f && uAlong < PLAN_MAX_AHEAD_MM &&
        fabs(uLat) < PILLAR_MAX_LAT_MM && uAlong < nearestAhead - 150.0f) {
      // only pillars already alongside may constrain us now
      float lo2 = -LANE_LIMIT_MM, hi2 = LANE_LIMIT_MM;
      for (int i = 0; i < MAX_TRACKS; i++) {
        if (!tracks[i].used || tracks[i].hits < TRACK_CONFIRM || tracks[i].along - laneAlongMm > 50.0f) continue;
        if ((tracks[i].color == VIS_RED) != tracks[i].flipped) hi2 = fminf(hi2, tracks[i].lat - PASS_CLEAR_MM);
        else                                                  lo2 = fmaxf(lo2, tracks[i].lat + PASS_CLEAR_MM);
      }
      float t = (uAlong > UNK_COMMIT_MM) ? uLat
              : (uLat < 0 ? uLat + PASS_CLEAR_MM : uLat - PASS_CLEAR_MM);
      target = (lo2 <= hi2) ? constrain(t, lo2, hi2) : t;
      aimMm  = (uAlong > UNK_COMMIT_MM) ? fmaxf(uAlong - UNK_COMMIT_MM, 250.0f)
                                        : fmaxf(uAlong - PASS_LEAD_MM, PASS_AIM_MIN_MM);
      passActive = true;
    }
  }
  latTarget = constrain(target, -LANE_LIMIT_MM, LANE_LIMIT_MM);

  if (!laneOffOk) { latYawCmd = 0.0f; return; }          // no walls: just hold heading
  // Centring authority is deliberately gentle (CENTRE_YAW_MAX) so the car does
  // not weave down a straight. Right after a corner that is the wrong trade:
  // the car has a whole lane width to recover and only a few hundred mm to do
  // it in, so it gets PASS-level authority for the first stretch.
  float ymax = passActive ? PASS_YAW_MAX
             : (laneAlongMm < POST_CORNER_BOOST_MM ? POST_CORNER_YAW_MAX
                                                   : CENTRE_YAW_MAX);
  // + lateral error = target is to the LEFT = yaw left
  latYawCmd = constrain(atan2f(latTarget - laneOffMm, aimMm) / DEG_TO_RAD, -ymax, ymax);
}

// ============================================================
// LEVELLING  - pull the IMU lane heading onto the fitted wall direction
// ============================================================
// The Pi fits both walls in 45 deg cones and reports the car's yaw
// relative to them (wallAngDeg, + = pointing left). The lane direction is
// then gHeading - wallAngDeg. That estimate is noisy per rev but has no
// drift, the IMU is smooth but drifts - so the lane heading is nudged a
// small step toward it once per rev. Only while the fit is trustworthy:
// on a straight (past the post-corner lockout), both cone walls inside a
// corridor width, no pillar close enough to be steering the car, and the
// estimate within LEVEL_MAX_DIFF of what the IMU already believes (a big
// disagreement is a bad fit, not drift).
       float LEVEL_GAIN        = 0.05;   // fraction of the error removed per rev (10 Hz)
       float LEVEL_MAX_STEP    = 0.3;    // deg per rev, hard cap
       float LEVEL_MAX_DIFF    = 8.0;    // deg - reject bigger disagreements
       float LEVEL_MAX_WALLANG = 20.0;   // deg - car too yawed for a clean fit

bool  levelEnabled   = false;           // set per state; off in turns / recover
float levelTotalDeg  = 0.0;             // running total, logged per corner

void updateLevel() {
  if (!levelEnabled || !lidarNewRev || lidarStale || !wallAngValid) return;   // gHeading = latest IMU
  if (coneL > LANE_VALID_MAX_MM || coneR > LANE_VALID_MAX_MM) return;
  if (fabs(wallAngDeg) > LEVEL_MAX_WALLANG) return;
  if (passActive) return;                  // swerving round a pillar: not level

  float est  = wrapDeg(gHeading - wallAngDeg);
  float diff = wrapDeg(est - laneHeading);
  if (fabs(diff) > LEVEL_MAX_DIFF) return;

  float step = constrain(LEVEL_GAIN * diff, -LEVEL_MAX_STEP, LEVEL_MAX_STEP);
  laneHeading   = wrapDeg(laneHeading + step);
  targetHeading = laneHeading;
  levelTotalDeg += step;
}

// ---- called once at the top of every loop ----
void serviceSensors() {
  lidarNewFrame = false;
  lidarNewRev   = false;
  serviceLidar();

  gImuFresh = false;
  if (myIMU.wasReset()) myIMU.enableGameRotationVector();
  if (myIMU.getSensorEvent() &&
      myIMU.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
    gImuFresh = true;
    float h = readHeading();
    unsigned long now = millis();
    float dt = (now - gPrevHT) / 1000.0;
    if (dt > 0.0) gYawRate = wrapDeg(h - gPrevH) / dt;
    gPrevH = h; gPrevHT = now;
    gHeading = h;
  }

  gRawColor = classifyColor();
  if (currentState != STATE_FINISHED) {
    digitalWrite(LED2_PIN, gRawColor == COLOR_ORANGE ? HIGH : LOW);
    digitalWrite(LED3_PIN, gRawColor == COLOR_BLUE   ? HIGH : LOW);
  }

  updatePlanner();
  updateLevel();
}

// ============================================================
// SYSTEM INITIALIZATION
// ============================================================
void initHardware() {
  pinMode(MOT_RPWM_PIN, OUTPUT);
  pinMode(MOT_LPWM_PIN, OUTPUT);
  setMotorSpeed(0);

  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(LED3_PIN, OUTPUT);
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(LED3_PIN, LOW);

  steeringServo.attach(SERVO_PIN, SERVO_MIN_PULSE_US, SERVO_MAX_PULSE_US);
  setServoAngle(SERVO_TRUE_STRAIGHT);

  // ---- TIM5 encoder on PA0 / PA1 (AF2) ----
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_TIM5_CLK_ENABLE();
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  GPIO_InitStruct.Pin       = GPIO_PIN_0 | GPIO_PIN_1;
  GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull      = GPIO_PULLUP;
  GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF2_TIM5;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  TIM_Encoder_InitTypeDef sConfig = {0};
  static TIM_HandleTypeDef htim5 = {0};
  htim5.Instance         = TIM5;
  htim5.Init.Prescaler   = 0;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period      = 0xFFFFFFFF;
  sConfig.EncoderMode  = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity  = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Polarity  = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  HAL_TIM_Encoder_Init(&htim5, &sConfig);
  HAL_TIM_Encoder_Start(&htim5, TIM_CHANNEL_ALL);

  // ---- I2C + colour ----
  resetTCA();
  Wire.setSCL(I2C_SCL);
  Wire.setSDA(I2C_SDA);
  Wire.begin();
  Wire.setClock(400000);
  delay(100);

  tcaselect(TCS_CH);
  delay(10);
  tcsOk = tcs.begin();
  Serial.println(tcsOk ? F("# colour CH4 READY") : F("# colour CH4 FAILED"));

  // ---- IMU over SPI1 ----
  SPI_IMU.begin();
  if (myIMU.beginSPI(IMU_CS_PIN, IMU_INT_PIN, IMU_RST_PIN, 3000000, SPI_IMU)) {
    delay(500);
    myIMU.enableGameRotationVector();
    delay(100);
    myIMU.getSensorEvent();
    zeroYaw();
  } else {
    Serial.println(F("# ERROR IMU not found"));
  }
}

// ============================================================
// DRIVE STEERING  = heading PID (with wall-centre offset) blended with
//                   the pillar PD by pillar area - old main.ino blend
// ============================================================
       float HEAD_KP        = 2.0;
       float YAW_FILT_ALPHA = 0.35;
       float SERVO_SLEW     = 2.5;     // servo deg per IMU update (~100 Hz)
       float INTEGRAL_CLAMP = 300.0;
       float HEAD_KI        = 0.0;
       float HEAD_KD        = 0.0;

unsigned long pidPrevTime  = 0;
float         pidIntegral  = 0.0;
float         yawFilt      = 0.0;
float         prevServoCmd = SERVO_TRUE_STRAIGHT;

void resetHeadingPid() {
  pidPrevTime  = millis();
  pidIntegral  = 0.0;  yawFilt = 0.0;
  prevServoCmd = SERVO_TRUE_STRAIGHT;
}

// usePlanner = follow the lane-position planner (centring + pillars);
// false = plain heading hold on laneH.
void updateDriveSteer(float laneH, bool usePlanner) {
  if (!gImuFresh) return;
  unsigned long now = millis();
  yawFilt += YAW_FILT_ALPHA * (gYawRate - yawFilt);
  float dt = (now - pidPrevTime) / 1000.0;
  if (dt <= 0.0) dt = 0.001;

  float target = usePlanner ? wrapDeg(laneH + latYawCmd) : laneH;   // + = left
  float error  = wrapDeg(target - gHeading);
  pidIntegral += error * dt;
  pidIntegral  = constrain(pidIntegral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
  float correction = HEAD_KP * error + HEAD_KI * pidIntegral - HEAD_KD * yawFilt;

  float want = SERVO_TRUE_STRAIGHT - correction;             // below straight = left
  float dcmd = constrain(want - prevServoCmd, -SERVO_SLEW, SERVO_SLEW);
  float cmd  = constrain(prevServoCmd + dcmd, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  if (cmd <= SERVO_MAX_LEFT || cmd >= SERVO_MAX_RIGHT) pidIntegral -= error * dt;
  setServoAngle(cmd);
  prevServoCmd = cmd;
  pidPrevTime  = now;
}

// ============================================================
// NAV STEERING  - track the Pi's planned line
// ============================================================
// Three terms, each doing one job:
//
//   feedforward  atan(L * curvature) is the steer angle that HOLDS the
//                planned curvature in steady state. Without it the tracker
//                has to build a standing cross-track error before it
//                generates any steering at all - for a pure-pursuit law
//                that error is L^2/(2R), which on these corners is about
//                78 mm, i.e. the entire passing margin.
//   heading      closes the angle between the car and the line.
//   cross-track  closes what is left of the offset.
//
// The servo slew limit and the end stops are shared with the old law, so
// the actuator behaves identically either way.
void updateNavSteer() {
  if (!gImuFresh) return;
  float ff    = atanf(NAV_WHEELBASE_MM * navCurv);
  float delta = ff
              + NAV_KP_HEAD * (navHdgDeg * DEG_TO_RAD)
              - atanf(NAV_KP_CROSS * (float)navCross);
  float dDeg  = constrain(delta / DEG_TO_RAD, -NAV_MAX_STEER, NAV_MAX_STEER);

  // front-wheel angle -> servo, using the MEASURED asymmetric linkage
  // (full lock: 27 cm radius left, 25 cm right; 135.85 mm wheelbase)
  const float kL = 26.7f / (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT);
  const float kR = 28.5f / (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT);
  float want = (dDeg >= 0.0f) ? (SERVO_TRUE_STRAIGHT - dDeg / kL)
                              : (SERVO_TRUE_STRAIGHT - dDeg / kR);

  float dcmd = constrain(want - prevServoCmd, -SERVO_SLEW, SERVO_SLEW);
  float cmd  = constrain(prevServoCmd + dcmd, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  setServoAngle(cmd);
  prevServoCmd = cmd;
}

// ============================================================
// EASED TURN LAW - wider arc than the open round
// ============================================================
// Full lock is 27 cm (left) / 25 cm (right). The obstacle round needs
// room for pillars in the corner section, so the arc is capped at
// TURN_LOCK_FRACTION of each side's travel (0.70 -> roughly 35-38 cm
// radius; measure on the mat and tune this one number). The last
// TURN_MAX_STEER / TURN_KP degrees of heading error still ease off.
       float TURN_LOCK_FRACTION   = 0.70;
       float TURN_KP              = 2.5;
      float TURN_MAX_STEER_LEFT  = 39.6;    // derived, see recomputeDerived()
      float TURN_MAX_STEER_RIGHT = 44.5;    // derived, see recomputeDerived()
       float TURN_MIN_STEER       = 8.0;
#define     TURN_PWM               DRIVE_PWM
       float TURN_STOP_DEG        = 15.0;   // hand over to DRIVE this close to the new heading:
                                            //   the heading PID finishes the last degrees while the
                                            //   pillar planner is already live. (sim: finishing the
                                            //   arc to 0.3 deg took ~600 mm of the next straight and
                                            //   left a pillar there no room)
       float TURN_CAP_CM          = 150.0;   // odometry backstop (wider arc = longer path)

// Where the arc should leave the car, and the arc that gets it there.
//
// Measured in the sim (bare lap, exit offset just after each corner, + = outer,
// lane limit +/-398):
//
//   TURN_FRONT_MM  0.55   0.70   0.85   1.00   <- TURN_LOCK_FRACTION
//         500      +382   +339   +282   +241
//         600      +335   +241   +190   +150
//         750      +194   +108    +51    -27
//         900       +48    -41   -109   -160
//
// So the exit is commandable across the whole corridor: firing the arc while
// the wall ahead is still far, and tightening it, walks the car from the outer
// wall to the inner one. Nothing else the car does afterwards has that
// authority - a pillar 200 mm past the corner leaves no room to cross - which
// is why this is decided BEFORE the arc, not after.
float turnFrontCmd = 600.0;      // the TURN_FRONT_MM actually used this corner
float turnLockCmd  = 0.70;       // the TURN_LOCK_FRACTION actually used

void planCornerExit() {
  float outer = clockwiseMode ? 1.0f : -1.0f;      // + lane offset = left of centre

  // Which colour is waiting on the next straight? The SECONDARY pillar is the
  // direct evidence - a pillar the camera can still see that is further away
  // than the one being passed - so it wins. The cross-corner sighting (a
  // primary pillar whose lane coordinates put it outside this corridor) is the
  // fallback for a frame where only one pillar was visible at all.
  int want = plannedNextColour();
  const char *src = (USE_SECONDARY_CORNER && secSeenColor != VIS_NONE &&
                     laneAlongMm - secSeenAt <= SECONDARY_ZONE_MM)
                    ? " (2nd pillar)" : " (across the corner)";

  bool inner = false;
  if (want != VIS_NONE) {
    // The passing rule is about the LANE, not the lap: red is passed on its
    // right, which is the right-hand side of the lane (negative offset),
    // whichever way round the car is going. Whether that side is the inner or
    // the outer one then depends on the direction - clockwise the inner wall
    // is on the right, anticlockwise it is on the left. Tying red to "inner"
    // was right for CW and exactly backwards for CCW.
    cornerExitCmd = (want == VIS_RED ? -1.0f : 1.0f) * CORNER_EXIT_BIAS_MM;
    inner = ((cornerExitCmd > 0.0f) != (outer > 0.0f));
    Serial.print(want == VIS_RED ? F("# next straight RED") : F("# next straight GREEN"));
    Serial.print(src);
    Serial.println(inner ? F(" -> SHORT corner, exit INNER")
                         : F(" -> WIDE corner, exit OUTER"));
  } else {
    cornerExitCmd = outer * CORNER_EXIT_MM;        // nothing seen: the default
  }

  // Three shapes, not two:
  //   INNER   turn early and tight  - the short, near corner
  //   OUTER   run on, then turn on a looser arc - the wide corner
  //   unknown the middle setting, which exits near the lane centre and leaves
  //           both sides reachable
  if (!CORNER_ARC_ADAPT || want == VIS_NONE) {
    turnFrontCmd = TURN_FRONT_MM;
    turnLockCmd  = TURN_LOCK_FRACTION;
  } else if (inner) {
    turnFrontCmd = TURN_FRONT_INNER_MM;            // larger = fires further out = earlier
    turnLockCmd  = TURN_LOCK_INNER;
  } else {
    turnFrontCmd = TURN_FRONT_OUTER_MM;            // smaller = runs on before turning
    turnLockCmd  = TURN_LOCK_OUTER;
  }
}

long turnStartTicks = 0;
long turnCapTicks   = 0;

bool turnArcStep(float target) {
  if (absEnc(readEncoder() - turnStartTicks) >= turnCapTicks) return true;
  if (gImuFresh) {
    float err = wrapDeg(target - gHeading);
    if (fabs(err) < TURN_STOP_DEG) return true;
    float mag      = fabs(err);
    float maxSteer = (err > 0) ? (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT)  * turnLockCmd
                               : (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT) * turnLockCmd;
    float steer    = constrain(TURN_KP * mag, TURN_MIN_STEER, maxSteer);
    float servo    = (err > 0) ? (SERVO_TRUE_STRAIGHT - steer) : (SERVO_TRUE_STRAIGHT + steer);
    setServoAngle(servo);
    setMotorSpeed(TURN_PWM);
  }
  return false;
}

// ============================================================
// STATE HELPERS + working vars
// ============================================================
void goState(RobotState s) { currentState = s; entered = false; }
bool innerPillarNearCorner();       // corner maneuver decision, defined with its state

BlockColor dcWantColor;
bool       dcColorArmed;
long       dcBaseTicks;
long       dcLockoutTicks;
long       dcSafetyTicks;
bool       dcTurnArmed;          // corner trigger seen; TURN_DELAY_MM still to run
long       dcTurnArmTicks;
bool       dcTurnManeuver;       // decided at the arm point, acted on after the delay

float turnTarget;
float turnAmount;

long    fsTargetTicks;

void finishCorner() {
  targetHeading = laneHeading;
  if (cornerCount >= TARGET_CORNERS) goState(STATE_FINAL_STRAIGHT);
  else                               goState(STATE_DRIVE_TO_CORNER);
}

// ============================================================
// STATE: RECOVER  (wall too close - back off, then resume)
// Same as the open round. Not entered while the camera sees a pillar -
// a short front reading is then the pillar, and the pillar PD handles it.
// ============================================================
RobotState recoverReturnState   = STATE_DRIVE_TO_CORNER;
bool       recoverReturnEntered = false;
long       recoverBaseTicks     = 0;
int        recoverTries         = 0;

void enterRecovery() {
  recoverReturnState   = currentState;
  recoverReturnEntered = entered;
  levelEnabled  = false;             // reversing near a wall: no levelling
  plannerEnabled = false;
  colorMuted    = true;
  colorMuteFrom = readEncoder();
  currentState  = STATE_RECOVER;
  entered = false;
}

void recoverStep() {
  if (!entered) {
    entered = true;
    setMotorSpeed(0);
    setServoAngle(2.0 * SERVO_TRUE_STRAIGHT - lastServoCmd);
    setMotorSpeed(-RECOVER_PWM);
    recoverBaseTicks = readEncoder();
    Serial.print(F("# RECOVER front=")); Serial.println(lidarF);
  }

  bool clear     = !lidarStale && (lidarF >= WALL_CLEAR_MM);
  bool backedFar = absEnc(readEncoder() - recoverBaseTicks)
                   >= (long)(RECOVER_MAX_CM * TICKS_PER_CM);
  if (!clear && !backedFar) return;

  setMotorSpeed(0);
  if (clear) { recoverTries = 0;  Serial.println(F("# recover clear")); }
  else       { recoverTries++;    Serial.print(F("# recover capped, try "));
               Serial.println(recoverTries); }

  currentState = recoverReturnState;
  entered      = recoverReturnEntered;
  resetHeadingPid();

  switch (recoverReturnState) {
    case STATE_DRIVE_TO_CORNER:
    case STATE_FINAL_STRAIGHT:
      setMotorSpeed(DRIVE_PWM);
      break;
    default:
      break;     // TURNING sets its own PWM each step
  }
}

// ============================================================
// STATE: WAIT START  (armed - waits for START from the Pi page)
// ============================================================
void waitStartStep() {
  if (!entered) {
    entered = true;
    startRequested = false;          // a START sent before arming is ignored
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    Serial.println(F("# WAIT_START armed - press Start on the Pi page"));
  }
  digitalWrite(LED1_PIN, ((millis() / 250) & 1) ? HIGH : LOW);

  if (!startRequested) return;
  startRequested = false;

  if (lidarStale) {
    Serial.println(F("# START refused: lidar stale"));
    return;
  }

  digitalWrite(LED1_PIN, HIGH);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(LED3_PIN, LOW);
  lockedColor      = COLOR_NONE;
  lastFirstColor   = COLOR_NONE;
  clockwiseMode    = true;
  cornerCount      = 0;
  colorMuted       = false;
  recoverTries     = 0;
  firstSegmentCm   = 0.0;
  fullStartStraightCm = 0.0;
  haveFullStraight = false;
  finalDistanceCm  = FINAL_STRAIGHT_CM;
  laneHeading      = gHeading;
  targetHeading    = laneHeading;
  levelEnabled     = false;          // DRIVE turns it on
  levelTotalDeg    = 0.0;
  cornerExitCmd    = 0.0;
  clearSecondary();
  clearTracks();
  resetLaneAlong();
  zeroEncoder();
  Serial.println(F("# GO"));
  goState(STATE_DRIVE_TO_CORNER);
}

// ============================================================
// STATE: DRIVE TO CORNER
// Same corner logic as the open round; steering is now the blended
// heading + wall-centre + pillar law.
// ============================================================
void driveStep() {
  if (!entered) {
    entered = true;
    dcWantColor    = lockedColor;
    dcColorArmed   = false;
    dcTurnArmed    = false;
    dcTurnManeuver = false;
    sideOpenCount  = 0;
    sideWallCount  = 0;
    resetColorDetector();
    resetHeadingPid();
    dcBaseTicks = readEncoder();
    setMotorSpeed(DRIVE_PWM);
    Serial.println(F("# DRIVE"));

    dcLockoutTicks = (cornerCount > 0) ? (long)(POST_CORNER_LOCKOUT_CM * TICKS_PER_CM) : 0;
    dcSafetyTicks  = (long)(SEARCH_SAFETY_CM * TICKS_PER_CM);
  }

  long straightTicks = absEnc(readEncoder() - dcBaseTicks);
  if (straightTicks >= dcSafetyTicks) {
    Serial.println(F("# WARN no turn trigger within safety distance, retrying"));
    entered = false;
    return;
  }

  // ---- NAV PATH ----
  // When the Pi's plan is live it owns the whole lap, corners included: its
  // reference line already goes round them, so this state never hands over
  // to TURNING and the SIDE_OPEN_MM corner trigger is not consulted at all.
  // That trigger is the thing that guesses where the corner is from a single
  // side beam; the plan knows.
  //
  // The instant navValid() goes false the code below this block runs exactly
  // as it always did, from the same state, with the same variables. There is
  // no mode flag to get stuck in and no re-flash to revert.
  if (navValid()) {
    plannerEnabled = false;          // the Pi is planning; don't fight it
    levelEnabled   = false;          // the pose is absolute, nothing to level
    updateNavSteer();
    setMotorSpeed(DRIVE_PWM);
    if (navDone) {
      Serial.println(F("# nav: 3 laps complete"));
      setMotorSpeed(0);
      setServoAngle(SERVO_TRUE_STRAIGHT);
      goState(STATE_FINISHED);
    }
    return;
  }

  // Vision is live from the first tick after the arc, lockout included -
  // a pillar right after the corner is handled immediately.
  plannerEnabled = true;
  updateDriveSteer(targetHeading, true);

  if (absEnc(readEncoder()) <= dcLockoutTicks) { levelEnabled = false; return; }
  levelEnabled = true;

  bool directionLocked = (lockedColor != COLOR_NONE);

  // ---- 1. wall evidence ----
  // This MUST run before the colour gate's early return.
  //
  // sideWallCount is the "I have seen the inner wall on this straight" proof
  // that stops a false corner. On the FIRST corner the colour gate does not arm
  // until the car crosses the orange line - which on this mat is already past
  // the end of the inner wall. Counting only after the gate therefore never saw
  // the wall, sideOpenCount could never rise, and the turn never fired: on an
  // empty mat the car drove into the far wall and sat in RECOVER forever. It
  // only appeared to work when a pillar happened to sit to the turn side and
  // stood in for the wall.
  uint16_t sideNow = turnSideMm();
  bool aligned = fabs(wrapDeg(gHeading - laneHeading)) < TURN_TRIGGER_MAX_YAW;
  if (lidarStale || !aligned) {
    sideOpenCount = 0;
  } else if (lidarNewFrame) {
    if (sideNow > SIDE_OPEN_MM) {
      if (sideWallCount >= SIDE_WALL_FRAMES && sideOpenCount < 250) sideOpenCount++;
    } else {
      sideOpenCount = 0;
      if (sideWallCount < 250) sideWallCount++;
    }
  }

  // ---- 2. colour gate: first corner, or lidar-dead fallback ----
  if (!dcColorArmed && (!directionLocked || lidarDead)) {
    BlockColor c = detectColor(dcWantColor);
    if (c != COLOR_NONE) {
      dcColorArmed = true;
      if (!directionLocked) lastFirstColor = c;
      sideOpenCount = 0;
      resetColorDetector();
      Serial.print(F("# gate ")); Serial.print(c == COLOR_ORANGE ? F("ORANGE") : F("BLUE"));
      Serial.print(F(" side=")); Serial.println(turnSideMm());
    }
  }
  if (!directionLocked && !dcColorArmed) return;

  // ---- 3. turn confirm ----
  bool sideOpen = (sideOpenCount >= SIDE_OPEN_FRAMES);

  // Arm on the trigger, turn TURN_DELAY_MM later. The corner measurements are
  // taken at the ARM point (that is where the corner really is); the delay only
  // moves where the arc starts, which is what sets the exit lane position.
  if (!dcTurnArmed && (sideOpen || (lidarDead && dcColorArmed))) {
    dcTurnArmed = true;
    dcTurnArmTicks = readEncoder();
    if (sideOpen) { Serial.print(F("# turn: side open ")); Serial.println(sideNow); }
    else          Serial.println(F("# turn: colour only (lidar dead)"));

    if (!directionLocked) {
      lockedColor   = lastFirstColor;
      clockwiseMode = (lockedColor == COLOR_ORANGE);
      Serial.println(clockwiseMode ? F("# LOCKED CW (orange)") : F("# LOCKED CCW (blue)"));
    }

    float segCm = absEnc(readEncoder()) / TICKS_PER_CM;
    if (cornerCount == 0) {
      firstSegmentCm = segCm;
      Serial.print(F("# A=")); Serial.println(firstSegmentCm);
    } else if (cornerCount == 4) {
      fullStartStraightCm = segCm; haveFullStraight = true;
    } else if (cornerCount == 8 && haveFullStraight) {
      fullStartStraightCm = 0.5 * (fullStartStraightCm + segCm);
    }
    if (haveFullStraight) {
      finalDistanceCm = fullStartStraightCm - firstSegmentCm;
      if (finalDistanceCm < 0) finalDistanceCm = 0;
      Serial.print(F("# L=")); Serial.print(fullStartStraightCm);
      Serial.print(F(" final=")); Serial.println(finalDistanceCm);
    }

    planCornerExit();                 // needs nextStraightColor, still valid here
    dcTurnManeuver = innerPillarNearCorner();
  }

  if (!dcTurnArmed) return;

  // ---- 4. when to actually start the arc ----
  // The turn side opening says the corner is HERE; it does not say the car is
  // deep enough into the corner square to arc. Firing on the opening alone
  // starts the 90 deg arc at the mouth of the square and lands the car hard on
  // the inner wall of the next straight.
  //
  // The physical cue is the wall AHEAD: start the arc when it is about one
  // turn radius away. TURN_FRONT_MM is that distance (0 disables it and the
  // odometry delay alone decides). TURN_DELAY_MM is a minimum run-out after
  // arming, and TURN_ARM_MAX_MM is the backstop for a stale or missing front
  // reading so the car can never coast past the corner waiting for a cue.
  float sinceArm = absEnc(readEncoder() - dcTurnArmTicks) / TICKS_PER_MM;
  if (sinceArm < TURN_DELAY_MM) return;
  if (turnFrontCmd > 0.0f && sinceArm < TURN_ARM_MAX_MM &&
      (lidarStale || lidarF > turnFrontCmd)) return;

  if (dcTurnManeuver) {
    Serial.println(F("# inner-side pillar before the corner -> 3-point corner"));
    goState(STATE_CORNER_MANEUVER);
  } else {
    goState(STATE_TURNING);
  }
}

// ============================================================
// STATE: CORNER MANEUVER  (3-point corner)
// ============================================================
// CLOCKWISE ONLY. Problem: a RED pillar passed on its right (the inner
// side) just before the corner leaves the car hugging the inner wall. A normal arc from
// there lands it on the inner side of the next straight, facing a pillar
// it hasn't seen yet with no room to cross (e.g. a green right after the
// corner that must be passed on its OUTER side).
// Instead:
//   SWING    drive deeper into the corner while moving toward the OUTER side
//   ARC_FWD  full lock toward the turn until MNV_ARC_FWD_DEG of the 90 is done
//            (or the wall ahead gets close)
//   STOP     settle
//   ARC_REV  reverse at the OPPOSITE full lock - this keeps rotating the car
//            the same way - until it faces the new lane (repeat once if short)
// It ends high in the corner square, facing down the next straight: the
// camera sees that straight's first pillar early and the car has room to
// go either side of it.
// ON by default, now that it only fires when it is actually the right answer:
// an inner-side pillar before the corner AND the next straight needing the
// inner side too. Gated that way it is a clear win across every layout
// (finished 16/24 vs 15, wrong-side passes 13 % vs 15 %); firing it on every
// inner pass, as the trigger alone would, cost the outer-side cases 60-300 mm
// of clearance for nothing. Settable from the Pi page - it is a parameter now,
// not a rebuild.
       bool     USE_CORNER_MANEUVER   = true;
       float    MNV_LEG_STEP_DEG      = 12.0;   // extra rotation each forward leg must add
       float    MNV_ZONE_MM           = 700.0;  // inner-side pillar within this before the trigger
       float    MNV_SWING_LAT_MM      = 150.0;  // aim this far to the OUTER side of the lane centre
       float    MNV_SWING_YAW_MAX     = 20.0;   // gentle: a big yaw here must be undone by the arc
       uint16_t MNV_DEEP_FRONT_MM     = 520;    // start the forward arc when the wall ahead is this close
       float    MNV_DEEP_CAP_CM       = 90.0;   // odometry backstop for SWING
       float    MNV_ARC_FWD_DEG       = 55.0;   // forward arc until this much of the 90 is done
       uint16_t MNV_ARC_STOP_FRONT_MM = 170;    // ... or the wall ahead is this close
       unsigned long MNV_STOP_MS      = 150;
       float    MNV_REV_CAP_CM        = 40.0;   // reverse at most this far per leg
       float    MNV_DONE_DEG          = 6.0;    // facing the new lane within this = done
       uint8_t  MNV_MAX_LEGS          = 3;      // forward/reverse legs before giving up and driving

enum MnvPhase { MNV_SWING, MNV_ARC_FWD, MNV_STOP, MNV_ARC_REV };
MnvPhase mnvPhase;
long     mnvBaseTicks;
unsigned long mnvT0;
float    mnvNewLane;
uint8_t  mnvLegs;
float    mnvLegStartDeg;   // mnvTurned() when this leg began

bool innerPillarNearCorner() {
  if (!USE_CORNER_MANEUVER) return false;
  if (!clockwiseMode) return false;                      // clockwise only (your call): red before the corner

  // Only when the corner has to come out on the INNER side. planCornerExit()
  // has already run, so cornerExitCmd says which side that is.
  //
  // The 3-point exists because an inner-side pillar before the corner pins the
  // car against the inner wall, and a plain arc from there lands on the outer
  // side of the next straight - fatal if the next pillar also needs the inner
  // side. If the next one needs the OUTER side, that plain arc is doing exactly
  // the right thing and stopping to shuffle only costs clearance: in sim the
  // greens went from 150-393 mm down to 79-92 mm when the maneuver fired on
  // them unnecessarily.
  if (cornerExitCmd > 0.0f) return false;                // CW: positive = outer
  if (haveInnerPass && laneAlongMm - lastInnerPassAt < MNV_ZONE_MM) return true;
  for (int i = 0; i < MAX_TRACKS; i++) {                 // still alongside / just ahead
    if (!tracks[i].used || tracks[i].hits < TRACK_CONFIRM) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel < -MNV_ZONE_MM || rel > 250.0f) continue;
    bool passRight = (tracks[i].color == VIS_RED) != tracks[i].flipped;
    if (passRight == clockwiseMode) return true;
  }
  return false;
}

// how far the car has rotated toward the turn since the old lane (deg, +)
float mnvTurned() {
  float d = wrapDeg(gHeading - laneHeading);
  return clockwiseMode ? -d : d;
}

void mnvFinish() {
  setMotorSpeed(0);
  laneHeading = mnvNewLane;
  zeroEncoder();
  resetLaneAlong();
  clearTracks();
  resetHeadingPid();
  Serial.print(F("# maneuver done, lane heading ")); Serial.println(laneHeading);
  finishCorner();
}

void maneuverStep() {
  float lockTurn = clockwiseMode ? SERVO_MAX_RIGHT : SERVO_MAX_LEFT;   // toward the corner
  float lockBack = clockwiseMode ? SERVO_MAX_LEFT  : SERVO_MAX_RIGHT;  // reverse = keeps rotating
  if (!entered) {
    entered = true;
    levelEnabled = false;
    plannerEnabled = false;
    clearTracks();
    Serial.print(F("# level ")); Serial.println(levelTotalDeg);
    levelTotalDeg = 0.0;
    cornerCount++;
    Serial.print(F("# TURN ")); Serial.print(cornerCount);
    Serial.print('/'); Serial.print(TARGET_CORNERS); Serial.println(F(" 3-POINT"));
    turnAmount   = clockwiseMode ? 90.0 : -90.0;
    mnvNewLane   = wrapDeg(laneHeading - turnAmount);
    mnvPhase       = MNV_SWING;
    mnvBaseTicks   = readEncoder();
    mnvLegs        = 0;
    mnvLegStartDeg = 0.0;
    resetHeadingPid();
    setMotorSpeed(DRIVE_PWM);
  }

  switch (mnvPhase) {
    case MNV_SWING: {
      // Swing toward the side the corner is meant to come out on. The old
      // code always swung OUTER, which is the side a plain arc already
      // over-delivers - the opposite of what an after-corner red needs.
      // MNV_SWING_LAT_MM is now a magnitude applied to the planned exit side.
      float side   = (cornerExitCmd >= 0.0f) ? 1.0f : -1.0f;
      if (cornerExitCmd == 0.0f) side = clockwiseMode ? 1.0f : -1.0f;
      float target = side * fabs(MNV_SWING_LAT_MM);
      float yaw = 0.0f;
      if (laneOffOk) {
        float aim = fmaxf((float)lidarF - MNV_DEEP_FRONT_MM, 150.0f);
        yaw = constrain(atan2f(target - laneOffMm, aim) / DEG_TO_RAD, -MNV_SWING_YAW_MAX, MNV_SWING_YAW_MAX);
      }
      setMotorSpeed(DRIVE_PWM);
      updateDriveSteer(wrapDeg(laneHeading + yaw), false);
      bool deep = (!lidarStale && lidarF <= MNV_DEEP_FRONT_MM) ||
                  absEnc(readEncoder() - mnvBaseTicks) >= (long)(MNV_DEEP_CAP_CM * TICKS_PER_CM);
      if (deep) {
        Serial.print(F("# 3pt arc, front=")); Serial.println(lidarF);
        mnvPhase = MNV_ARC_FWD;
      }
      break;
    }
    // Each forward leg must add MNV_LEG_STEP_DEG of NEW rotation.
    //
    // The original test was `mnvTurned() >= MNV_ARC_FWD_DEG + legs*15`, against
    // the total turned since the old lane. But reversing at the opposite lock
    // keeps rotating the car the SAME way, so by the time a reverse leg ends
    // the total already exceeds the next threshold - every forward leg after
    // the first ended on its own first pass without moving. The maneuver
    // silently collapsed into one forward arc, one reverse, then MNV_MAX_LEGS.
    // Measuring each leg from where it started fixes it.
    case MNV_ARC_FWD: {
      setServoAngle(lockTurn);
      setMotorSpeed(DRIVE_PWM);
      if (fabs(wrapDeg(mnvNewLane - gHeading)) < MNV_DONE_DEG) { mnvFinish(); return; }
      float want = (mnvLegs == 0) ? MNV_ARC_FWD_DEG : MNV_LEG_STEP_DEG;
      if (mnvTurned() - mnvLegStartDeg >= want ||
          (!lidarStale && lidarF <= MNV_ARC_STOP_FRONT_MM)) {
        setMotorSpeed(0);
        mnvT0 = millis();
        mnvPhase = MNV_STOP;
      }
      break;
    }
    case MNV_STOP:
      setMotorSpeed(0);
      setServoAngle(lockBack);                            // swing the wheels while stopped
      if (millis() - mnvT0 >= MNV_STOP_MS) {
        mnvBaseTicks   = readEncoder();
        mnvLegStartDeg = mnvTurned();
        mnvLegs++;
        Serial.print(F("# 3pt reverse, turned ")); Serial.println(mnvTurned());
        mnvPhase = MNV_ARC_REV;
      }
      break;
    case MNV_ARC_REV: {
      setServoAngle(lockBack);
      setMotorSpeed(-DRIVE_PWM);
      float left = wrapDeg(mnvNewLane - gHeading);
      bool capped = absEnc(readEncoder() - mnvBaseTicks) >= (long)(MNV_REV_CAP_CM * TICKS_PER_CM);
      if (fabs(left) < MNV_DONE_DEG || mnvTurned() > 90.0f) { mnvFinish(); return; }
      if (capped) {
        if (mnvLegs >= MNV_MAX_LEGS) { mnvFinish(); return; }      // good enough: DRIVE squares up
        setMotorSpeed(0);
        mnvBaseTicks   = readEncoder();
        mnvLegStartDeg = mnvTurned();
        mnvPhase       = MNV_ARC_FWD;                              // another forward leg
      }
      break;
    }
  }
}

// ============================================================
// STATE: TURNING  (wider eased 90 deg arc, vision off)
// ============================================================
void turningStep() {
  if (!entered) {
    entered = true;
    levelEnabled = false;
    plannerEnabled = false;
    clearTracks();
    Serial.print(F("# level ")); Serial.println(levelTotalDeg);
    levelTotalDeg = 0.0;
    cornerCount++;
    Serial.print(F("# TURN ")); Serial.print(cornerCount);
    Serial.print('/'); Serial.println(TARGET_CORNERS);
    turnAmount     = clockwiseMode ? 90.0 : -90.0;
    turnTarget     = wrapDeg(laneHeading - turnAmount);
    turnStartTicks = readEncoder();
    turnCapTicks   = (long)(TURN_CAP_CM * TICKS_PER_CM);
  }

  if (turnArcStep(turnTarget)) {
    laneHeading = wrapDeg(laneHeading - turnAmount);
    zeroEncoder();
    resetLaneAlong();
    clearTracks();                 // anything seen mid-turn was in the old lane frame
    Serial.print(F("# lane heading ")); Serial.println(laneHeading);
    finishCorner();
  }
}

// ============================================================
// STATE: FINAL STRAIGHT  (drive L - A from the end of turn 12)
// Pillars can sit in the start section, so the full steering law runs.
// ============================================================
void finalStraightStep() {
  if (!entered) {
    entered = true;
    Serial.print(F("# FINAL_STRAIGHT ")); Serial.println(finalDistanceCm);
    resetHeadingPid();
    setMotorSpeed(DRIVE_PWM);
    fsTargetTicks = (long)(finalDistanceCm * TICKS_PER_CM);
  }
  if (navValid()) {
    levelEnabled = false;
    plannerEnabled = false;
    updateNavSteer();
    setMotorSpeed(DRIVE_PWM);
    if (navDone) {
      setMotorSpeed(0);
      setServoAngle(SERVO_TRUE_STRAIGHT);
      goState(STATE_FINISHED);
    }
    return;
  }
  levelEnabled = absEnc(readEncoder()) > (long)(POST_CORNER_LOCKOUT_CM * TICKS_PER_CM);
  plannerEnabled = true;
  updateDriveSteer(laneHeading, true);
  if (absEnc(readEncoder()) >= fsTargetTicks) {
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    goState(STATE_FINISHED);
  }
}

// ============================================================
// PARAMETER TABLE  (Pi-owned: RAM only, re-pushed after every reset)
// ============================================================
// Every tunable above is a plain variable, and this table is the only thing
// that knows their names, types and legal ranges. The Pi holds the truth in
// tuning.json and pushes the whole set whenever it sees a new boot id, so the
// firmware needs no storage of its own and can never boot with a half-applied
// set.
//
// WIRE FORMAT  (all lines, both directions, are newline-terminated ASCII;
// they share the serial port with the sensor frames and the '#' log)
//
//   Pi -> STM32
//     P<id> <value>      set parameter <id>      e.g. "P12 1500"
//     N <name> <value>   set by name (for typing by hand)
//     ?P                 dump the whole table (streamed, see below)
//     ?V                 report version / boot id
//     C                  commit: parameters applied, derived values recomputed
//
//   STM32 -> Pi
//     !V <ver> <count> <boot>       on boot and on ?V
//     !P <id> <name> <type> <val> <lo> <hi> <group>     one per ?P line
//     !p <id> <val>                 acknowledge a set
//     !E <what>                     rejected (unknown id/name, out of range)
//
// The dump is STREAMED: ?P only arms a cursor, and loop() emits at most
// PARAM_DUMP_PER_LOOP lines per pass, and only while the USB CDC buffer has
// room. Printing 80 lines in one go would block the control loop for as long
// as the host took to drain them.
//
// A set is applied to the live variable immediately. Derived values
// (PASS_CLEAR_MM, LANE_LIMIT_MM, TICKS_PER_MM, TURN_MAX_STEER_*) are
// recomputed on every set, so order of arrival never matters.

const uint16_t PARAM_VERSION        = 4;     // bump when ids are added/removed
const uint8_t  PARAM_DUMP_PER_LOOP  = 2;
const uint16_t PARAM_TX_HEADROOM    = 96;    // bytes free before a dump line

enum PType : uint8_t { PT_F, PT_I, PT_U32, PT_U16, PT_U8, PT_B };

// groups, purely for how the Pi page lays the table out
enum PGroup : uint8_t {
  G_DRIVE, G_TURN, G_PLAN, G_PASS, G_LEVEL, G_MNV, G_CORNER, G_SAFE, G_LINK
};

struct ParamDesc {
  const char *name;
  void       *ptr;
  PType       type;
  float       lo, hi;
  uint8_t     group;
};

#define PF(n, g, lo, hi) { #n, (void *)&n, PT_F,   lo, hi, g }
#define PI_(n, g, lo, hi){ #n, (void *)&n, PT_I,   lo, hi, g }
#define PL(n, g, lo, hi) { #n, (void *)&n, PT_U32, lo, hi, g }
#define PS(n, g, lo, hi) { #n, (void *)&n, PT_U16, lo, hi, g }
#define PC(n, g, lo, hi) { #n, (void *)&n, PT_U8,  lo, hi, g }
#define PB(n, g)         { #n, (void *)&n, PT_B,    0,  1, g }

const ParamDesc PARAMS[] = {
  // ---- drive / calibration ----
  PF(TICKS_PER_CM,          G_DRIVE,   1,   100),
  PF(SERVO_TRUE_STRAIGHT,   G_DRIVE,  40,   120),
  PF(SERVO_MAX_LEFT,        G_DRIVE,   0,    90),
  PF(SERVO_MAX_RIGHT,       G_DRIVE,  90,   180),
  PF(IMU_YAW_SIGN,          G_DRIVE,  -1,     1),
  PI_(DRIVE_PWM,            G_DRIVE,   0,   255),
  PF(HEAD_KP,               G_DRIVE,   0,    10),
  PF(HEAD_KI,               G_DRIVE,   0,     5),
  PF(HEAD_KD,               G_DRIVE,   0,     5),
  PF(YAW_FILT_ALPHA,        G_DRIVE,   0,     1),
  PF(SERVO_SLEW,            G_DRIVE, 0.2,    30),
  PF(INTEGRAL_CLAMP,        G_DRIVE,   0,  2000),

  // ---- corner trigger + arc ----
  PS(SIDE_OPEN_MM,          G_TURN,  300,  3000),
  PC(SIDE_OPEN_FRAMES,      G_TURN,    1,    50),
  PC(SIDE_WALL_FRAMES,      G_TURN,    0,    50),
  PF(TURN_TRIGGER_MAX_YAW,  G_TURN,    5,    90),
  PF(TURN_LOCK_FRACTION,    G_TURN,  0.2,     1),
  PF(TURN_KP,               G_TURN,  0.2,    10),
  PF(TURN_MIN_STEER,        G_TURN,    0,    45),
  PF(TURN_STOP_DEG,         G_TURN,    0,    45),
  PF(TURN_CAP_CM,           G_TURN,   20,   400),
  PI_(TARGET_CORNERS,       G_TURN,    1,    48),
  PF(FINAL_STRAIGHT_CM,     G_TURN,    0,   400),
  PF(SEARCH_SAFETY_CM,      G_TURN,   50,  1000),
  PF(POST_CORNER_LOCKOUT_CM,G_TURN,    0,   200),

  // ---- corner exit shaping (see CORNER EXIT below) ----
  PF(TURN_DELAY_MM,         G_CORNER,   0,  1000),
  PF(TURN_FRONT_MM,         G_CORNER,   0,  2000),
  PF(TURN_ARM_MAX_MM,       G_CORNER,  50,  2000),
  PB(CORNER_ARC_ADAPT,      G_CORNER),
  PB(USE_SECONDARY_CORNER,  G_CORNER),
  PF(SECONDARY_ZONE_MM,     G_CORNER,   0,  3000),
  PF(TURN_FRONT_OUTER_MM,   G_CORNER,   0,  2000),
  PF(TURN_LOCK_OUTER,       G_CORNER, 0.2,     1),
  PF(TURN_FRONT_INNER_MM,   G_CORNER,   0,  2000),
  PF(TURN_LOCK_INNER,       G_CORNER, 0.2,     1),
  PF(CORNER_EXIT_MM,        G_CORNER,-400,   400),
  PF(CORNER_EXIT_BIAS_MM,   G_CORNER,   0,   400),
  PF(POST_CORNER_YAW_MAX,   G_CORNER,   5,    75),
  PF(POST_CORNER_BOOST_MM,  G_CORNER,   0,  1500),
  PF(PRE_CORNER_ZONE_MM,    G_CORNER,   0,  1500),
  PF(PRE_CORNER_SWING_MM,   G_CORNER,-400,   400),
  PB(CARRY_TRACKS,          G_CORNER),

  // ---- lane / pillar planner ----
  PF(CORRIDOR_MM,           G_PLAN,  300,  2000),
  PF(CAR_HALF_W_MM,         G_PLAN,   20,   200),
  PF(PILLAR_HALF_MM,        G_PLAN,    5,   100),
  PF(PASS_MARGIN_MM,        G_PLAN,    0,   300),
  PF(WALL_MARGIN_MM,        G_PLAN,    0,   300),
  PF(CENTRE_AIM_MM,         G_PLAN,  100,  2000),
  PF(OFF_JUMP_MM,           G_PLAN,   20,   500),
  PF(CENTRE_YAW_MAX,        G_PLAN,    2,    60),
  PS(LANE_VALID_MAX_MM,     G_PLAN,  300,  3000),
  PF(PLAN_MAX_AHEAD_MM,     G_PLAN,  300,  3000),
  PF(PILLAR_MAX_LAT_MM,     G_PLAN,  100,  1000),

  // ---- passing a pillar ----
  PF(PASS_LEAD_MM,          G_PASS,    0,   600),
  PF(PASS_AIM_MIN_MM,       G_PASS,   40,   600),
  PF(HOLD_AIM_MM,           G_PASS,   50,  1000),
  PF(PASS_YAW_MAX,          G_PASS,    5,    89),
  PF(PASS_HOLD_MM,          G_PASS,    0,   600),
  PF(UNK_COMMIT_MM,         G_PASS,   50,  1500),
  PF(TURN_RADIUS_MM,        G_PASS,  100,   800),
  PF(GIVEUP_LEAD_MM,        G_PASS,    0,   400),
  PF(AVOID_MARGIN_MM,       G_PASS,    0,   150),
  PB(ALLOW_GIVE_UP,         G_PASS),
  PF(TRACK_MATCH_MM,        G_PASS,   50,   600),
  PC(TRACK_CONFIRM,         G_PASS,    1,    20),
  PF(TRACK_FORGET_MM,       G_PASS,   50,  1500),

  // ---- wall levelling ----
  PF(LEVEL_GAIN,            G_LEVEL,   0,     1),
  PF(LEVEL_MAX_STEP,        G_LEVEL,   0,     5),
  PF(LEVEL_MAX_DIFF,        G_LEVEL,   0,    45),
  PF(LEVEL_MAX_WALLANG,     G_LEVEL,   0,    45),

  // ---- 3-point corner ----
  PB(USE_CORNER_MANEUVER,   G_MNV),
  PF(MNV_ZONE_MM,           G_MNV,     0,  2000),
  PF(MNV_SWING_LAT_MM,      G_MNV,  -400,   400),
  PF(MNV_SWING_YAW_MAX,     G_MNV,     0,    60),
  PS(MNV_DEEP_FRONT_MM,     G_MNV,   100,  1500),
  PF(MNV_DEEP_CAP_CM,       G_MNV,    10,   250),
  PF(MNV_ARC_FWD_DEG,       G_MNV,    10,    89),
  PS(MNV_ARC_STOP_FRONT_MM, G_MNV,    80,   800),
  PL(MNV_STOP_MS,           G_MNV,     0,  2000),
  PF(MNV_REV_CAP_CM,        G_MNV,     5,   120),
  PF(MNV_DONE_DEG,          G_MNV,     1,    30),
  PC(MNV_MAX_LEGS,          G_MNV,     1,    10),
  PF(MNV_LEG_STEP_DEG,      G_MNV,     0,    60),

  // ---- safety ----
  PS(WALL_PANIC_MM,         G_SAFE,    0,  1000),
  PS(WALL_CLEAR_MM,         G_SAFE,   50,  1500),
  PF(RECOVER_MAX_CM,        G_SAFE,     5,  100),
  PI_(RECOVER_MAX_TRIES,    G_SAFE,     0,   20),

  // ---- nav link (map-based path tracking on the Pi) ----
  PB(USE_NAV_TRACK,         G_LINK),
  PF(NAV_KP_HEAD,           G_LINK,    0,     5),
  PF(NAV_KP_CROSS,          G_LINK,    0,  0.05),
  PF(NAV_MAX_STEER,         G_LINK,    5,    35),
  PL(NAV_STALE_MS,          G_LINK,   50,  2000),

  // ---- link / sensors ----
  PL(LIDAR_STALE_MS,        G_LINK,   20,  2000),
  PL(LIDAR_DEAD_MS,         G_LINK,  100, 10000),
  PS(LIDAR_MAX_VALID_MM,    G_LINK,  500,  9000),
  PL(COLOR_CONFIRM_MS,      G_LINK,    0,   500),
};
const int PARAM_COUNT = (int)(sizeof(PARAMS) / sizeof(PARAMS[0]));

// Boot id: the Pi pushes the whole table whenever this changes, which is how a
// mid-session STM32 reset can never leave the car running default gains.
uint32_t bootId = 0;

void recomputeDerived() {
  PASS_CLEAR_MM  = PILLAR_HALF_MM + CAR_HALF_W_MM + PASS_MARGIN_MM;
  LANE_LIMIT_MM  = CORRIDOR_MM / 2 - CAR_HALF_W_MM - WALL_MARGIN_MM;
  if (LANE_LIMIT_MM < 0) LANE_LIMIT_MM = 0;
  TICKS_PER_MM   = TICKS_PER_CM / 10.0f;
  TURN_MAX_STEER_LEFT  = (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT)  * TURN_LOCK_FRACTION;
  TURN_MAX_STEER_RIGHT = (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT) * TURN_LOCK_FRACTION;
}

float paramGet(int i) {
  const ParamDesc &d = PARAMS[i];
  switch (d.type) {
    case PT_F:   return *(float *)d.ptr;
    case PT_I:   return (float)*(int *)d.ptr;
    case PT_U32: return (float)*(unsigned long *)d.ptr;
    case PT_U16: return (float)*(uint16_t *)d.ptr;
    case PT_U8:  return (float)*(uint8_t *)d.ptr;
    case PT_B:   return *(bool *)d.ptr ? 1.0f : 0.0f;
  }
  return 0.0f;
}

// false = out of range, nothing written
bool paramSet(int i, float v) {
  const ParamDesc &d = PARAMS[i];
  if (!(v >= d.lo && v <= d.hi)) return false;     // also rejects NaN
  switch (d.type) {
    case PT_F:   *(float *)d.ptr         = v; break;
    case PT_I:   *(int *)d.ptr           = (int)lroundf(v); break;
    case PT_U32: *(unsigned long *)d.ptr = (unsigned long)lroundf(v); break;
    case PT_U16: *(uint16_t *)d.ptr      = (uint16_t)lroundf(v); break;
    case PT_U8:  *(uint8_t *)d.ptr       = (uint8_t)lroundf(v); break;
    case PT_B:   *(bool *)d.ptr          = (v >= 0.5f); break;
  }
  recomputeDerived();
  return true;
}

int paramFind(const char *name) {
  for (int i = 0; i < PARAM_COUNT; i++)
    if (strcmp(PARAMS[i].name, name) == 0) return i;
  return -1;
}

void reportVersion() {
  Serial.print(F("!V ")); Serial.print(PARAM_VERSION);
  Serial.print(' ');      Serial.print(PARAM_COUNT);
  Serial.print(' ');      Serial.println(bootId);
}

// ---- streamed dump ----
int paramDumpIdx = -1;                 // -1 = idle

void paramDumpStart() { paramDumpIdx = 0; }

void paramDumpStep() {
  if (paramDumpIdx < 0) return;
  for (uint8_t k = 0; k < PARAM_DUMP_PER_LOOP; k++) {
    if (paramDumpIdx >= PARAM_COUNT) { paramDumpIdx = -1; reportVersion(); return; }
    if (Serial.availableForWrite() < PARAM_TX_HEADROOM) return;   // host not draining
    const ParamDesc &d = PARAMS[paramDumpIdx];
    Serial.print(F("!P ")); Serial.print(paramDumpIdx);
    Serial.print(' ');      Serial.print(d.name);
    Serial.print(' ');      Serial.print((int)d.type);
    Serial.print(' ');      Serial.print(paramGet(paramDumpIdx), 4);
    Serial.print(' ');      Serial.print(d.lo, 4);
    Serial.print(' ');      Serial.print(d.hi, 4);
    Serial.print(' ');      Serial.println((int)d.group);
    paramDumpIdx++;
  }
}

// ---- one tuning line, already NUL-terminated in lidarBuf ----
// Returns true if the line was a tuning command (so the frame parser skips it).
bool parseTuning(char *s) {
  if (s[0] == '?' && s[1] == 'P' && s[2] == '\0') { paramDumpStart(); return true; }
  if (s[0] == '?' && s[1] == 'V' && s[2] == '\0') { reportVersion();  return true; }
  if (s[0] == 'C' && s[1] == '\0') { recomputeDerived(); Serial.println(F("!C")); return true; }

  if (s[0] == 'P' && s[1] >= '0' && s[1] <= '9') {
    char *end;
    long id = strtol(s + 1, &end, 10);
    if (end == s + 1) return false;
    float v = strtof(end, &end);
    if (id < 0 || id >= PARAM_COUNT) { Serial.println(F("!E id")); return true; }
    if (!paramSet((int)id, v))       { Serial.println(F("!E range")); return true; }
    Serial.print(F("!p ")); Serial.print((int)id);
    Serial.print(' ');      Serial.println(paramGet((int)id), 4);
    return true;
  }

  if (s[0] == 'N' && s[1] == ' ') {
    char *name = s + 2;
    char *sp   = strchr(name, ' ');
    if (!sp) { Serial.println(F("!E syntax")); return true; }
    *sp = '\0';
    int id = paramFind(name);
    *sp = ' ';
    if (id < 0)                        { Serial.println(F("!E name"));  return true; }
    if (!paramSet(id, strtof(sp, 0)))  { Serial.println(F("!E range")); return true; }
    Serial.print(F("!p ")); Serial.print(id);
    Serial.print(' ');      Serial.println(paramGet(id), 4);
    return true;
  }
  return false;
}

// ============================================================
// MAIN
// ============================================================
void setup() {
  Serial.begin(115200);
  bootId = (uint32_t)millis() ^ 0x5A5A0000u;   // any value the Pi has not seen
  recomputeDerived();
  initHardware();
  reportVersion();
}

// ---- telemetry to the Pi, 50 Hz ----
// The Pi dead-reckons its pose between LiDAR revolutions from exactly these
// two numbers. Without them it would have to integrate its own motion model
// and would drift badly during the 100 ms between scans.
unsigned long telemLastMs = 0;
const unsigned long TELEM_PERIOD_MS = 20;

void sendTelemetry() {
  if (millis() - telemLastMs < TELEM_PERIOD_MS) return;
  telemLastMs = millis();
  if (Serial.availableForWrite() < 32) return;      // never block the loop
  Serial.print(F("T "));
  Serial.print((int)lroundf(gHeading * 10.0f));     // deci-degrees
  Serial.print(' ');
  Serial.print(readEncoder());
  Serial.print(' ');
  Serial.println((int)currentState);
}

void loop() {
  serviceSensors();
  paramDumpStep();        // streamed, at most PARAM_DUMP_PER_LOOP lines
  sendTelemetry();

  // ---- startup gate: nothing runs until the Pi's first frame ----
  if (!fsmStarted) {
    if (lidarFrames == 0) {
      digitalWrite(LED1_PIN, ((millis() / 500) & 1) ? HIGH : LOW);
      return;
    }
    fsmStarted = true;
    Serial.println(F("# first Pi frame received - FSM starting"));
  }

  // START only means something while armed or finished. Anything else -
  // a press mid-run, or the repeat copies of the press that started this
  // run - is discarded so it can't auto-restart the car when it finishes.
  if (currentState != STATE_WAIT_START && currentState != STATE_FINISHED)
    startRequested = false;

  // STOP from the page: works in every state.
  if (stopRequested) {
    stopRequested = false;
    if (currentState != STATE_FINISHED) {
      Serial.println(F("# STOP from Pi"));
      goState(STATE_FINISHED);
    }
  }

  if (colorMuted && currentState != STATE_RECOVER && readEncoder() >= colorMuteFrom) {
    colorMuted = false;
    Serial.println(F("# colour re-enabled"));
  }

  // Wall panic overlay. Skipped while the camera sees a pillar: the short
  // front reading is the pillar, and the pillar PD is steering round it.
  if (!lidarStale &&
      !pillarSeen() &&
      currentState != STATE_RECOVER &&
      currentState != STATE_WAIT_START &&
      currentState != STATE_CORNER_MANEUVER &&     // it watches the wall ahead itself
      currentState != STATE_FINISHED &&
      recoverTries < RECOVER_MAX_TRIES &&
      lidarF <= WALL_PANIC_MM) {
    enterRecovery();
  }

  if (currentState != STATE_WAIT_START && currentState != STATE_FINISHED) {
    digitalWrite(LED1_PIN, lidarStale ? (((millis() / 100) & 1) ? HIGH : LOW) : HIGH);
  }

  switch (currentState) {
    case STATE_WAIT_START:      waitStartStep();     break;
    case STATE_DRIVE_TO_CORNER: driveStep();         break;
    case STATE_TURNING:         turningStep();       break;
    case STATE_FINAL_STRAIGHT:  finalStraightStep(); break;
    case STATE_RECOVER:         recoverStep();       break;
    case STATE_CORNER_MANEUVER: maneuverStep();      break;

    case STATE_FINISHED:
      if (!entered) {
        entered = true;
        Serial.println(F("# FINISHED"));
        levelEnabled = false;
        plannerEnabled = false;
        startRequested = false;      // only a NEW press restarts
        setMotorSpeed(0);
        setServoAngle(SERVO_TRUE_STRAIGHT);
        digitalWrite(LED1_PIN, HIGH);
        digitalWrite(LED2_PIN, HIGH);
        digitalWrite(LED3_PIN, HIGH);
      }
      // START again from the page re-arms and runs a fresh 3 laps
      // (put the car back in the start section first).
      if (startRequested) {
        goState(STATE_WAIT_START);   // WAIT_START clears the flag on entry,
        waitStartStep();             // so arm first ...
        startRequested = true;       // ... then honour this press
      }
      break;
  }
}
