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
// OBSTACLE ROUND - NON-BLOCKING FIRMWARE  (v7, param table v6)
//
// The STM32 owns all driving. The Pi (obstacleRound.py) is a sensor pipe:
// LiDAR + camera in, one CSV line out at up to 50 Hz, plus START/STOP and
// tuning lines on the same port. Nothing here blocks after setup().
//
// SERIAL FRAME (Pi -> STM32), 18 comma-separated integers:
//   0-2   left,front,right   mm at 90/0/270 deg, 65535 = no return
//   3     rev                LiDAR revolution counter (debounce on this,
//                            not on frames - a bearing changes once per rev)
//   4     color              first sign: 1 red, 0 green, 2 none
//   5-7   err,area,vseq      camera debug, not used here (field 5 is the
//                            slot reserved for a rear ToF later)
//   8-9   coneL,coneR        perpendicular mm to each wall, 65535 = no fit
//   10    wallAng            car yaw to the walls, deci-deg, + = left, 32767 = none
//   11-12 pX,pY              first sign centre, mm, x fwd / y left, 32767 = none
//   13-14 uX,uY              nearest uncoloured LiDAR object - parsed past, NOT used
//                            (steering toward it pulled the car into other lanes)
//   15    sColor             second sign colour (2 = none)
//   16-17 sX,sY              second sign centre
// A shorter line still parses: 4 fields = open-round feed (no camera).
//
// COMMANDS  S = START (armed or finished only), X = STOP (any state).
// BUTTON    PB12 to GND (INPUT_PULLUP). One press does what the page does:
//           armed or finished -> START, anything else -> STOP. Ignored until
//           the Pi's first frame; a START with the LiDAR stale is refused.
// TUNING    N <name> <value>, ?P (dump table), ?V (version / boot id).
//
// PER CORNER
//   1. DRIVE: IMU heading + lane planner (centring, sign passing).
//   2. CORNER TRIGGER = LiDAR only: the inner-side beam reads more than
//      SIDE_OPEN_MM (1500) on SIDE_OPEN_REVS new revolutions, after that side
//      has shown a wall on SIDE_WALL_REVS revolutions this straight, with the
//      car within TURN_TRIGGER_MAX_YAW of the lane. Until the direction is
//      known BOTH sides are watched: the outer wall never ends, so the first
//      side to open is the inner side - right = clockwise, left = anticlockwise.
//      The floor colour sensor only drives LED2/LED3 now.
//   3. TURNING, shaped by the LAST SIGN of the straight:
//        passed on the OUTER side (CW green / CCW red):
//          drive straight to front <= TURN_OUTER_FRONT_MM (400), then one
//          forward arc eased onto the new lane heading
//        passed on the INNER side (CW red / CCW green), or no sign seen:
//          drive straight to front <= TURN_INNER_FRONT_MM (200), stop, then a
//          REVERSE arc at the opposite lock (which keeps rotating the car the
//          same way) eased onto the new lane heading by the IMU
//      then stop TURN_VIEW_MS so the camera sees the next straight, and drive.
//      Optional TURN_NEXT_OVERRIDE: a forward-arc corner whose NEXT straight
//      starts with a sign needing the inner side arcs early instead.
//   4. A sign whose correct side cannot be reached any more makes the car
//      reverse in a straight line (BACKOFF) until the pass is reachable,
//      up to BACKOFF_MAX_MM, then it commits to the correct side.
//
// BENCH-VERIFIED
//   motor    PA2 forward, PA3 reverse
//   encoder  TIM5 PA0/PA1, negated so forward counts up
//   IMU      BNO08x SPI1 ~100 Hz, clockwise = negative yaw
//   colour   TCS34725 on TCA9548A channel 4 (LEDs only)
//            white pR 47 / pB 19, orange pR 69 / pB 11, blue pR 36 / pB 27
//   servo    500-2500 us, straight 76.5, left stop 20, right stop 140
//            (below straight steers LEFT)
//   button   PB12 to GND, internal pull-up (v7). LED1 moved to the Black
//            Pill's on-board LED, PC13 (active LOW), because PB12 is the button.
// ============================================================

enum BlockColor { COLOR_NONE, COLOR_ORANGE, COLOR_BLUE };

enum RobotState {
  STATE_WAIT_START,
  STATE_DRIVE_TO_CORNER,
  STATE_TURNING,
  STATE_FINAL_STRAIGHT,
  STATE_RECOVER,
  STATE_FINISHED,
  STATE_BACKOFF              // overlay: reverse until a sign's correct side is reachable
};

// A sign as the Pi reports it: colour + centre in the car frame (mm from the
// LiDAR, x forward, y left). Declared up here, before any function, because the
// Arduino IDE inserts its auto-generated prototypes above the first function -
// a type used in a parameter list must already exist at that point.
struct Sighting { bool valid; int color; float x, y; };

// ============================================================
// HARDWARE PINS & OBJECTS
// ============================================================
const int MOT_RPWM_PIN = PA2;     // forward  (TIM2_CH3; TIM5 is the encoder)
const int MOT_LPWM_PIN = PA3;     // reverse  (TIM2_CH4)
const int SERVO_PIN    = PA8;

const int IMU_CS_PIN  = PA4;
const int IMU_INT_PIN = PB0;
const int IMU_RST_PIN = PB1;

const int LED1_PIN = PC13;        // ON-BOARD LED, active LOW (PB12 is the button now).
                                  // slow blink (500 ms) = waiting for Pi; medium blink (250 ms) =
                                  // armed, waiting for START; solid = running; fast blink = lidar stale
const int BTN_PIN  = PB12;        // start / stop button to GND, internal pull-up: pressed = LOW
const int LED2_PIN = PB13;        // lit while ORANGE is under the sensor
const int LED3_PIN = PB14;        // lit while BLUE is under the sensor

// LED1 is the on-board PC13 LED, which lights when the pin is LOW.
inline void led1(bool on) { digitalWrite(LED1_PIN, on ? LOW : HIGH); }

#define I2C_SCL     PB6
#define I2C_SDA     PB7
#define TCA_RST_PIN PB8
#define TCA_ADDR    0x70
#define TCS_CH      4             // TCS34725

// ---- calibration ----
       float TICKS_PER_CM        = 14.853;

const int   SERVO_MIN_PULSE_US  = 500;
const int   SERVO_MAX_PULSE_US  = 2500;
       float SERVO_TRUE_STRAIGHT = 79.5;
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
// PI LINK
// ============================================================
       unsigned long LIDAR_STALE_MS      = 200;
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
uint32_t      lidarFrames = 0;
bool          lidarNewFrame = false;   // true only on the loop a frame was parsed

const int VIS_RED   = 1;
const int VIS_GREEN = 0;
const int VIS_NONE  = 2;

// 45 deg cone wall fits
const long  CONE_NONE     = 65535;
const long  ANG_NONE      = 32767;
uint16_t coneL = LIDAR_FAR, coneR = LIDAR_FAR;   // perpendicular mm, FAR = no fit
bool     wallAngValid = false;
float    wallAngDeg   = 0.0;                     // + = car pointing left of the walls
uint32_t lidarRev     = 0;
bool     lidarNewRev  = false;                   // true only on the loop rev changed

// Two signs per frame (struct Sighting is declared at the top of the file).
const long PXY_NONE = 32767;
Sighting sign1 = { false, VIS_NONE, 0, 0 };      // largest accepted blob
Sighting sign2 = { false, VIS_NONE, 0, 0 };      // next one back (often the next straight's)
int visColor = VIS_NONE;                         // sign1's colour, even when not located

char    lidarBuf[128];               // an 18-field frame is 83 chars typical, 104 worst
uint8_t lidarLen = 0;

bool startRequested = false;
bool stopRequested  = false;
bool stopByButton   = false;         // only for the log line

bool parseTuning(char *s);           // PARAMETER TABLE, near the bottom

static bool xyOk(long x, long y) { return x != PXY_NONE && y != PXY_NONE; }

void parseLine() {
  if (lidarBuf[0] == 'S' && lidarBuf[1] == '\0') { startRequested = true; return; }
  if (lidarBuf[0] == 'X' && lidarBuf[1] == '\0') { stopRequested  = true; return; }
  if (parseTuning(lidarBuf)) return;   // a frame always starts with a digit or '-'

  const uint8_t NF = 18;
  long f[NF];
  uint8_t n = 0;
  char *p = lidarBuf;
  while (n < NF) {
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
  if (n >= 4 && (uint32_t)f[3] != lidarRev) { lidarRev = (uint32_t)f[3]; lidarNewRev = true; }

  visColor = (n >= 5 && (f[4] == VIS_RED || f[4] == VIS_GREEN)) ? (int)f[4] : VIS_NONE;
  // fields 5-7 (err, area, vseq) are camera debug; skipped

  if (n >= 11) {
    coneL = (f[8] == CONE_NONE) ? LIDAR_FAR : lidarSanitize(f[8]);
    coneR = (f[9] == CONE_NONE) ? LIDAR_FAR : lidarSanitize(f[9]);
    wallAngValid = (f[10] != ANG_NONE);
    wallAngDeg   = wallAngValid ? f[10] / 10.0f : 0.0f;
  } else {                           // open-round feed: fall back to the 90/270 beams
    coneL = lidarL; coneR = lidarR; wallAngValid = false;
  }

  sign1.valid = (visColor != VIS_NONE && n >= 13 && xyOk(f[11], f[12]));
  sign1.color = visColor;
  if (sign1.valid) { sign1.x = (float)f[11]; sign1.y = (float)f[12]; }

  // fields 13-14 (uX, uY: uncoloured LiDAR object) are ignored - see the header

  sign2.valid = (n >= 18 && (f[15] == VIS_RED || f[15] == VIS_GREEN) && xyOk(f[16], f[17]));
  if (sign2.valid) { sign2.color = (int)f[15]; sign2.x = (float)f[16]; sign2.y = (float)f[17]; }

  lidarLastMs   = millis();
  lidarStale    = false;
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
  if (millis() - lidarLastMs > LIDAR_STALE_MS) lidarStale = true;
}

// ---- corner trigger (LiDAR only) ----
       uint16_t SIDE_OPEN_MM      = 1500;   // inner side above this = inner wall gone
       uint8_t  SIDE_OPEN_REVS    = 2;      // consecutive NEW revolutions (not frames)
       uint8_t  SIDE_WALL_REVS    = 2;      // that side must first show a wall this many revs:
                                            //   after a turn the car starts in the corner square,
                                            //   where the inner side looks "open" across the old lane
       float    TURN_TRIGGER_MAX_YAW = 20.0; // only while the car is this close to the lane
                                            //   direction: mid-swerve the "side" beam isn't sideways
       float    CORNER_LOCKOUT_MM = 700.0;  // no corner trigger this soon after a corner: the next
                                            //   corner mouth is a full 1000 mm straight away
       bool     SIDE_OPEN_NEED_CONE = true; // open also needs that side's 45 deg wall fit to be gone:
                                            //   a single beam grazing past the island corner while the
                                            //   car is still yawed out of a turn reads 2800 mm (sim)

// ---- wall recovery ----
       uint16_t WALL_PANIC_MM     = 200;
       uint16_t WALL_CLEAR_MM     = 350;
       float    RECOVER_MAX_CM    = 30.0;
       int      RECOVER_MAX_TRIES = 3;
       float    PANIC_PILLAR_MM   = 350.0;  // a located sign this close ahead IS the short
                                            //   front reading: the planner owns it, no panic

// ============================================================
// RUN CONSTANTS
// ============================================================
       int   TARGET_CORNERS         = 12;
       float FINAL_STRAIGHT_CM      = 100;
       float POST_CORNER_LOCKOUT_CM = 50.0;   // no levelling this soon after a corner

// ---- corner exit ----
// The colour of the next straight's first sign, seen across the corner, sets
// the lane offset the planner aims for over the first POST_CORNER_BOOST_MM:
// red -> right of centre, green -> left, CORNER_EXIT_BIAS_MM either way.
// Nothing seen -> CORNER_EXIT_MM (+ = outer side of the lap). The first stretch
// also gets POST_CORNER_YAW_MAX of steering authority instead of CENTRE_YAW_MAX.
      float CORNER_EXIT_MM       = 0.0;
      float CORNER_EXIT_BIAS_MM  = 200.0;
      float POST_CORNER_YAW_MAX  = 45.0;
      float POST_CORNER_BOOST_MM = 600.0;

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

// Driving direction: unknown until the first colour line of the run.
bool dirLocked     = false;
bool clockwiseMode = true;       // valid once dirLocked
int  cornerCount   = 0;

float laneHeading  = 0.0;        // IMU heading of the current straight

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

// the inner-side beam: right when turning clockwise, left anticlockwise
uint16_t turnSideMm() { return clockwiseMode ? lidarR : lidarL; }

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

// Floor colour: diagnostic only (LED2 orange, LED3 blue) - no decision uses it.
// Chromaticity thresholds sit midway between the measured mat values:
//   white pR 47 / pB 19, orange pR 69 / pB 11, blue pR 36 / pB 27.
       float ORANGE_PR_MIN = 58.0;
       float ORANGE_PB_MAX = 15.0;
       float BLUE_PB_MIN   = 23.0;
       float BLUE_PR_MAX   = 41.0;
       float COLOR_SUM_MIN = 100.0;   // R+G+B below this = too dark to judge

BlockColor classifyColor() {
  uint16_t r, g, b, c;
  readColor(r, g, b, c);
  float total = (float)r + (float)g + (float)b;
  if (total < COLOR_SUM_MIN) return COLOR_NONE;
  float pR = (r / total) * 100.0f;
  float pB = (b / total) * 100.0f;
  if (pR > ORANGE_PR_MIN && pB < ORANGE_PB_MAX) return COLOR_ORANGE;
  if (pB > BLUE_PB_MIN   && pR < BLUE_PR_MAX)   return COLOR_BLUE;
  return COLOR_NONE;
}

// ============================================================
// LANE POSITION + SIGN PASS PLANNER
// ============================================================
// Everything is a LATERAL POSITION in the lane (mm, + = left of centre):
//   car    laneOffMm   from the two cone wall fits (perpendicular mm)
//   sign   track.lat   car offset + the sign's car-frame position (Pi:
//                      camera bearing + LiDAR range), rotated by the car's
//                      yaw to the lane
// Rule 9.19, written ONCE in passRight(): red is passed on its right, green
// on its left. Each confirmed sign in play adds a one-sided bound:
//   pass right -> car lat <= sign lat - PASS_CLEAR
//   pass left  -> car lat >= sign lat + PASS_CLEAR
// The target is the middle of the free gap between the nearest sign's face
// and the wall on its passing side (GAP_CENTRE_W blends toward the old
// "hug the sign" target), clamped to every bound. The car steers there with
// a yaw command (heading = laneHeading + yaw) aimed at a point PASS_LEAD_MM
// before the sign, which the IMU heading PID tracks.
       float    CORRIDOR_MM       = 1000.0;
       float    CAR_HALF_W_MM     = 57.0;
       float    PILLAR_HALF_MM    = 25.0;
       float    PASS_MARGIN_MM    = 80.0;    // air gap car side <-> sign face
      float    PASS_CLEAR_MM     = 162.0;   // derived, see recomputeDerived()
       float    WALL_MARGIN_MM    = 45.0;
      float    LANE_LIMIT_MM     = 398.0;   // derived, see recomputeDerived()
       float    GAP_CENTRE_W      = 1.0;     // 1 = middle of the sign-to-wall gap, 0 = hug the sign
       float    CENTRE_AIM_MM     = 600.0;   // centring: aim at the target this far ahead
       float    PASS_LEAD_MM      = 160.0;   // be at the pass position this far BEFORE the sign
       float    PASS_AIM_MIN_MM   = 250.0;   // shortest aim distance (sharpest swerve); v7: was 120
       float    HOLD_AIM_MM       = 400.0;   // alongside a sign: gentle hold; v7: was 250
       float    OFF_JUMP_MM       = 120.0;   // lane offset can't move this much in one revolution
       uint8_t  OFF_JUMP_REVS     = 3;       // ... unless the jump persists this many revolutions
       float    CORRIDOR_SUM_TOL_MM = 150.0; // coneL + coneR must be CORRIDOR_MM within this
       float    CENTRE_YAW_MAX    = 20.0;    // deg, plain centring
       float    PASS_YAW_MAX      = 40.0;    // deg, while a sign is in play; v7: was 75 - with
                                             //   HEAD_KP 2 anything over ~28 deg is full lock anyway
       float    TURN_RADIUS_MM    = 270.0;   // full-lock radius (worse side) for the reach estimate
       float    REACH_LEAD_MM     = 60.0;    // reach is judged to this far before the sign
       float    REACH_NEED_MIN_MM = 60.0;    // a lateral move smaller than this is always reachable
       float    REACH_MARGIN_MM   = 30.0;    // unreachable = need > reach + this
       float    PLAN_MAX_AHEAD_MM = 1600.0;  // ignore sightings farther ahead
       float    SIGHT_MIN_ALONG_MM = -50.0;  // ... or farther behind
       float    SIGHT_MAX_YAW     = 30.0;    // deg: car turned further than this off the lane ->
                                             //   ignore NEW sightings (the camera is looking across
                                             //   the corner / island into other lanes); tracks
                                             //   already confirmed keep steering the car
       float    PILLAR_MAX_LAT_MM = 330.0;   // seats are at +/-100; beyond this = another straight's sign
                                             //   v7: was 420, let other lanes' signs in after a corner
       float    CROSS_MAX_LAT_MM  = 1300.0;  // beyond this it is not even the next straight
       float    PASS_HOLD_MM      = 250.0;   // keep a sign's side until it is this far behind the car
       float    TRACK_MATCH_MM    = 200.0;   // same sign if within this (along and lateral)
       float    TRACK_GAIN        = 0.5;     // how far a new sighting moves a track estimate
       uint8_t  TRACK_CONFIRM     = 2;       // sightings before a sign steers the car
       float    TRACK_FORGET_MM   = 300.0;   // unconfirmed and not seen for this far -> dropped
       uint16_t LANE_VALID_MAX_MM = 1100;    // cone farther than this = no wall on that side
      float    TICKS_PER_MM      = 1.4853;  // derived, see recomputeDerived()

// ---- reverse and re-plan (BACKOFF) ----
// When the correct side of the nearest sign can no longer be reached, the car
// reverses in a straight line: every mm reversed is a mm of extra run-up. It
// drives again as soon as the pass is reachable with BACKOFF_MARGIN_MM to
// spare. After BACKOFF_MAX_MM for one sign it commits to the correct side at
// full authority - touching a sign is allowed while it stays in its circle
// (rule 9.20); a wrong-side pass ends the round (9.24.5). Reversing is BLIND
// (nothing measures behind the car yet), so keep BACKOFF_MAX_MM short.
       bool     BACKOFF_ENABLE    = true;
       int      BACKOFF_PWM       = 50;
       float    BACKOFF_MAX_MM    = 250.0;   // per sign
       float    BACKOFF_MARGIN_MM = 60.0;    // reach must beat need by this to drive again
       float    BACKOFF_MIN_MM    = 80.0;    // reverse at least this far each time (no 20 mm dithering)
       float    BACKOFF_BEHIND_MM = 200.0;   // may reverse this far behind where the straight
                                             //   began: after a forward-arc corner the car exits on the
                                             //   outer side with ~300 mm of corner square behind it, and
                                             //   a sign right at the exit needs that run-up (blind!)

struct PillarTrack { bool used; int color; float lat; float along; uint8_t hits; float lastSeen; float backedMm; };
const int   MAX_TRACKS = 4;
PillarTrack tracks[MAX_TRACKS];

float laneOffMm   = 0.0;     // + = car left of lane centre
bool  laneOffOk   = false;
float laneAlongMm = 0.0;     // distance along the lane since the last corner
long  laneAlongEnc = 0;
float latYawCmd   = 0.0;     // + = yaw left of laneHeading
float latTarget   = 0.0;
bool  passActive  = false;
bool  plannerEnabled = false;  // DRIVE / FINAL / BACKOFF - sightings mid-turn are in the wrong lane frame

int   backoffWanted = -1;      // track index whose correct side is unreachable, -1 = none
bool  passFeasible  = true;    // nearest sign reachable with BACKOFF_MARGIN_MM to spare
float reachNeedMm   = 0.0, reachHaveMm = 0.0;   // last reach check, for the log

// Rule 9.19 - the ONLY place the colour -> side rule is written.
bool passRight(int color) { return color == VIS_RED; }

void clearTracks() { for (int i = 0; i < MAX_TRACKS; i++) tracks[i].used = false; passActive = false; backoffWanted = -1; }

// ---- CROSS-CORNER SIGHTING ----
// The camera sees the next straight's first sign across the corner. Its
// position in this lane frame is meaningless, but its COLOUR decides which
// side to leave the corner on (red -> right of centre, green -> left).
int   nextStraightColor = VIS_NONE;
float cornerExitCmd     = 0.0;           // lane offset for the first POST_CORNER_BOOST_MM

// colour of the last confirmed sign already passed on this straight
int   lastPassedColor   = VIS_NONE;

void resetLaneAlong() { laneAlongMm = 0.0; laneAlongEnc = readEncoder();
                        nextStraightColor = VIS_NONE; lastPassedColor = VIS_NONE; }

bool laneOffsetRaw(float &off) {
  bool l = coneL <= LANE_VALID_MAX_MM, r = coneR <= LANE_VALID_MAX_MM;
  if (l && r) {
    float both = 0.5f * ((float)coneR - (float)coneL);
    float sum  = (float)coneL + (float)coneR;
    if (fabs(sum - CORRIDOR_MM) < CORRIDOR_SUM_TOL_MM || !laneOffOk) { off = both; return true; }
    // walls don't add up to the corridor: one cone is fitted to something
    // else (a sign beside the car). Keep the side that agrees with before.
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
// Accept it only if it persists (then it's real, e.g. after a corner).
bool laneOffset(float &off) {
  float o;
  if (!laneOffsetRaw(o)) { offJumps = 0; return false; }
  if (laneOffOk && fabs(o - off) > OFF_JUMP_MM && offJumps < OFF_JUMP_REVS) {
    if (lidarNewRev) offJumps++;
    return true;                                           // keep the previous value
  }
  offJumps = 0;
  off = o;
  return true;
}

// Wall positions in the lane frame. The lane frame is itself built from the
// two wall fits (offset = (coneR - coneL) / 2), so the walls sit at
// +/- CORRIDOR_MM / 2 by construction. Using the raw cone distance instead
// breaks exactly where it matters: just after a corner one side looks across
// the corner square and "fits" a wall 1000+ mm away, which put the gap centre
// outside the lane (sim: need 546 mm -> spurious BACKOFF).
float wallLatRight() { return -CORRIDOR_MM / 2; }
float wallLatLeft()  { return  CORRIDOR_MM / 2; }

// One sign from the Pi -> lane frame -> track update, or the next straight's colour.
void addSighting(const Sighting &s) {
  if (!s.valid || lidarStale || !laneOffOk) return;
  // Swerving hard (or still yawed out of a turn) the camera looks sideways, across
  // the corner or the island, at other straights' signs. Nothing seen now can be
  // trusted to be in THIS lane - not even as the next straight's colour - so every
  // sighting is dropped until the car is back near the lane direction. Tracks that
  // are already confirmed are untouched and keep steering the pass.
  if (fabs(wrapDeg(gHeading - laneHeading)) > SIGHT_MAX_YAW) return;
  float yaw = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
  float along  = s.x * cosf(yaw) - s.y * sinf(yaw);
  float lat    = laneOffMm + s.x * sinf(yaw) + s.y * cosf(yaw);
  if (along < SIGHT_MIN_ALONG_MM || along > PLAN_MAX_AHEAD_MM) return;
  if (fabs(lat) > PILLAR_MAX_LAT_MM) {
    // another straight's sign, seen across the corner: keep only its colour,
    // and only if it lies on the side the car is about to turn toward
    bool towardTurn = dirLocked && (clockwiseMode ? (lat < 0.0f) : (lat > 0.0f));
    if (towardTurn && along > 0.0f && fabs(lat) < CROSS_MAX_LAT_MM) nextStraightColor = s.color;
    return;
  }
  float at = laneAlongMm + along;

  int slot = -1, freeSlot = -1, oldest = 0;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used) { if (freeSlot < 0) freeSlot = i; continue; }
    if (tracks[i].color == s.color &&
        fabs(tracks[i].along - at) < TRACK_MATCH_MM && fabs(tracks[i].lat - lat) < TRACK_MATCH_MM) { slot = i; break; }
    if (tracks[i].along < tracks[oldest].along) oldest = i;
  }
  if (slot >= 0) {                                         // refine
    tracks[slot].lat   += TRACK_GAIN * (lat - tracks[slot].lat);
    tracks[slot].along += TRACK_GAIN * (at  - tracks[slot].along);
    tracks[slot].lastSeen = laneAlongMm;
    if (tracks[slot].hits < 255) tracks[slot].hits++;
    if (tracks[slot].hits == TRACK_CONFIRM) {
      Serial.print(F("# pillar ")); Serial.print(s.color == VIS_RED ? F("RED") : F("GREEN"));
      Serial.print(F(" lat=")); Serial.print((int)tracks[slot].lat);
      Serial.print(F(" at=")); Serial.println((int)tracks[slot].along);
    }
    return;
  }
  slot = (freeSlot >= 0) ? freeSlot : oldest;
  tracks[slot].used = true; tracks[slot].color = s.color;
  tracks[slot].lat = lat;   tracks[slot].along = at;
  tracks[slot].hits = 1;    tracks[slot].lastSeen = laneAlongMm; tracks[slot].backedMm = 0.0f;
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

bool trackConfirmed(int i) { return tracks[i].used && tracks[i].hits >= TRACK_CONFIRM; }

// The LAST sign of this straight, for the corner plan: a confirmed sign still
// being tracked (alongside, just passed, or ahead) is later in the straight
// than any sign already dropped as passed, so the farthest-along one wins;
// otherwise the last one passed. VIS_NONE = no sign on this straight.
int lastSignOfStraight() {
  int best = -1;
  for (int i = 0; i < MAX_TRACKS; i++)
    if (trackConfirmed(i) && (best < 0 || tracks[i].along > tracks[best].along)) best = i;
  return (best >= 0) ? tracks[best].color : lastPassedColor;
}

void updatePlanner() {
  // lane distance: encoder projected onto the lane direction (negative when reversing)
  long enc = readEncoder();
  float dmm = (enc - laneAlongEnc) / TICKS_PER_MM;
  laneAlongEnc = enc;
  laneAlongMm += dmm * cosf(wrapDeg(gHeading - laneHeading) * DEG_TO_RAD);

  if (!lidarNewFrame) return;
  laneOffOk = laneOffset(laneOffMm);
  if (!plannerEnabled) { latYawCmd = 0.0f; passActive = false; backoffWanted = -1; passFeasible = true; return; }
  addSighting(sign1);
  addSighting(sign2);

  // tracks in play: everything still within PASS_HOLD behind, up to and
  // including the NEAREST sign ahead (farther ones wait their turn)
  float nearestAhead = 1e9; int nearestIdx = -1;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel < -PASS_HOLD_MM) {                                               // passed
      if (tracks[i].hits >= TRACK_CONFIRM) lastPassedColor = tracks[i].color;
      tracks[i].used = false; continue;
    }
    if (tracks[i].hits < TRACK_CONFIRM) {                                    // not trusted yet
      if (laneAlongMm - tracks[i].lastSeen > TRACK_FORGET_MM) tracks[i].used = false;
      continue;
    }
    if (rel > 0 && rel < nearestAhead) { nearestAhead = rel; nearestIdx = i; }
  }

  float lo = -LANE_LIMIT_MM, hi = LANE_LIMIT_MM;
  float urgentRel = 1e9, urgentBound = 0; int urgentIdx = -1; bool any = false;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!trackConfirmed(i)) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel > nearestAhead + 1.0f) continue;
    any = true;
    float bound;
    if (passRight(tracks[i].color)) { bound = tracks[i].lat - PASS_CLEAR_MM; if (bound < hi) hi = bound; }
    else                            { bound = tracks[i].lat + PASS_CLEAR_MM; if (bound > lo) lo = bound; }
    if (rel < urgentRel) { urgentRel = rel; urgentBound = bound; urgentIdx = i; }
  }
  passActive = any;

  // Default target. No sign in play: lane centre, or the planned corner-exit
  // offset for the first stretch after a corner. A sign in play: the middle of
  // the gap between the most urgent sign's face and the wall on its passing side.
  float target = 0.0f;
  if (!any) {
    if (cornerCount > 0 && laneAlongMm < POST_CORNER_BOOST_MM) target = cornerExitCmd;
  } else {
    const PillarTrack &u = tracks[urgentIdx];
    float mid = passRight(u.color) ? 0.5f * ((u.lat - PILLAR_HALF_MM) + wallLatRight())
                                   : 0.5f * ((u.lat + PILLAR_HALF_MM) + wallLatLeft());
    target = GAP_CENTRE_W * mid;
  }
  if (lo > hi) target = urgentBound;                     // conflict: most urgent sign wins
  else         target = constrain(target, lo, hi);

  float aimMm = CENTRE_AIM_MM;
  if (!any) passFeasible = true;     // with a sign in play it keeps its last computed value
  if (any) {
    // aim at the pass point PASS_LEAD before the nearest sign: the swerve
    // sharpens as it nears and arrives in time. Alongside / past: gentle hold.
    if (nearestAhead < 1e8 && nearestAhead > PASS_LEAD_MM)
      aimMm = fmaxf(nearestAhead - PASS_LEAD_MM, PASS_AIM_MIN_MM);
    else
      aimMm = HOLD_AIM_MM;

    // Reach check: can the car still get to the target before the nearest
    // sign? If not, ask for a BACKOFF (reverse for run-up) while that sign
    // has budget left; otherwise commit to the correct side as is.
    if (laneOffOk && nearestIdx >= 0) {
      // Feasibility is judged on the RULE, not on comfort: the nearest lane
      // position that clears every sign in play (the bound), not the gap
      // centre. If the gap centre is out of reach but the bound is not, the
      // car aims at the bound instead - hugging a sign beats missing its side.
      float legal  = (lo > hi) ? urgentBound : constrain(laneOffMm, lo, hi);
      float need   = fabs(legal - laneOffMm);
      float sAvail = nearestAhead - REACH_LEAD_MM;
      float yawNow = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;       // + = left
      float psi0   = (legal > laneOffMm) ? yawNow : -yawNow;              // + = toward target
      float reach  = latReach(sAvail, psi0);
      if (fabs(target - laneOffMm) > reach) target = legal;
      reachNeedMm = need; reachHaveMm = reach;
      passFeasible = (need <= REACH_NEED_MIN_MM) || (need + BACKOFF_MARGIN_MM <= reach);
      bool unreachable = (need > REACH_NEED_MIN_MM) && (need > reach + REACH_MARGIN_MM);
      bool roomBehind  = laneAlongMm > -BACKOFF_BEHIND_MM + 20.0f;
      backoffWanted = (unreachable && BACKOFF_ENABLE && roomBehind &&
                       tracks[nearestIdx].backedMm < BACKOFF_MAX_MM) ? nearestIdx : -1;
    }
  } else {
    backoffWanted = -1;
  }

  latTarget = constrain(target, -LANE_LIMIT_MM, LANE_LIMIT_MM);

  if (!laneOffOk) { latYawCmd = 0.0f; return; }          // no walls: just hold heading
  // Centring is gentle (CENTRE_YAW_MAX) so the car does not weave; right
  // after a corner it gets POST_CORNER_YAW_MAX, with a sign in play PASS_YAW_MAX.
  float ymax = passActive ? PASS_YAW_MAX
             : (laneAlongMm < POST_CORNER_BOOST_MM ? POST_CORNER_YAW_MAX : CENTRE_YAW_MAX);
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
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(LED3_PIN, OUTPUT);
  led1(false);
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
// DRIVE STEERING  = heading PID on (lane heading + planner yaw)
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

// Forward. usePlanner = follow the lane planner (centring + signs);
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

// Reverse heading hold (P only). Reversing, the wheels act the other way:
// steering RIGHT swings the nose LEFT, so the sign of the correction flips.
void updateReverseSteer(float targetH) {
  if (!gImuFresh) return;
  float error = wrapDeg(targetH - gHeading);                 // + = nose must go left
  float want  = SERVO_TRUE_STRAIGHT + HEAD_KP * error;       // above straight = wheels right
  float dcmd  = constrain(want - prevServoCmd, -SERVO_SLEW, SERVO_SLEW);
  float cmd   = constrain(prevServoCmd + dcmd, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  setServoAngle(cmd);
  prevServoCmd = cmd;
}

// servo helpers for the turn: a steer magnitude to one side
float servoTravel(bool right) { return right ? (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT) : (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT); }
float servoSide(bool right, float mag) { return right ? SERVO_TRUE_STRAIGHT + mag : SERVO_TRUE_STRAIGHT - mag; }

// ============================================================
// CORNER EXIT PLAN  - decided at the corner trigger
// ============================================================
void planCornerExit() {
  float outer = clockwiseMode ? 1.0f : -1.0f;      // + lane offset = left of centre
  if (nextStraightColor != VIS_NONE) {
    // the passing rule is about the LANE: a sign passed on its right wants
    // the car on the right-hand side of the new lane, whatever the direction
    cornerExitCmd = (passRight(nextStraightColor) ? -1.0f : 1.0f) * CORNER_EXIT_BIAS_MM;
    Serial.print(nextStraightColor == VIS_RED ? F("# next straight RED -> exit ")
                                              : F("# next straight GREEN -> exit "));
    Serial.println(cornerExitCmd > 0 ? F("LEFT") : F("RIGHT"));
  } else {
    cornerExitCmd = outer * CORNER_EXIT_MM;        // nothing seen: the default
  }
}

// ============================================================
// STATE HELPERS
// ============================================================
void goState(RobotState s) { currentState = s; entered = false; }

// Overlays (RECOVER, BACKOFF) interrupt DRIVE or FINAL and return to it with
// its working variables intact - a one-deep return stack.
RobotState overlayReturnState   = STATE_DRIVE_TO_CORNER;
bool       overlayReturnEntered = false;
long       overlayBaseTicks     = 0;

void pushOverlay(RobotState s) {
  overlayReturnState   = currentState;
  overlayReturnEntered = entered;
  levelEnabled  = false;
  currentState  = s;
  entered       = false;
}

void popOverlay() {
  setMotorSpeed(0);
  currentState = overlayReturnState;
  entered      = overlayReturnEntered;
  resetHeadingPid();
  if (currentState == STATE_DRIVE_TO_CORNER || currentState == STATE_FINAL_STRAIGHT)
    setMotorSpeed(DRIVE_PWM);
}

int recoverTries = 0;

void finishCorner() {
  recoverTries = 0;                  // the panic budget is per straight
  if (cornerCount >= TARGET_CORNERS) goState(STATE_FINAL_STRAIGHT);
  else                               goState(STATE_DRIVE_TO_CORNER);
}

// ============================================================
// OVERLAY: RECOVER  (wall too close - back off, then resume)
// Not entered when the short front reading is a located sign right ahead:
// the planner is already steering round it.
// ============================================================
void recoverStep() {
  if (!entered) {
    entered = true;
    plannerEnabled = false;
    setMotorSpeed(0);
    setServoAngle(2.0 * SERVO_TRUE_STRAIGHT - lastServoCmd);   // mirror: nose swings away
    setMotorSpeed(-DRIVE_PWM);
    overlayBaseTicks = readEncoder();
    Serial.print(F("# RECOVER front=")); Serial.println(lidarF);
  }
  bool clear     = !lidarStale && (lidarF >= WALL_CLEAR_MM);
  bool backedFar = absEnc(readEncoder() - overlayBaseTicks) >= (long)(RECOVER_MAX_CM * TICKS_PER_CM);
  if (!clear && !backedFar) return;
  recoverTries++;                    // every RECOVER counts: a "clear" that re-panics on the
                                     //   same obstacle next revolution looped forever (sim)
  if (clear) { Serial.print(F("# recover clear, try "));  Serial.println(recoverTries); }
  else       { Serial.print(F("# recover capped, try ")); Serial.println(recoverTries); }
  popOverlay();
}

// ============================================================
// OVERLAY: BACKOFF  (reverse until the sign's correct side is reachable)
// ============================================================
int   backoffTrack = -1;
float backoffStartMm = 0.0;          // the track's backedMm when this backoff began

void backoffStep() {
  if (!entered) {
    entered = true;
    setMotorSpeed(0);
    resetHeadingPid();
    overlayBaseTicks = readEncoder();
    backoffStartMm = (backoffTrack >= 0) ? tracks[backoffTrack].backedMm : 0.0f;
    Serial.print(F("# BACKOFF ")); Serial.print(tracks[backoffTrack].color == VIS_RED ? F("RED") : F("GREEN"));
    Serial.print(F(" need ")); Serial.print((int)reachNeedMm);
    Serial.print(F(" reach ")); Serial.println((int)reachHaveMm);
  }
  plannerEnabled = true;             // keeps re-checking the reach while reversing
  bool gone = (backoffTrack < 0) || !tracks[backoffTrack].used;
  float backed = absEnc(readEncoder() - overlayBaseTicks) / TICKS_PER_MM;
  if (!gone) tracks[backoffTrack].backedMm = backoffStartMm + backed;

  bool capped = !gone && (tracks[backoffTrack].backedMm >= BACKOFF_MAX_MM ||
                          laneAlongMm <= -BACKOFF_BEHIND_MM);
  bool enough = backed >= BACKOFF_MIN_MM;
  if (gone || capped || (passFeasible && enough)) {
    if (capped) tracks[backoffTrack].backedMm = BACKOFF_MAX_MM;   // budget spent: commit, never re-ask
    Serial.print(capped ? F("# backoff capped, commit after ") : F("# backoff done after "));
    Serial.print((int)backed); Serial.println(F(" mm"));
    popOverlay();
    return;
  }
  setMotorSpeed(-BACKOFF_PWM);
  updateReverseSteer(laneHeading);   // straight back along the lane
}

// ============================================================
// STATE: WAIT START  (armed - waits for START)
// ============================================================
void waitStartStep() {
  if (!entered) {
    entered = true;
    startRequested = false;          // a START sent before arming is ignored
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    Serial.println(F("# WAIT_START armed - press Start"));
  }
  led1((millis() / 250) & 1);

  if (!startRequested) return;
  startRequested = false;
  if (lidarStale) { Serial.println(F("# START refused: lidar stale")); return; }

  led1(true);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(LED3_PIN, LOW);
  dirLocked        = false;
  clockwiseMode    = true;
  cornerCount      = 0;
  recoverTries     = 0;
  firstSegmentCm   = 0.0;
  fullStartStraightCm = 0.0;
  haveFullStraight = false;
  finalDistanceCm  = FINAL_STRAIGHT_CM;
  laneHeading      = gHeading;
  levelEnabled     = false;
  levelTotalDeg    = 0.0;
  cornerExitCmd    = 0.0;
  laneOffOk        = false;
  clearTracks();
  zeroEncoder();                     // zero FIRST, then take the lane origin from it
  resetLaneAlong();
  Serial.println(F("# GO"));
  goState(STATE_DRIVE_TO_CORNER);
}

// ============================================================
// STATE: DRIVE TO CORNER
//   drive on the lane planner; the corner is triggered by the LiDAR alone
// ============================================================
// Per side (0 = left, 1 = right): revolutions the wall was seen this straight,
// and consecutive revolutions it has read open since. Before the direction is
// known both sides are watched; after, only the inner side.
uint8_t sideWallRevs[2];
uint8_t sideOpenRevs[2];

void resetSideCounters() { sideWallRevs[0] = sideWallRevs[1] = 0; sideOpenRevs[0] = sideOpenRevs[1] = 0; }

// one new revolution of evidence for one side; true = that side is open now.
// Open = the side beam reads past SIDE_OPEN_MM AND (with SIDE_OPEN_NEED_CONE)
// the 45 deg cone fit on that side has lost the wall - the beam alone can be
// fooled, the cone needs most of a 45 deg wedge of wall to be missing.
bool sideEvidence(uint8_t s, uint16_t mm, uint16_t cone) {
  bool open = mm > SIDE_OPEN_MM && (!SIDE_OPEN_NEED_CONE || cone > LANE_VALID_MAX_MM);
  if (open) {
    if (sideWallRevs[s] >= SIDE_WALL_REVS && sideOpenRevs[s] < 255) sideOpenRevs[s]++;
  } else {
    sideOpenRevs[s] = 0;
    if (mm <= SIDE_OPEN_MM && sideWallRevs[s] < 255) sideWallRevs[s]++;
  }
  return sideOpenRevs[s] >= SIDE_OPEN_REVS;
}

// ---- how this corner will be turned (decided at the trigger) ----
enum TurnPlan { PLAN_ARC, PLAN_REVERSE };
TurnPlan turnPlan = PLAN_REVERSE;

// Optional (off by default = exactly the rule above). A forward arc started at
// TURN_OUTER_FRONT_MM leaves the car on the OUTER side of the new lane. If the
// NEXT straight's first sign must be passed on the new lane's INNER side (CW
// red / CCW green) and sits at the corner-exit seat, that pass is unreachable
// (sim: wrong side every lap). With this on, such a corner still arcs forward,
// but starts the arc at TURN_NEXT_INNER_FRONT_MM - early, as soon as the car is
// in the corner square - so the arc ends on the inner side of the new lane.
// (Switching to the reverse plan instead does NOT work: the reverse arc swings
// the car ~one turn radius toward the OLD lane's outer wall, and a car that
// approached on the outer side hits it.) The colour comes from a sighting
// across the corner before the trigger, or from the camera during the approach.
       bool     TURN_NEXT_OVERRIDE = false;
       uint16_t TURN_NEXT_INNER_FRONT_MM = 900;

// a sign the turn needs on the new lane's inner side
bool needsInnerOfNewLane(int color) { return color != VIS_NONE && passRight(color) == clockwiseMode; }

void driveStep() {
  if (!entered) {
    entered = true;
    resetSideCounters();
    resetHeadingPid();
    setMotorSpeed(DRIVE_PWM);
    Serial.println(F("# DRIVE"));
  }

  plannerEnabled = true;
  updateDriveSteer(laneHeading, true);

  float runMm = absEnc(readEncoder()) / TICKS_PER_MM;        // since the last corner (or START)
  levelEnabled = runMm > POST_CORNER_LOCKOUT_CM * 10.0f;

  // ---- corner trigger: the inner-side beam opens ----
  bool aligned = fabs(wrapDeg(gHeading - laneHeading)) < TURN_TRIGGER_MAX_YAW;
  if (lidarStale || !aligned) { sideOpenRevs[0] = sideOpenRevs[1] = 0; return; }
  if (!lidarNewRev) return;                                  // beams change once per revolution
  if (cornerCount > 0 && runMm < CORNER_LOCKOUT_MM) return;  // still leaving the last corner

  bool openL = (!dirLocked || !clockwiseMode) && sideEvidence(0, lidarL, coneL);
  bool openR = (!dirLocked ||  clockwiseMode) && sideEvidence(1, lidarR, coneR);
  if (!openL && !openR) return;
  if (!dirLocked) {
    if (openL && openR) {                                    // both at once: not a corner we can read
      sideOpenRevs[0] = sideOpenRevs[1] = 0;
      return;
    }
    dirLocked     = true;
    clockwiseMode = openR;                                   // outer wall never ends: open side = inner
    Serial.println(clockwiseMode ? F("# LOCKED CW (right side opened)")
                                 : F("# LOCKED CCW (left side opened)"));
  }
  Serial.print(F("# turn: side open ")); Serial.print(turnSideMm());
  Serial.print(F(" run=")); Serial.println((int)runMm);

  // segment lengths for the final straight: A = start -> first trigger,
  // L = the start straight driven in full (laps 2 and 3, averaged)
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

  // ---- the corner plan, from the last sign of this straight ----
  //   passed on the OUTER side (CW green, CCW red)  -> straight to 400 mm, forward arc
  //   passed on the INNER side (CW red, CCW green)  -> straight to 200 mm, reverse arc
  //   no sign on this straight                      -> as the inner case
  int last = lastSignOfStraight();
  bool outerPass = (last != VIS_NONE) && (passRight(last) != clockwiseMode);
  turnPlan = outerPass ? PLAN_ARC : PLAN_REVERSE;
  Serial.print(F("# last sign "));
  Serial.print(last == VIS_RED ? F("RED") : last == VIS_GREEN ? F("GREEN") : F("none"));
  Serial.println(outerPass ? F(" (outer) -> straight to TURN_OUTER_FRONT_MM, forward arc")
                           : F(" (inner/none) -> straight to TURN_INNER_FRONT_MM, reverse arc"));
  goState(STATE_TURNING);
}

// ============================================================
// STATE: TURNING
// ============================================================
// APPROACH  drive straight on the old lane heading until the wall ahead is
//           TURN_OUTER_FRONT_MM (PLAN_ARC) or TURN_INNER_FRONT_MM
//           (PLAN_REVERSE) away. TURN_APPROACH_CAP_CM is the odometry
//           backstop for a stale or missing front reading.
// FWD       PLAN_ARC: one forward arc toward the turn at TURN_LOCK_FRACTION of
//           full lock, eased by the IMU (TURN_KP x degrees still to go) onto
//           the new lane heading. If the wall ahead closes to
//           TURN_FWD_FRONT_MM first, it finishes with SWING + REV instead.
// SWING     stopped, wheels to the OPPOSITE lock, TURN_SWING_MS
// REV       PLAN_REVERSE: reverse at the opposite lock - that keeps rotating
//           the car the same way - eased by the IMU onto the new lane heading.
//           Blind: capped at TURN_REV_CAP_CM.
// VIEW      stopped, wheels straight, planner already in the new lane frame,
//           TURN_VIEW_MS - the camera sees the next straight before the car
//           commits to a side.
       uint16_t TURN_OUTER_FRONT_MM = 400;   // PLAN_ARC: arc when the wall ahead is this close
       uint16_t TURN_INNER_FRONT_MM = 200;   // PLAN_REVERSE: reverse arc from this close
       float    TURN_APPROACH_CAP_CM = 150.0; // approach backstop (odometry)
       float    TURN_LOCK_FRACTION = 1.0;    // forward arc: fraction of full lock
       uint16_t TURN_FWD_FRONT_MM  = 150;    // forward arc: wall this close -> finish in reverse
       float    TURN_FWD_CAP_CM    = 80.0;   // forward arc odometry backstop
       unsigned long TURN_SWING_MS = 150;    // stopped while the wheels swing across
       float    TURN_REV_LOCK      = 1.0;    // reverse arc: fraction of full opposite lock
       float    TURN_KP            = 2.5;    // easing: servo deg per deg still to go
       float    TURN_MIN_STEER     = 8.0;
       float    TURN_DONE_DEG      = 4.0;    // facing the new lane within this = done
       float    TURN_REV_CAP_CM    = 60.0;   // reverse at most this far (blind); a 90 deg
                                             //   arc at ~27 cm radius is ~42 cm
       unsigned long TURN_VIEW_MS  = 200;    // look down the new lane before driving (0 = off)

enum TurnPhase { TP_APPROACH, TP_FWD, TP_SWING, TP_REV, TP_VIEW };
TurnPhase     turnPhase;
bool          turnEarlyLogged = false;
float         turnOldLane, turnNewLane;
long          turnBaseTicks;
unsigned long turnT0;

// rotation toward the turn since the old lane (deg, + = the right way)
float turnTurned() {
  float d = wrapDeg(gHeading - turnOldLane);
  return clockwiseMode ? -d : d;
}

void startSwing(const __FlashStringHelper *why) {
  setMotorSpeed(0);
  turnT0 = millis();
  turnPhase = TP_SWING;
  Serial.print(F("# turn swing (")); Serial.print(why);
  Serial.print(F("), turned ")); Serial.print(turnTurned());
  Serial.print(F(" front ")); Serial.println(lidarF);
}

// the corner is done: the new lane becomes the reference frame
void enterTurnView() {
  setMotorSpeed(0);
  setServoAngle(SERVO_TRUE_STRAIGHT);
  laneHeading = turnNewLane;
  zeroEncoder();
  resetLaneAlong();
  clearTracks();                      // anything seen mid-turn was in the old lane frame
  laneOffOk = false;                  // accept the new lane's first offset without the jump filter
  offJumps  = 0;
  plannerEnabled = true;              // start sighting the new lane while stopped
  turnT0 = millis();
  turnPhase = TP_VIEW;
  Serial.print(F("# lane heading ")); Serial.print(laneHeading);
  Serial.print(F(" turned ")); Serial.println(turnTurned());
}

void turningStep() {
  bool turnRight = clockwiseMode;
  if (!entered) {
    entered = true;
    levelEnabled   = false;
    plannerEnabled = false;
    clearTracks();
    Serial.print(F("# level ")); Serial.println(levelTotalDeg);
    levelTotalDeg = 0.0;
    cornerCount++;
    Serial.print(F("# TURN ")); Serial.print(cornerCount);
    Serial.print('/'); Serial.print(TARGET_CORNERS);
    Serial.println(turnPlan == PLAN_ARC ? F(" ARC") : F(" REVERSE"));
    turnOldLane   = laneHeading;
    turnNewLane   = wrapDeg(laneHeading + (clockwiseMode ? -90.0f : 90.0f));
    turnPhase     = TP_APPROACH;
    turnEarlyLogged = false;
    turnBaseTicks = readEncoder();
    resetHeadingPid();
    setMotorSpeed(DRIVE_PWM);
  }

  float left = 90.0f - turnTurned();            // degrees still to rotate (< 0 = overshoot)

  switch (turnPhase) {
    case TP_APPROACH: {
      // a sign on the turn side, well off the old lane's line, belongs to the
      // next straight - remember its colour for the override
      for (const Sighting *sg : { &sign1, &sign2 })
        if (sg->valid && sg->x > 150.0f && (clockwiseMode ? sg->y < -300.0f : sg->y > 300.0f))
          nextStraightColor = sg->color;
      bool early = TURN_NEXT_OVERRIDE && turnPlan == PLAN_ARC && needsInnerOfNewLane(nextStraightColor);
      uint16_t stopAt = (turnPlan == PLAN_REVERSE) ? TURN_INNER_FRONT_MM
                      : (early ? TURN_NEXT_INNER_FRONT_MM : TURN_OUTER_FRONT_MM);
      if (early && !turnEarlyLogged) {
        turnEarlyLogged = true;
        Serial.println(F("# next straight needs the inner side -> early arc"));
      }
      bool close  = !lidarStale && lidarF <= stopAt;
      bool capped = absEnc(readEncoder() - turnBaseTicks) >= (long)(TURN_APPROACH_CAP_CM * TICKS_PER_CM);
      if (close || capped) {
        if (capped && !close) Serial.println(F("# turn approach capped (no front reading)"));
        turnBaseTicks = readEncoder();
        if (turnPlan == PLAN_ARC) {
          turnPhase = TP_FWD;
          Serial.print(F("# turn arc, front ")); Serial.println(lidarF);
        } else {
          startSwing(F("reverse plan"));
        }
        break;
      }
      setMotorSpeed(DRIVE_PWM);
      updateDriveSteer(turnOldLane, false);     // straight on the old lane heading
      break;
    }
    case TP_FWD: {
      if (left <= TURN_DONE_DEG) { enterTurnView(); break; }
      bool wall   = !lidarStale && lidarF <= TURN_FWD_FRONT_MM;
      bool capped = absEnc(readEncoder() - turnBaseTicks) >= (long)(TURN_FWD_CAP_CM * TICKS_PER_CM);
      if (wall || capped) { startSwing(wall ? F("wall ahead") : F("forward cap")); break; }
      float mag = constrain(TURN_KP * left, TURN_MIN_STEER, TURN_LOCK_FRACTION * servoTravel(turnRight));
      setServoAngle(servoSide(turnRight, mag));
      setMotorSpeed(DRIVE_PWM);
      break;
    }
    case TP_SWING:
      setMotorSpeed(0);
      setServoAngle(servoSide(!turnRight, TURN_REV_LOCK * servoTravel(!turnRight)));
      if (millis() - turnT0 >= TURN_SWING_MS) {
        turnBaseTicks = readEncoder();
        turnPhase = TP_REV;
      }
      break;
    case TP_REV: {
      bool capped = absEnc(readEncoder() - turnBaseTicks) >= (long)(TURN_REV_CAP_CM * TICKS_PER_CM);
      if (left <= TURN_DONE_DEG || capped) {
        if (capped && left > TURN_DONE_DEG) { Serial.print(F("# turn reverse capped, left ")); Serial.println(left); }
        enterTurnView();
        break;
      }
      float mag = constrain(TURN_KP * left, TURN_MIN_STEER, TURN_REV_LOCK * servoTravel(!turnRight));
      setServoAngle(servoSide(!turnRight, mag));
      setMotorSpeed(-DRIVE_PWM);
      break;
    }
    case TP_VIEW:
      setMotorSpeed(0);
      if (millis() - turnT0 >= TURN_VIEW_MS) finishCorner();
      break;
  }
}

// ============================================================
// STATE: FINAL STRAIGHT  (drive L - A from the end of turn 12)
// Signs can sit in the start section, so the full planner runs.
// ============================================================
long fsTargetTicks;

void finalStraightStep() {
  if (!entered) {
    entered = true;
    Serial.print(F("# FINAL_STRAIGHT ")); Serial.println(finalDistanceCm);
    resetHeadingPid();
    setMotorSpeed(DRIVE_PWM);
    fsTargetTicks = (long)(finalDistanceCm * TICKS_PER_CM);
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
// firmware needs no storage of its own.
//
// WIRE FORMAT  (newline-terminated ASCII, sharing the port with the frames)
//   Pi -> STM32   N <name> <value>   set by name
//                 ?P                 dump the whole table (streamed)
//                 ?V                 report version / boot id
//   STM32 -> Pi   !V <ver> <count> <boot>                      boot and ?V
//                 !P <id> <name> <type> <val> <lo> <hi> <group> one per ?P line
//                 !p <id> <val>                                 set acknowledged
//                 !E <what>                                     rejected
//
// The dump is streamed: at most PARAM_DUMP_PER_LOOP lines per loop, and only
// while the USB TX buffer has room, so it never blocks the control loop.
// Derived values (PASS_CLEAR_MM, LANE_LIMIT_MM, TICKS_PER_MM) are recomputed
// on every set, so arrival order never matters.

const uint16_t PARAM_VERSION        = 6;     // bump when ids are added/removed
const uint8_t  PARAM_DUMP_PER_LOOP  = 2;
const uint16_t PARAM_TX_HEADROOM    = 96;    // bytes free before a dump line

enum PType : uint8_t { PT_F, PT_I, PT_U32, PT_U16, PT_U8, PT_B };

// groups, purely for how the Pi page lays the table out
enum PGroup : uint8_t {
  G_DRIVE, G_TURN, G_PLAN, G_PASS, G_LEVEL, G_BACK, G_CORNER, G_SAFE, G_LINK
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

  // ---- corner trigger (LiDAR) + turn ----
  PS(SIDE_OPEN_MM,          G_TURN,  300,  3000),
  PC(SIDE_OPEN_REVS,        G_TURN,    1,    20),
  PC(SIDE_WALL_REVS,        G_TURN,    0,    20),
  PF(TURN_TRIGGER_MAX_YAW,  G_TURN,    5,    90),
  PF(CORNER_LOCKOUT_MM,     G_TURN,    0,  2000),
  PB(SIDE_OPEN_NEED_CONE,   G_TURN),
  PS(TURN_OUTER_FRONT_MM,   G_TURN,   80,  2000),
  PS(TURN_INNER_FRONT_MM,   G_TURN,   80,  2000),
  PF(TURN_APPROACH_CAP_CM,  G_TURN,   10,   400),
  PF(TURN_LOCK_FRACTION,    G_TURN,  0.2,     1),
  PS(TURN_FWD_FRONT_MM,     G_TURN,    0,  1500),
  PF(TURN_FWD_CAP_CM,       G_TURN,   10,   300),
  PL(TURN_SWING_MS,         G_TURN,    0,  2000),
  PF(TURN_REV_LOCK,         G_TURN,  0.2,     1),
  PF(TURN_KP,               G_TURN,  0.2,    10),
  PF(TURN_MIN_STEER,        G_TURN,    0,    45),
  PF(TURN_DONE_DEG,         G_TURN,  0.5,    30),
  PF(TURN_REV_CAP_CM,       G_TURN,    5,   150),
  PL(TURN_VIEW_MS,          G_TURN,    0,  3000),
  PB(TURN_NEXT_OVERRIDE,    G_TURN),
  PS(TURN_NEXT_INNER_FRONT_MM, G_TURN, 80, 2000),
  PI_(TARGET_CORNERS,       G_TURN,    1,    48),
  PF(FINAL_STRAIGHT_CM,     G_TURN,    0,   400),
  PF(POST_CORNER_LOCKOUT_CM,G_TURN,    0,   200),

  // ---- corner exit ----
  PF(CORNER_EXIT_MM,        G_CORNER,-400,   400),
  PF(CORNER_EXIT_BIAS_MM,   G_CORNER,   0,   400),
  PF(POST_CORNER_YAW_MAX,   G_CORNER,   5,    75),
  PF(POST_CORNER_BOOST_MM,  G_CORNER,   0,  1500),

  // ---- lane / sign planner ----
  PF(CORRIDOR_MM,           G_PLAN,  300,  2000),
  PF(CAR_HALF_W_MM,         G_PLAN,   20,   200),
  PF(PILLAR_HALF_MM,        G_PLAN,    5,   100),
  PF(PASS_MARGIN_MM,        G_PLAN,    0,   300),
  PF(WALL_MARGIN_MM,        G_PLAN,    0,   300),
  PF(GAP_CENTRE_W,          G_PLAN,    0,     1),
  PF(CENTRE_AIM_MM,         G_PLAN,  100,  2000),
  PF(OFF_JUMP_MM,           G_PLAN,   20,   500),
  PC(OFF_JUMP_REVS,         G_PLAN,    0,    20),
  PF(CORRIDOR_SUM_TOL_MM,   G_PLAN,   20,   500),
  PF(CENTRE_YAW_MAX,        G_PLAN,    2,    60),
  PS(LANE_VALID_MAX_MM,     G_PLAN,  300,  3000),
  PF(PLAN_MAX_AHEAD_MM,     G_PLAN,  300,  3000),
  PF(SIGHT_MIN_ALONG_MM,    G_PLAN, -500,     0),
  PF(SIGHT_MAX_YAW,         G_PLAN,    5,    90),
  PF(PILLAR_MAX_LAT_MM,     G_PLAN,  100,  1000),
  PF(CROSS_MAX_LAT_MM,      G_PLAN,  400,  3000),

  // ---- passing a sign ----
  PF(PASS_LEAD_MM,          G_PASS,    0,   600),
  PF(PASS_AIM_MIN_MM,       G_PASS,   40,   600),
  PF(HOLD_AIM_MM,           G_PASS,   50,  1000),
  PF(PASS_YAW_MAX,          G_PASS,    5,    89),
  PF(PASS_HOLD_MM,          G_PASS,    0,   600),
  PF(TURN_RADIUS_MM,        G_PASS,  100,   800),
  PF(REACH_LEAD_MM,         G_PASS,    0,   400),
  PF(REACH_NEED_MIN_MM,     G_PASS,    0,   300),
  PF(REACH_MARGIN_MM,       G_PASS,    0,   300),
  PF(TRACK_MATCH_MM,        G_PASS,   50,   600),
  PF(TRACK_GAIN,            G_PASS, 0.05,     1),
  PC(TRACK_CONFIRM,         G_PASS,    1,    20),
  PF(TRACK_FORGET_MM,       G_PASS,   50,  1500),

  // ---- wall levelling ----
  PF(LEVEL_GAIN,            G_LEVEL,   0,     1),
  PF(LEVEL_MAX_STEP,        G_LEVEL,   0,     5),
  PF(LEVEL_MAX_DIFF,        G_LEVEL,   0,    45),
  PF(LEVEL_MAX_WALLANG,     G_LEVEL,   0,    45),

  // ---- reverse and re-plan ----
  PB(BACKOFF_ENABLE,        G_BACK),
  PI_(BACKOFF_PWM,          G_BACK,    0,   255),
  PF(BACKOFF_MAX_MM,        G_BACK,    0,  1000),
  PF(BACKOFF_MARGIN_MM,     G_BACK,    0,   300),
  PF(BACKOFF_BEHIND_MM,     G_BACK,    0,   600),
  PF(BACKOFF_MIN_MM,        G_BACK,    0,   500),

  // ---- safety ----
  PS(WALL_PANIC_MM,         G_SAFE,    0,  1000),
  PS(WALL_CLEAR_MM,         G_SAFE,   50,  1500),
  PF(RECOVER_MAX_CM,        G_SAFE,     5,  100),
  PI_(RECOVER_MAX_TRIES,    G_SAFE,     0,   20),
  PF(PANIC_PILLAR_MM,       G_SAFE,     0,  1000),

  // ---- link / sensors ----
  PL(LIDAR_STALE_MS,        G_LINK,   20,  2000),
  PS(LIDAR_MAX_VALID_MM,    G_LINK,  500,  9000),
  PF(ORANGE_PR_MIN,         G_LINK,    0,   100),   // floor colour: LEDs only
  PF(ORANGE_PB_MAX,         G_LINK,    0,   100),
  PF(BLUE_PB_MIN,           G_LINK,    0,   100),
  PF(BLUE_PR_MAX,           G_LINK,    0,   100),
  PF(COLOR_SUM_MIN,         G_LINK,    0,  5000),
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
  if (s[0] == '?' && s[1] == 'P' && s[2] == '\0') { paramDumpIdx = 0; return true; }
  if (s[0] == '?' && s[1] == 'V' && s[2] == '\0') { reportVersion();  return true; }
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

bool moving()    { return currentState == STATE_DRIVE_TO_CORNER || currentState == STATE_FINAL_STRAIGHT; }

// ---- start / stop button (PB12 to GND, pull-up) ----
// Debounced: the level must hold BTN_DEBOUNCE_MS before it counts. A press
// fires once, on the press edge; the button must be released before the next
// one counts, and presses closer than BTN_LOCKOUT_MS are ignored, so a bounce
// or a long press can never STOP a run it has just STARTed.
const unsigned long BTN_DEBOUNCE_MS = 30;
const unsigned long BTN_LOCKOUT_MS  = 600;
bool          btnStable   = HIGH;       // debounced level (HIGH = released)
bool          btnLastRaw  = HIGH;
unsigned long btnRawSince = 0;
unsigned long btnLastFire = 0;
bool          btnFired    = false;      // no lockout before the first press

bool buttonPressed() {
  bool raw = digitalRead(BTN_PIN);
  unsigned long now = millis();
  if (raw != btnLastRaw) { btnLastRaw = raw; btnRawSince = now; }
  if (raw == btnStable || now - btnRawSince < BTN_DEBOUNCE_MS) return false;
  btnStable = raw;
  if (btnStable != LOW) return false;                  // a release, not a press
  if (btnFired && now - btnLastFire < BTN_LOCKOUT_MS) return false;
  btnFired    = true;
  btnLastFire = now;
  return true;
}

// Same effect as the page: armed / finished -> START, otherwise -> STOP.
void serviceButton() {
  if (!buttonPressed()) return;
  if (currentState == STATE_WAIT_START || currentState == STATE_FINISHED) {
    Serial.println(F("# START from button"));
    startRequested = true;
  } else {
    stopRequested = true;
    stopByButton  = true;
  }
}

void loop() {
  serviceSensors();
  paramDumpStep();        // streamed, at most PARAM_DUMP_PER_LOOP lines

  // ---- startup gate: nothing runs until the Pi's first frame ----
  if (!fsmStarted) {
    if (lidarFrames == 0) {
      led1((millis() / 500) & 1);
      return;
    }
    fsmStarted = true;
    Serial.println(F("# first Pi frame received - FSM starting"));
  }

  serviceButton();        // sets startRequested / stopRequested exactly like the page

  // START only means something while armed or finished; anything else (a
  // press mid-run, the repeat copies of the press that started this run) is
  // discarded so it can't auto-restart the car when it finishes.
  if (currentState != STATE_WAIT_START && currentState != STATE_FINISHED)
    startRequested = false;

  // STOP works in every state.
  if (stopRequested) {
    stopRequested = false;
    if (currentState != STATE_FINISHED) {
      Serial.println(stopByButton ? F("# STOP from button") : F("# STOP from Pi"));
      goState(STATE_FINISHED);
    }
    stopByButton = false;
  }

  // Wall panic -> RECOVER, from DRIVE or FINAL only (TURNING stops short of
  // the wall itself). Skipped when the short front reading is a located sign
  // right ahead: the planner is steering round it.
  bool signAhead = sign1.valid && sign1.x < PANIC_PILLAR_MM &&
                   fabs(sign1.y) < CAR_HALF_W_MM + PILLAR_HALF_MM + 50.0f;
  if (moving() && !lidarStale && !signAhead &&
      recoverTries < RECOVER_MAX_TRIES && lidarF <= WALL_PANIC_MM) {
    plannerEnabled = false;
    pushOverlay(STATE_RECOVER);
  }

  // The planner asks for a BACKOFF when the nearest sign's correct side is
  // out of reach and that sign still has reverse budget.
  if (moving() && !lidarStale && backoffWanted >= 0) {
    backoffTrack  = backoffWanted;
    backoffWanted = -1;
    pushOverlay(STATE_BACKOFF);
  }

  if (currentState != STATE_WAIT_START && currentState != STATE_FINISHED) {
    led1(lidarStale ? ((millis() / 100) & 1) : true);
  }

  switch (currentState) {
    case STATE_WAIT_START:      waitStartStep();     break;
    case STATE_DRIVE_TO_CORNER: driveStep();         break;
    case STATE_TURNING:         turningStep();       break;
    case STATE_FINAL_STRAIGHT:  finalStraightStep(); break;
    case STATE_RECOVER:         recoverStep();       break;
    case STATE_BACKOFF:         backoffStep();       break;

    case STATE_FINISHED:
      if (!entered) {
        entered = true;
        Serial.println(F("# FINISHED"));
        levelEnabled = false;
        plannerEnabled = false;
        startRequested = false;      // only a NEW press restarts
        setMotorSpeed(0);
        setServoAngle(SERVO_TRUE_STRAIGHT);
        led1(true);
        digitalWrite(LED2_PIN, HIGH);
        digitalWrite(LED3_PIN, HIGH);
      }
      // START again re-arms and runs a fresh 3 laps
      // (put the car back in the start section first).
      if (startRequested) {
        goState(STATE_WAIT_START);   // WAIT_START clears the flag on entry,
        waitStartStep();             // so arm first ...
        startRequested = true;       // ... then honour this press
      }
      break;
  }
}
