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
// OBSTACLE ROUND - NON-BLOCKING FIRMWARE  (v9, param table v8)
//
// v9 vs v7
//   - one sign per frame (the largest blob); fields 15-17 are parsed past
//   - corner trigger: the side beam alone (no cone test); before the direction
//     is known the side with the LONGER beam is the candidate; an odometry
//     window (path since the last corner trigger) gates it:
//       < CORNER_MIN_RUN_MM (2000)       never a corner
//       > CORNER_FALLBACK_RUN_MM (2700)  relaxed evidence (1 rev, no wall history)
//   - more air to a sign (PASS_MARGIN_MM 150) and a yaw cap while alongside
//     one (ALONGSIDE_YAW_MAX), so the swept body corner cannot clip it
//   - reach check rebuilt: a forward simulation of the real steering loop,
//     judged against the car's swept outline, decided once per LiDAR rev in a
//     distance window, confirmed over REACH_CONFIRM revs (stopped); a BACKOFF
//     reverses a PLANNED distance in one go (see "REACH" below), never with
//     the rear corner at a wall
//   - the planner looks at ONE sign ahead only - the nearest - never at the
//     one after it; PASS_HOLD_MM 250 -> 30
//   - a PLAN_REVERSE corner approaches on the inner half (TURN_REV_INNER_MM):
//     the reverse arc swings the car toward the old lane's outer wall
//   All of it was run against a host simulation of the field (the real .ino
//   compiled for the PC, 100 random sign layouts x 2 seat patterns).
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
//   15-17 sColor,sX,sY       second sign - parsed past, NOT used (v9: the Pi
//                            sends 2,32767,32767; one sign, the largest blob)
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
//      known the side with the longer beam is the candidate: the outer wall
//      never ends, so the side that opens is the inner side - right =
//      clockwise, left = anticlockwise. After the first corner, odometry
//      gates it (CORNER_MIN_RUN_MM / CORNER_FALLBACK_RUN_MM).
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
//   4. A sign whose correct side the reach simulation says cannot be reached
//      makes the car reverse in a straight line (BACKOFF) by a planned
//      distance, up to BACKOFF_MAX_MM per sign, then it commits.
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

// Heading PID (used by updateDriveSteer, and mirrored by the REACH simulation)
       float HEAD_KP        = 2.0;
       float YAW_FILT_ALPHA = 0.35;
       float SERVO_SLEW     = 2.5;     // servo deg per IMU update (~100 Hz)
       float INTEGRAL_CLAMP = 300.0;
       float HEAD_KI        = 0.0;
       float HEAD_KD        = 0.0;


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

// One sign per frame (struct Sighting is declared at the top of the file).
const long PXY_NONE = 32767;
Sighting sign1 = { false, VIS_NONE, 0, 0 };      // largest accepted blob
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

  // fields 13-14 (uX, uY: uncoloured LiDAR object) and 15-17 (second sign)
  // are ignored - see the header

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
// Odometry window (v9), after the first corner only - the start position in
// the first straight is unknown. Measured on the encoder since the last turn
// FINISHED (signed: a BACKOFF / RECOVER reverse counts back) - the length of
// this straight so far. In the host simulation of the full lap the real
// corners trigger at 1500-2100 mm of that (reverse-arc corners end farther
// back than forward-arc ones); a false "open" seen right after a turn came
// at ~800-900 mm.
//   < CORNER_MIN_RUN_MM       no corner, whatever the beam says
//   MIN .. FALLBACK           normal evidence (SIDE_WALL_REVS + SIDE_OPEN_REVS)
//   > CORNER_FALLBACK_RUN_MM  overdue: 1 open revolution is enough, no wall
//                             history needed - a safety net, not the main rule
       float    CORNER_MIN_RUN_MM      = 1200.0;
       float    CORNER_FALLBACK_RUN_MM = 2300.0;

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

// TIM5 encoder, negated so driving forward counts up (zeroEncoder() is with
// the corner odometer, below the planner constants)
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
       float    CAR_HALF_LEN_MM   = 85.0;    // v9: LiDAR to the far end of the body (measure it!).
                                             //   A car yawed psi sweeps a band
                                             //   W cos(psi) + L sin(psi) wide each side, not W:
                                             //   at 40 deg that is 98 mm, not 57
       float    PILLAR_HALF_MM    = 25.0;
       float    PASS_MARGIN_MM    = 150.0;   // air gap car side <-> sign face; v9: was 80 (clipped)
      float    PASS_CLEAR_MM     = 232.0;   // derived, see recomputeDerived()
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
       float    ALONGSIDE_YAW_MAX = 15.0;    // v9: deg, while any sign in play is beside the body:
                                             //   the swept corner stays within ~20 mm of W
       float    TURN_RADIUS_MM    = 270.0;   // full-lock radius (worse side) for the reach simulation
       float    PLAN_MAX_AHEAD_MM = 1600.0;  // ignore sightings farther ahead
       float    SIGHT_MIN_ALONG_MM = -50.0;  // ... or farther behind
       float    SIGHT_MAX_YAW     = 30.0;    // deg: car turned further than this off the lane ->
                                             //   ignore NEW sightings (the camera is looking across
                                             //   the corner / island into other lanes); tracks
                                             //   already confirmed keep steering the car
       float    PILLAR_MAX_LAT_MM = 330.0;   // seats are at +/-100; beyond this = another straight's sign
                                             //   v7: was 420, let other lanes' signs in after a corner
       float    CROSS_MAX_LAT_MM  = 1300.0;  // beyond this it is not even the next straight
       float    PASS_HOLD_MM      = 30.0;    // keep a sign's side until it is this far behind the LiDAR;
                                             //   v9: was 250. Once the sign is behind the LiDAR,
                                             //   turning toward it swings the body part beside it
                                             //   AWAY, and the next sign needs every mm of run-up
                                             //   (host sim: 250 -> late swing -> next sign hit)
       float    TRACK_MATCH_MM    = 200.0;   // same sign if within this (along and lateral)
       float    TRACK_GAIN        = 0.5;     // how far a new sighting moves a track estimate
       uint8_t  TRACK_CONFIRM     = 2;       // sightings before a sign steers the car
       float    TRACK_FORGET_MM   = 300.0;   // unconfirmed and not seen for this far -> dropped
       uint16_t LANE_VALID_MAX_MM = 1100;    // cone farther than this = no wall on that side
      float    TICKS_PER_MM      = 1.4853;  // derived, see recomputeDerived()

// ---- REACH: can the car still get to the sign's correct side? (v9) ----
// v7 compared an ideal "full lock, then straight" arc against the distance
// left, every frame, against the comfort target. It was wrong both ways:
// the real loop (HEAD_KP, SERVO_SLEW, the aim law) turns much later than
// full lock, so it said "reachable" and the car arrived still yawed and
// clipped the sign; one noisy frame said "unreachable" and the car reversed
// for nothing. Now:
//   1. SIMULATE the real loop: 10 mm steps of the same aim law, the same
//      yaw caps, the same PID and slew (at the measured speed, never below
//      REACH_SPEED_MMPS) and the servo -> curvature of TURN_RADIUS_MM.
//   2. JUDGE the swept outline, not the centre: over the whole stretch where
//      the body overlaps the sign along the lane, the side of the car toward
//      the sign (W cos psi + L sin psi) must stay REACH_SAFETY_MM outside the
//      sign's face. Result: the smallest clearance, mm (< 0 = contact).
//   3. DECIDE rarely and only where the answer is good: once per new LiDAR
//      revolution, only for the nearest sign, only while it is between
//      REACH_DECIDE_NEAR_MM and REACH_DECIDE_FAR_MM ahead, only once it has
//      REACH_MIN_HITS sightings. Nearer than NEAR the car never reverses.
//   4. CONFIRM: REACH_CONFIRM revolutions in a row must fail before acting.
//      On the first failing one the car STOPS (REACH_PAUSE_MS at most) and
//      confirms standing still, so it is not swerving deeper while it decides.
//      The sign before this one (being passed, or passed and remembered) is
//      part of the simulation: the car is held on its line until it is
//      PASS_HOLD behind. Nothing beyond the nearest sign ahead is planned for.
//   5. ACT, cheapest first: the gap-centre target fails but the rule line
//      (the bound, PASS_MARGIN from the sign) passes -> aim at the bound for
//      this sign ("hug"). Both fail -> BACKOFF by a PLANNED distance: the
//      shortest reverse after which the simulation clears by
//      BACKOFF_MARGIN_MM, reversed in one go (no re-checking every frame).
//      Budget spent (BACKOFF_MAX_MM per sign) -> hug and commit.
// Every decision is logged as a "# REACH" line.
       float    REACH_SAFETY_MM      = 20.0;   // extra air the simulated outline must keep
       float    REACH_DECIDE_NEAR_MM = 300.0;  // closer than this: never reverse, commit
       float    REACH_DECIDE_FAR_MM  = 900.0;  // farther than this: the track is too rough to judge
       uint8_t  REACH_MIN_HITS       = 3;      // sightings before a sign is judged
       uint8_t  REACH_CONFIRM        = 3;      // failing revolutions in a row before acting
       float    REACH_SPEED_MMPS     = 400.0;  // slowest speed the simulation assumes
       unsigned long REACH_PAUSE_MS  = 700;    // on the FIRST failing revolution the car stops and
                                               //   confirms standing still (host sim: while it kept
                                               //   driving, the swerve toward the sign yawed it 20-30
                                               //   deg before the verdict, and a yawed car at the
                                               //   outer wall cannot back off); 0 = never stop

// ---- reverse and re-plan (BACKOFF) ----
// Reverses in a straight line along the lane (every mm is a mm of run-up) by
// the distance the REACH plan asked for, then drives again. Reversing is
// BLIND (nothing measures behind the car yet), so keep BACKOFF_MAX_MM short.
// Touching a sign is allowed while it stays in its circle (rule 9.20); a
// wrong-side pass ends the round (9.24.5) - that is why a spent budget still
// commits to the correct side.
       bool     BACKOFF_ENABLE    = true;
       int      BACKOFF_PWM       = 50;
       float    BACKOFF_MAX_MM    = 300.0;   // per sign; v9: was 250
       float    BACKOFF_MARGIN_MM = 30.0;    // v9: the planned reverse must make the simulated
                                             //   clearance at least this (was: reach - need)
       float    BACKOFF_MIN_MM    = 80.0;    // reverse at least this far each time (no 20 mm dithering)
       float    BACKOFF_BEHIND_MM = 200.0;   // may reverse this far behind where the straight
                                             //   began: after a forward-arc corner the car exits on the
                                             //   outer side with ~300 mm of corner square behind it, and
                                             //   a sign right at the exit needs that run-up (blind!)
       unsigned long BACKOFF_TIMEOUT_MS = 3000; // v9: no encoder progress for this long = stuck, stop
       float    BACKOFF_WALL_MM   = 10.0;    // v9: rear corner <-> wall while reversing (host sim: a
                                             //   BACKOFF started yawed 20 deg at the outer limit backed
                                             //   the rear corner into the wall)

struct PillarTrack { bool used; int color; float lat; float along; uint8_t hits; float lastSeen;
                     float backedMm; bool hug; bool passed; };
const int   MAX_TRACKS = 6;
const float TRACK_KEEP_BEHIND_MM = 600.0f;  // v9: a passed sign is remembered this far behind, so
                                            //   a BACKOFF that reverses the car back beside it
                                            //   finds its bound still there
PillarTrack tracks[MAX_TRACKS];

float laneOffMm   = 0.0;     // + = car left of lane centre
bool  laneOffOk   = false;
float laneAlongMm = 0.0;     // distance along the lane since the last corner
long  laneAlongEnc = 0;
float latYawCmd   = 0.0;     // + = yaw left of laneHeading
float latTarget   = 0.0;
bool  passActive  = false;
bool  plannerEnabled = false;  // DRIVE / FINAL / BACKOFF - sightings mid-turn are in the wrong lane frame

int     backoffWanted = -1;    // track index the REACH plan wants a BACKOFF for, -1 = none
float   backoffPlanMm = 0.0;   // ... and how far
int     reachTrack    = -1;    // track the failing-revolution count belongs to
uint8_t reachBad      = 0;     // consecutive failing revolutions
float   reachClearMm  = 0.0;   // last simulated clearance, for the log
bool    reachPause    = false; // stopped while a failing REACH verdict is confirmed
unsigned long reachPauseMs = 0;

// ---- odometry (v9) ----
float gSpeedMmps   = 0.0;      // measured forward speed, filtered (REACH simulation)
long  odoLastEnc   = 0;
unsigned long odoLastMs = 0;
float odoSpeedAcc  = 0.0;
bool  overdueLogged = false;   // "corner overdue" printed once per straight

// once per loop, and before every encoder zero (the speed must not see the jump)
void odoUpdate() {
  long e = readEncoder();
  float mm = (e - odoLastEnc) / TICKS_PER_MM;
  odoLastEnc = e;
  odoSpeedAcc  += mm;
  unsigned long now = millis();
  if (now - odoLastMs >= 50) {                               // 20 Hz speed estimate
    float v = odoSpeedAcc * 1000.0f / (float)(now - odoLastMs);
    gSpeedMmps += 0.5f * (v - gSpeedMmps);
    odoSpeedAcc = 0.0f; odoLastMs = now;
  }
}

void zeroEncoder() { odoUpdate(); TIM5->CNT = 0; odoLastEnc = 0; }

// Rule 9.19 - the ONLY place the colour -> side rule is written.
bool passRight(int color) { return color == VIS_RED; }

void clearTracks() { for (int i = 0; i < MAX_TRACKS; i++) tracks[i].used = false; passActive = false;
                     backoffWanted = -1; reachTrack = -1; reachBad = 0; }

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
  tracks[slot].hug = false;  tracks[slot].passed = false;
}

// ---- REACH simulation (v9) ----
// Half-width of the band the body sweeps toward one side at yaw psi (rad).
float sweptHalfW(float psi) { return CAR_HALF_W_MM * cosf(psi) + CAR_HALF_LEN_MM * fabsf(sinf(psi)); }

// The aim distance the planner uses with a sign `togo` mm ahead (< 0 = passed).
float passAimMm(float togo) {
  return (togo > PASS_LEAD_MM) ? fmaxf(togo - PASS_LEAD_MM, PASS_AIM_MIN_MM) : HOLD_AIM_MM;
}

// Along-lane half-length of "the body is beside the sign".
float alongsideMm() { return CAR_HALF_LEN_MM + PILLAR_HALF_MM; }

// Drive the real steering loop forward, in the lane frame, toward lateral
// target `tgt` for a sign `rel` mm ahead at lateral `plat`, passed on its
// right if `pr`. Start: lateral y0 (mm, + left), yaw psi0Deg (+ left of the
// lane), servo offset d0 (deg from straight, + = steering left), speed v.
// Returns the smallest clearance (mm) between the swept outline and the
// sign's face + REACH_SAFETY_MM while the body is beside it. < 0 = contact.
const float REACH_SIM_STEP_MM = 10.0f;
const float IMU_HZ            = 100.0f;    // SERVO_SLEW is per IMU update

// holdS / holdTgt: for the first holdS mm the car is still bound to the sign
// before this one (the planner aims at holdTgt until that sign is PASS_HOLD
// behind).
float reachSim(float y0, float psi0Deg, float d0, float tgt, float rel, float plat, bool pr, float v,
               float holdS, float holdTgt) {
  const float ds   = REACH_SIM_STEP_MM;
  const float slew = SERVO_SLEW * IMU_HZ * ds / fmaxf(v, REACH_SPEED_MMPS);
  const float travL = SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT;     // + offsets
  const float travR = SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT;    // - offsets
  const float win  = alongsideMm();
  const float line = pr ? plat - PILLAR_HALF_MM - REACH_SAFETY_MM
                        : plat + PILLAR_HALF_MM + REACH_SAFETY_MM;
  float y = y0, psi = psi0Deg * DEG_TO_RAD, d = d0, worst = 1e9f;
  for (float s = 0.0f; s <= rel + win; s += ds) {
    float togo = rel - s;
    float ymax = (fabsf(togo) <= win) ? fminf(ALONGSIDE_YAW_MAX, PASS_YAW_MAX) : PASS_YAW_MAX;
    bool  held = s < holdS;
    float aimT = held ? holdTgt : tgt;
    float aimD = held ? passAimMm(holdS - PASS_HOLD_MM - s) : passAimMm(togo);
    float cmd  = constrain(atan2f(aimT - y, aimD) / DEG_TO_RAD, -ymax, ymax);
    float want = HEAD_KP * (cmd - psi / DEG_TO_RAD);            // servo deg, + = left
    d += constrain(want - d, -slew, slew);
    d  = constrain(d, -travR, travL);
    float k = (d >= 0.0f ? d / travL : d / travR) / TURN_RADIUS_MM;   // 1/mm, + = left
    psi += k * ds;
    y   += ds * sinf(psi);
    if (fabsf(togo) <= win) {
      float m = pr ? line - (y + sweptHalfW(psi)) : (y - sweptHalfW(psi)) - line;
      if (m < worst) worst = m;
    }
  }
  return worst;
}

// The two lateral targets for passing track t alone: the middle of the gap
// between its face and the wall on its passing side (clamped to the rule
// line), and the rule line itself (PASS_CLEAR from its centre).
void passTargets(const PillarTrack &t, float &gap, float &bound) {
  bool pr = passRight(t.color);
  bound = pr ? t.lat - PASS_CLEAR_MM : t.lat + PASS_CLEAR_MM;
  float mid = pr ? 0.5f * ((t.lat - PILLAR_HALF_MM) + wallLatRight())
                 : 0.5f * ((t.lat + PILLAR_HALF_MM) + wallLatLeft());
  gap = GAP_CENTRE_W * mid;
  gap = pr ? fminf(gap, bound) : fmaxf(gap, bound);
  gap   = constrain(gap,   -LANE_LIMIT_MM, LANE_LIMIT_MM);
  bound = constrain(bound, -LANE_LIMIT_MM, LANE_LIMIT_MM);
}

// Clearance for track t from the car's state now, or after reversing
// `back` mm straight along the lane (heading hold -> yaw and servo end ~0).
// The confirmed sign just before t (in play, or passed and remembered) holds
// the car on its line until it is PASS_HOLD behind.
float reachFrom(const PillarTrack &t, float tgt, float back) {
  float rel  = t.along - laneAlongMm + back;
  int prev = -1;
  for (int i = 0; i < MAX_TRACKS; i++)
    if (tracks[i].used && tracks[i].hits >= TRACK_CONFIRM && tracks[i].along < t.along &&
        (prev < 0 || tracks[i].along > tracks[prev].along)) prev = i;
  float holdS = 0.0f, holdTgt = 0.0f;
  if (prev >= 0) {
    holdS = tracks[prev].along + PASS_HOLD_MM - laneAlongMm + back;
    float g; passTargets(tracks[prev], g, holdTgt);
    if (!tracks[prev].hug) holdTgt = g;
  }
  float psi  = wrapDeg(gHeading - laneHeading);
  float y    = laneOffMm;
  float d0   = SERVO_TRUE_STRAIGHT - lastServoCmd;
  float v    = gSpeedMmps;
  if (back > 0.0f) { y -= 0.5f * back * sinf(psi * DEG_TO_RAD); psi = 0.0f; d0 = 0.0f; v = 0.0f; }
  return reachSim(y, psi, d0, tgt, rel, t.lat, passRight(t.color), v, holdS, holdTgt);
}

// Reversing blind: air between the rear corner and the wall it swings toward.
// A car yawed psi (+ = nose left) backs away to the RIGHT (and the other way
// round), its rear corner already HALF_LEN sin|psi| out that way.
float rearWallGap() {
  float psi = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
  float reach = CAR_HALF_LEN_MM * fabsf(sinf(psi)) + CAR_HALF_W_MM * cosf(psi);
  return (psi >= 0.0f) ? (laneOffMm - reach) - wallLatRight()
                       : wallLatLeft() - (laneOffMm + reach);
}

// May a BACKOFF start here? The reverse heading hold straightens the car at
// up to full lock, so the rear keeps drifting toward that wall for about
// TURN_RADIUS (1 - cos psi) and then backs straight; the rear corner itself
// swings away as the nose comes round. 0 = no, the rear would reach the wall.
float backoffWallLimit() {
  float psi = wrapDeg(gHeading - laneHeading) * DEG_TO_RAD;
  float drift = TURN_RADIUS_MM * (1.0f - cosf(psi));
  return (rearWallGap() - drift >= BACKOFF_WALL_MM) ? 1e9f : 0.0f;
}

// Shortest reverse (BACKOFF_MIN_MM .. avail, 20 mm steps) after which target
// tgt clears by BACKOFF_MARGIN_MM; -1 = none within avail.
float planBackoff(const PillarTrack &t, float tgt, float avail) {
  for (float b = BACKOFF_MIN_MM; b <= avail + 0.5f; b += 20.0f)
    if (reachFrom(t, tgt, b) >= BACKOFF_MARGIN_MM) return b;
  return -1.0f;
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

const char *colName(int c) { return c == VIS_RED ? "RED" : (c == VIS_GREEN ? "GREEN" : "none"); }

// Once per new LiDAR revolution, for the nearest sign ahead (see REACH above).
void reachDecide(int k, float rel) {
  PillarTrack &t = tracks[k];
  if (k != reachTrack) { reachTrack = k; reachBad = 0; }
  if (rel > REACH_DECIDE_FAR_MM || rel < REACH_DECIDE_NEAR_MM || t.hits < REACH_MIN_HITS) {
    reachBad = 0; reachPause = false;
    return;
  }
  if (t.hug && t.backedMm >= BACKOFF_MAX_MM) return;     // committed: nothing left to decide
  float gap, bound;
  passTargets(t, gap, bound);
  float clr = reachFrom(t, t.hug ? bound : gap, 0.0f);
  reachClearMm = clr;
  if (clr >= 0.0f) { reachBad = 0; reachPause = false; return; }
  if (++reachBad < REACH_CONFIRM) {
    if (reachBad == 1 && REACH_PAUSE_MS > 0) {
      reachPause = true; reachPauseMs = millis();
      Serial.print(F("# REACH pause, clr=")); Serial.println((int)clr);
    }
    return;
  }
  reachBad = 0;
  reachPause = false;

  Serial.print(F("# REACH ")); Serial.print(colName(t.color));
  Serial.print(F(" rel=")); Serial.print((int)rel);
  Serial.print(F(" lat=")); Serial.print((int)t.lat);
  Serial.print(F(" off=")); Serial.print((int)laneOffMm);
  Serial.print(F(" yaw=")); Serial.print(wrapDeg(gHeading - laneHeading), 1);
  Serial.print(F(" v=")); Serial.print((int)gSpeedMmps);
  Serial.print(F(" clr=")); Serial.print((int)clr);

  // cheapest first: the rule line instead of the gap centre
  if (!t.hug) {
    float cb = reachFrom(t, bound, 0.0f);
    if (cb >= 0.0f) {
      t.hug = true;
      Serial.print(F(" -> HUG (line clr=")); Serial.print((int)cb); Serial.println(')');
      return;
    }
  }
  float avail = fminf(BACKOFF_MAX_MM - t.backedMm, laneAlongMm + BACKOFF_BEHIND_MM);
  float wallLim = backoffWallLimit();
  if (wallLim < avail) {                       // yawed near a wall: the rear would hit it
    avail = wallLim;
    Serial.print(F(" [rear too close to the wall to reverse]"));
  }
  if (!BACKOFF_ENABLE || avail < BACKOFF_MIN_MM) {
    t.hug = true;
    t.backedMm = BACKOFF_MAX_MM;                  // never asks again
    Serial.println(F(" -> COMMIT (no reverse budget)"));
    return;
  }
  bool  hugAfter = false;
  float back = planBackoff(t, gap, avail);
  if (back < 0.0f) { back = planBackoff(t, bound, avail); hugAfter = true; }
  if (back < 0.0f) { back = avail;                        hugAfter = true; }
  t.hug = hugAfter;
  backoffWanted = k;
  backoffPlanMm = back;
  Serial.print(F(" -> BACKOFF ")); Serial.print((int)back);
  Serial.println(hugAfter ? F(" mm, then the line") : F(" mm, then the gap"));
}

void updatePlanner() {
  // lane distance: encoder projected onto the lane direction (negative when reversing)
  long enc = readEncoder();
  float dmm = (enc - laneAlongEnc) / TICKS_PER_MM;
  laneAlongEnc = enc;
  laneAlongMm += dmm * cosf(wrapDeg(gHeading - laneHeading) * DEG_TO_RAD);

  if (!lidarNewFrame) return;
  laneOffOk = laneOffset(laneOffMm);
  if (!plannerEnabled) { latYawCmd = 0.0f; passActive = false; backoffWanted = -1; return; }
  addSighting(sign1);

  // tracks in play: everything still within PASS_HOLD behind, up to and
  // including the NEAREST sign ahead (farther ones wait their turn)
  float nearestAhead = 1e9; int nearestIdx = -1;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!tracks[i].used) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel < -TRACK_KEEP_BEHIND_MM) { tracks[i].used = false; continue; }   // long gone
    if (rel < -PASS_HOLD_MM) {                                               // passed: out of play,
      if (!tracks[i].passed && tracks[i].hits >= TRACK_CONFIRM)              //   but remembered
        lastPassedColor = tracks[i].color;
      tracks[i].passed = true;
      continue;
    }
    tracks[i].passed = false;                                                // (back in play after a reverse)
    if (tracks[i].hits < TRACK_CONFIRM) {                                    // not trusted yet
      if (laneAlongMm - tracks[i].lastSeen > TRACK_FORGET_MM) tracks[i].used = false;
      continue;
    }
    if (rel > 0 && rel < nearestAhead) { nearestAhead = rel; nearestIdx = i; }
  }

  float lo = -LANE_LIMIT_MM, hi = LANE_LIMIT_MM;
  float urgentRel = 1e9, urgentBound = 0; int urgentIdx = -1; bool any = false, beside = false;
  for (int i = 0; i < MAX_TRACKS; i++) {
    if (!trackConfirmed(i) || tracks[i].passed) continue;
    float rel = tracks[i].along - laneAlongMm;
    if (rel > nearestAhead + 1.0f) continue;
    any = true;
    if (fabs(rel) <= alongsideMm()) beside = true;       // (a passed sign no longer caps the yaw:
                                                         //   host sim, the late swing hit the next one)
    float bound;
    if (passRight(tracks[i].color)) { bound = tracks[i].lat - PASS_CLEAR_MM; if (bound < hi) hi = bound; }
    else                            { bound = tracks[i].lat + PASS_CLEAR_MM; if (bound > lo) lo = bound; }
    if (rel < urgentRel) { urgentRel = rel; urgentBound = bound; urgentIdx = i; }
  }
  passActive = any;

  // Default target. No sign in play: lane centre, or the planned corner-exit
  // offset for the first stretch after a corner. A sign in play: the middle of
  // the gap between the most urgent sign's face and the wall on its passing
  // side - or its rule line, once REACH has said the gap is out of reach.
  float target = 0.0f;
  if (!any) {
    if (cornerCount > 0 && laneAlongMm < POST_CORNER_BOOST_MM) target = cornerExitCmd;
  } else {
    float gap, bound;
    passTargets(tracks[urgentIdx], gap, bound);
    target = tracks[urgentIdx].hug ? bound : gap;
  }
  if (lo > hi) target = urgentBound;                     // conflict: most urgent sign wins
  else         target = constrain(target, lo, hi);

  float aimMm = CENTRE_AIM_MM;
  if (any) {
    // aim at the pass point PASS_LEAD before the nearest sign: the swerve
    // sharpens as it nears and arrives in time. Alongside / past: gentle hold.
    aimMm = (nearestAhead < 1e8) ? passAimMm(nearestAhead) : HOLD_AIM_MM;

    // REACH: judged on new revolutions only, while DRIVE / FINAL (not while
    // reversing), for the nearest sign ahead (the one being passed, if any,
    // is part of the simulation - see reachFrom).
    bool driving = currentState == STATE_DRIVE_TO_CORNER || currentState == STATE_FINAL_STRAIGHT;
    if (lidarNewRev && driving && laneOffOk && nearestIdx >= 0)
      reachDecide(nearestIdx, nearestAhead);
    else if (nearestIdx < 0)
      reachBad = 0;
  } else {
    backoffWanted = -1;
    reachBad = 0;
  }

  latTarget = constrain(target, -LANE_LIMIT_MM, LANE_LIMIT_MM);

  if (!laneOffOk) { latYawCmd = 0.0f; return; }          // no walls: just hold heading
  // Centring is gentle (CENTRE_YAW_MAX) so the car does not weave; right
  // after a corner it gets POST_CORNER_YAW_MAX, with a sign in play PASS_YAW_MAX,
  // and with the body beside a sign ALONGSIDE_YAW_MAX (the swept corner).
  float ymax = passActive ? PASS_YAW_MAX
             : (laneAlongMm < POST_CORNER_BOOST_MM ? POST_CORNER_YAW_MAX : CENTRE_YAW_MAX);
  if (beside) ymax = fminf(ymax, ALONGSIDE_YAW_MAX);
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

  odoUpdate();
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
// (the gains HEAD_KP ... HEAD_KD are declared with the calibration at the
// top: the REACH simulation uses them too)

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
// OVERLAY: BACKOFF  (reverse the distance the REACH plan asked for)
// ============================================================
int   backoffTrack = -1;
float backoffStartMm = 0.0;          // the track's backedMm when this backoff began
float backoffGoalMm  = 0.0;          // this backoff's planned distance
long  backoffLastTicks = 0;          // stuck watch: last encoder reading that moved
unsigned long backoffLastMoveMs = 0;

void backoffStep() {
  if (!entered) {
    entered = true;
    setMotorSpeed(0);
    resetHeadingPid();
    overlayBaseTicks = readEncoder();
    backoffLastTicks = overlayBaseTicks;
    backoffLastMoveMs = millis();
    backoffStartMm = (backoffTrack >= 0) ? tracks[backoffTrack].backedMm : 0.0f;
    backoffGoalMm  = backoffPlanMm;
    Serial.print(F("# BACKOFF ")); Serial.print(backoffTrack >= 0 ? colName(tracks[backoffTrack].color) : "?");
    Serial.print(F(" plan ")); Serial.print((int)backoffGoalMm);
    Serial.print(F(" mm, clr ")); Serial.println((int)reachClearMm);
  }
  plannerEnabled = true;             // keeps the tracks moving with the car (no decisions)
  bool gone = (backoffTrack < 0) || !tracks[backoffTrack].used;
  long now = readEncoder();
  float backed = absEnc(now - overlayBaseTicks) / TICKS_PER_MM;
  if (!gone) tracks[backoffTrack].backedMm = backoffStartMm + backed;
  if (absEnc(now - backoffLastTicks) > (long)(10.0f * TICKS_PER_MM)) {
    backoffLastTicks = now; backoffLastMoveMs = millis();
  }

  bool done   = backed >= backoffGoalMm;
  bool capped = !gone && (tracks[backoffTrack].backedMm >= BACKOFF_MAX_MM ||
                          laneAlongMm <= -BACKOFF_BEHIND_MM);
  bool stuck  = millis() - backoffLastMoveMs > BACKOFF_TIMEOUT_MS;
  bool wall   = laneOffOk && rearWallGap() < 0.5f * BACKOFF_WALL_MM;   // hard stop
  if (gone || done || capped || stuck || wall) {
    if ((capped || stuck || wall) && !gone) tracks[backoffTrack].backedMm = BACKOFF_MAX_MM;   // commit, never re-ask
    Serial.print(done ? F("# backoff done after ") : stuck ? F("# backoff STUCK after ")
                      : wall ? F("# backoff stopped, rear near the wall, after ")
                      : capped ? F("# backoff capped, commit after ") : F("# backoff: sign lost after "));
    Serial.print((int)backed); Serial.println(F(" mm"));
    reachBad = 0;
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
  overdueLogged = false;
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
// Open = the side beam reads past SIDE_OPEN_MM (v9: the beam alone, no cone
// test). `candidate` = this side may open this revolution (before the
// direction is known only the side with the longer beam may). `overdue` =
// the odometry says the corner is past due: one revolution is enough and no
// wall history is needed.
bool sideEvidence(uint8_t s, uint16_t mm, bool candidate, bool overdue) {
  bool open = candidate && mm > SIDE_OPEN_MM;
  if (open) {
    if ((overdue || sideWallRevs[s] >= SIDE_WALL_REVS) && sideOpenRevs[s] < 255) sideOpenRevs[s]++;
  } else {
    sideOpenRevs[s] = 0;
    if (mm <= SIDE_OPEN_MM && sideWallRevs[s] < 255) sideWallRevs[s]++;
  }
  return sideOpenRevs[s] >= (overdue ? 1 : SIDE_OPEN_REVS);
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

  // odometry window - after the first corner only (the start spot is unknown)
  bool gated   = cornerCount > 0;
  bool tooSoon = gated && runMm < CORNER_MIN_RUN_MM;
  bool overdue = gated && runMm > CORNER_FALLBACK_RUN_MM;
  if (overdue && !overdueLogged) {
    overdueLogged = true;
    Serial.print(F("# corner overdue, run=")); Serial.println((int)runMm);
  }

  // candidates: after the lock the inner side only; before it the side with
  // the LONGER beam (the outer wall never ends, so the longer one is the
  // opening). Too soon: nobody - but the wall history still counts.
  bool longL = lidarL >= lidarR;
  bool candL = !tooSoon && (dirLocked ? !clockwiseMode : longL);
  bool candR = !tooSoon && (dirLocked ?  clockwiseMode : !longL);
  bool openL = sideEvidence(0, lidarL, candL, overdue);
  bool openR = sideEvidence(1, lidarR, candR, overdue);
  if (!openL && !openR) return;
  if (!dirLocked) {
    dirLocked     = true;
    clockwiseMode = openR;                                   // outer wall never ends: open side = inner
    Serial.print(clockwiseMode ? F("# LOCKED CW (right side opened) L=") : F("# LOCKED CCW (left side opened) L="));
    Serial.print(lidarL); Serial.print(F(" R=")); Serial.println(lidarR);
  }
  Serial.print(F("# turn: side open ")); Serial.print(turnSideMm());
  Serial.print(F(" run=")); Serial.print((int)runMm);
  Serial.println(overdue ? F(" (overdue rule)") : F(""));
  overdueLogged = false;

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
       float    TURN_REV_INNER_MM  = 150.0;  // v9: PLAN_REVERSE approach steers to this far on the
                                             //   inner side of the old lane (0 = straight, as v7)
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
      if (sign1.valid && sign1.x > 150.0f && (clockwiseMode ? sign1.y < -300.0f : sign1.y > 300.0f))
        nextStraightColor = sign1.color;
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
      if (turnPlan == PLAN_REVERSE && TURN_REV_INNER_MM > 0.0f && laneOffOk) {
        // v9: the reverse arc swings the car ~one turn radius toward the old
        // lane's OUTER wall - start it from the inner half (host sim: a
        // reverse corner begun right of centre hit that wall 11 times in 60 runs)
        float inner = clockwiseMode ? -1.0f : 1.0f;            // + lane offset = left
        float yaw = constrain(atan2f(inner * TURN_REV_INNER_MM - laneOffMm, CENTRE_AIM_MM) / DEG_TO_RAD,
                              -CENTRE_YAW_MAX, CENTRE_YAW_MAX);
        updateDriveSteer(wrapDeg(turnOldLane + yaw), false);
      } else {
        updateDriveSteer(turnOldLane, false);   // straight on the old lane heading
      }
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

const uint16_t PARAM_VERSION        = 8;     // bump when ids are added/removed
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
  PF(CORNER_MIN_RUN_MM,     G_TURN,    0,  5000),
  PF(CORNER_FALLBACK_RUN_MM,G_TURN,    0,  8000),
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
  PF(TURN_REV_INNER_MM,     G_TURN,    0,   400),
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
  PF(CAR_HALF_LEN_MM,       G_PLAN,   20,   300),
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
  PF(ALONGSIDE_YAW_MAX,     G_PASS,    2,    89),
  PF(REACH_SAFETY_MM,       G_PASS,  -50,   200),
  PF(REACH_DECIDE_NEAR_MM,  G_PASS,    0,  1500),
  PF(REACH_DECIDE_FAR_MM,   G_PASS,  100,  3000),
  PC(REACH_MIN_HITS,        G_PASS,    1,    20),
  PC(REACH_CONFIRM,         G_PASS,    1,    20),
  PF(REACH_SPEED_MMPS,      G_PASS,   50,  3000),
  PL(REACH_PAUSE_MS,        G_PASS,    0,  3000),
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
  PL(BACKOFF_TIMEOUT_MS,    G_BACK,  200, 10000),
  PF(BACKOFF_WALL_MM,       G_BACK,    0,   300),

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

  // REACH pause: stopped, wheels where they are, until the verdict is in (or
  // REACH_PAUSE_MS runs out). DRIVE / FINAL steps are skipped meanwhile.
  static bool paused = false;
  if (reachPause && (!moving() || millis() - reachPauseMs > REACH_PAUSE_MS)) reachPause = false;
  if (reachPause) {
    if (!paused) { paused = true; setMotorSpeed(0); }
    return;
  }
  if (paused) {
    paused = false;
    resetHeadingPid();
    prevServoCmd = lastServoCmd;
    if (moving()) setMotorSpeed(DRIVE_PWM);
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
