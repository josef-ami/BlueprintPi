// ============================================================================
// ObstacleRound.cpp - WRO Future Engineers obstacle round, EXECUTOR firmware.
//
// This is the previous ObstacleRound.cpp with the state machine and the lane
// planner REMOVED. Both moved to the Pi (control/fsm.py, control/planner.py),
// where the camera and the LiDAR already are. What is left here is everything
// a 50 Hz link cannot do:
//
//   heading PID      closed on the IMU at its own rate (~100 Hz), with the
//                    2.5 deg/update servo slew that stops the tyres scrubbing
//   the 90 deg arc   terminated on an IMU reading, not on a Pi frame
//   wall panic       a reflex whose whole purpose is to be faster than the
//                    crash; routing it through the link would add a frame of
//                    latency to the one thing that must not have any
//   odometry         TIM5 encoder, free-running, reported up
//   colour           TCS34725 classification, reported up (the DEBOUNCE and
//                    the gate logic are the Pi's now)
//   servo + motor    the only code that knows microseconds and duty
//
// Everything the old file did between those - tracks, pass planning, corner
// triggers, levelling, the 3-point sequencing, the corner-exit table - is
// gone from here and lives on the Pi. The parameter table shrank from 91
// entries to 20 for the same reason: this file no longer runs the logic those
// numbers described.
//
// WIRE  (full spec: docs/PI_STM32_PROTOCOL.md)
//   Pi -> us   DRIVE  19 bytes, sync AA 55, 50 Hz
//   us -> Pi   TELEM  21 bytes, sync 55 AA, 50 Hz
//   both       parameter lines, ASCII:  N / ?P / ?V / C  <->  !V !P !p !E !C
//   us -> Pi   '#' log lines, ASCII
//
// 0xAA is not a valid ASCII byte, so a log line or a parameter line can never
// contain a sync word and one port carries all four cleanly.
//
// NEW SINCE THE FSM MOVED: a link-loss motor cut. The old firmware had none
// deliberately - a car whose Pi died carried on under its own state machine.
// It has no state machine now, so silence means stop.
//
// HOST BUILD: firmware/sim/build.sh compiles this file unmodified for a PC,
// and tests/test_firmware_sim.py drives it over the real wire - including a
// full closed-loop lap with the Pi's FSM. That checks the logic and the
// protocol; it cannot check the hardware below.
//
// BENCH-VERIFIED (unchanged from the previous firmware)
//   motor    PA2 forward, PA3 reverse
//   encoder  TIM5 PA0/PA1, negated so forward counts up
//   IMU      BNO08x SPI1 ~100 Hz, clockwise = negative yaw
//   colour   TCS34725 CH4, white pR 47 / orange pR 69 / blue pB 27
//   servo    500-2500 us, straight 76.5, left stop 20, right stop 140
//            turning radius at full lock: left 27 cm, right 25 cm
// ============================================================================

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
// HARDWARE PINS & OBJECTS
// ============================================================
const int MOT_RPWM_PIN = PA2;     // forward  (TIM2_CH3; TIM5 is the encoder)
const int MOT_LPWM_PIN = PA3;     // reverse  (TIM2_CH4)
const int SERVO_PIN    = PA8;

const int IMU_CS_PIN  = PA4;
const int IMU_INT_PIN = PB0;
const int IMU_RST_PIN = PB1;

const int LED1_PIN = PB12;   // slow blink = no link; solid = link up and
                             // enabled; fast blink = link stale
const int LED2_PIN = PB13;   // lit while ORANGE is under the sensor
const int LED3_PIN = PB14;   // lit while BLUE is under the sensor

#define I2C_SCL     PB6
#define I2C_SDA     PB7
#define TCA_RST_PIN PB8
#define TCA_ADDR    0x70
#define TCS_CH      4

// ---- calibration (all of it tunable from the Pi) ----
       float TICKS_PER_CM        = 14.853;
const  int   SERVO_MIN_PULSE_US  = 500;
const  int   SERVO_MAX_PULSE_US  = 2500;
       float SERVO_TRUE_STRAIGHT = 76.5;
       float SERVO_MAX_LEFT      = 20.0;   // left hard stop  (BELOW straight)
       float SERVO_MAX_RIGHT     = 140.0;  // right hard stop (ABOVE straight)
       float STEER_LOCK_DEG      = 35.0;   // road-wheel angle at full lock,
                                           // measured at the knuckles. Maps
                                           // the Pi's DIRECT command (in
                                           // road-wheel degrees) onto this
                                           // car's servo travel.
       float IMU_YAW_SIGN        = 1.0;    // clockwise reads negative

SPIClass SPI_IMU(PA7, PA6, PA5);  // MOSI, MISO, SCLK
Servo steeringServo;
BNO08x myIMU;
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS,
                                          TCS34725_GAIN_16X);

bool  tcsOk = false;
float initialYawOffset = 0.0;

// ============================================================
// WIRE
// ============================================================
const uint8_t DRIVE_LEN     = 19;
const uint8_t DRIVE_PAYLOAD = 16;      // bytes 2..17
const uint8_t TELEM_LEN     = 21;
const uint8_t TELEM_PAYLOAD = 18;      // bytes 2..19

// DRIVE flags
const uint8_t F_ENABLE      = 0x01;
const uint8_t F_LIDAR_OK    = 0x02;
const uint8_t F_CAM_OK      = 0x04;
const uint8_t F_PILLAR_SEEN = 0x08;
const uint8_t F_MODE_MASK   = 0x30;
const uint8_t F_MODE_SHIFT  = 4;
const uint8_t F_REVERSE     = 0x40;

// modes
const uint8_t MODE_STOP = 0, MODE_HEADING = 1, MODE_DIRECT = 2, MODE_ARC = 3;

// TELEM status
const uint8_t S_ENABLED        = 0x01;
const uint8_t S_IMU_OK         = 0x02;
const uint8_t S_COLOUR_OK      = 0x04;
const uint8_t S_ARC_DONE       = 0x08;
const uint8_t S_RECOVERING     = 0x10;
const uint8_t S_RECOVER_CAPPED = 0x20;
const uint8_t S_LINK_STALE     = 0x40;
const uint8_t S_PARAMS_PUSHED  = 0x80;

const uint8_t CMD_NONE = 0, CMD_REBOOT = 1;
const uint16_t RANGE_NONE = 0xFFFF;
const uint16_t LIDAR_FAR  = 9999;

// ---- the newest DRIVE frame ----
// dLeft / dRight / dRev / dSeq are decoded but not acted on here: only the
// FRONT range feeds the panic reflex. They are kept because the decode is the
// layout's documentation, and because a diagnostic that needs them should not
// have to change the parser.
uint8_t  dSeq = 0, dFlags = 0, dSpeed = 0, dArcLock = 70;
int16_t  dHeading = 0, dSteer = 0;
uint16_t dLeft = LIDAR_FAR, dFront = LIDAR_FAR, dRight = LIDAR_FAR;
uint8_t  dRev = 0;

unsigned long linkLastMs = 0;
bool     linkStale = true;
bool     linkEverUp = false;
uint32_t driveFrames = 0;
uint8_t  cmdRun = 0, cmdLast = CMD_NONE;

       unsigned long LINK_STALE_MS = 200;
const  uint8_t CMD_CONFIRM_FRAMES = 4;   // identical frames before REBOOT acts

inline uint8_t mode() { return (dFlags & F_MODE_MASK) >> F_MODE_SHIFT; }
inline bool enabled()    { return dFlags & F_ENABLE; }
inline bool lidarOk()    { return dFlags & F_LIDAR_OK; }
inline bool pillarSeen() { return dFlags & F_PILLAR_SEEN; }

// A beam with no return must read FAR, never near.
inline uint16_t sane(uint16_t v) { return (v == RANGE_NONE || v == 0) ? LIDAR_FAR : v; }
inline uint16_t frontMm() { return sane(dFront); }

static_assert(DRIVE_LEN == 2 + DRIVE_PAYLOAD + 1, "DRIVE layout");
static_assert(TELEM_LEN == 2 + TELEM_PAYLOAD + 1, "TELEM layout");

uint8_t xor8(const uint8_t *p, uint8_t n) {
  uint8_t x = 0;
  for (uint8_t i = 0; i < n; i++) x ^= p[i];
  return x;
}

// ============================================================
// CACHED SENSORS
// ============================================================
// The values go on the wire as a byte, and the Arduino IDE's prototype
// generator inserts its auto-prototypes ABOVE this line when the file is
// built as a .ino - so a function returning `BlockColor` would be declared
// before the type exists. Plain uint8_t constants dodge that entirely and
// match what TELEM carries anyway.
enum BlockColor : uint8_t { COLOR_NONE = 0, COLOR_ORANGE = 1, COLOR_BLUE = 2 };

bool          gImuFresh = false;
float         gHeading  = 0.0;
float         gYawRate  = 0.0;
float         gPrevH    = 0.0;
unsigned long gPrevHT   = 0;
bool          gImuSeen  = false;
uint8_t       gRawColor = COLOR_NONE;

float wrapDeg(float a) {
  while (a > 180.0)  a -= 360.0;
  while (a < -180.0) a += 360.0;
  return a;
}

float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================
// ACTUATORS
// ============================================================
void setMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0)      { analogWrite(MOT_RPWM_PIN, speed); analogWrite(MOT_LPWM_PIN, 0); }
  else if (speed < 0) { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, -speed); }
  else                { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, 0); }
}

float lastServoCmd = 76.5;

void setServoAngle(float angleDeg) {
  angleDeg = clampf(angleDeg, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  lastServoCmd = angleDeg;
  int pulse = (int)((angleDeg / 180.0) * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US))
              + SERVO_MIN_PULSE_US;
  steeringServo.writeMicroseconds(pulse);
}

// Road-wheel degrees (+ = LEFT, the Pi's convention) -> this car's servo
// angle. The Pi works in physical units and never learns the trim or the
// stops; this is the only place the two meet.
float servoForSteer(float steerDeg) {
  float frac = clampf(steerDeg / STEER_LOCK_DEG, -1.0f, 1.0f);
  float travel = (frac > 0) ? (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT)
                            : (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT);
  return SERVO_TRUE_STRAIGHT - frac * travel;
}

// TIM5 encoder, negated so driving forward counts up. NEVER zeroed: the Pi
// keeps its own baselines, because asking for a zero over the link is a race.
long readEncoder() { return -(int32_t)TIM5->CNT; }
long absEnc(long v) { return v < 0 ? -v : v; }

// ============================================================
// IMU / COLOUR
// ============================================================
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
    if (myIMU.getSensorEvent() &&
        myIMU.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      initialYawOffset = readYaw();
      Serial.print(F("# zero yaw ")); Serial.println(initialYawOffset);
      return;
    }
    delay(10);
  }
  Serial.println(F("# ERROR no IMU event to zero"));
}

void tcaselect(uint8_t channel) {
  if (channel > 7) return;
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void resetTCA() {
  pinMode(TCA_RST_PIN, OUTPUT);
  digitalWrite(TCA_RST_PIN, LOW);  delay(10);
  digitalWrite(TCA_RST_PIN, HIGH); delay(10);
}

void readColor(uint16_t &r, uint16_t &g, uint16_t &b, uint16_t &c) {
  if (!tcsOk) { r = g = b = c = 0; return; }
  tcaselect(TCS_CH);
  Wire.beginTransmission(0x29);
  Wire.write(0x80 | 0x20 | 0x14);      // command | auto-increment | CDATAL
  Wire.endTransmission();
  Wire.requestFrom((uint8_t)0x29, (uint8_t)8);
  if (Wire.available() < 8) { r = g = b = c = 0; return; }
  c  = (uint16_t)Wire.read();  c |= (uint16_t)Wire.read() << 8;
  r  = (uint16_t)Wire.read();  r |= (uint16_t)Wire.read() << 8;
  g  = (uint16_t)Wire.read();  g |= (uint16_t)Wire.read() << 8;
  b  = (uint16_t)Wire.read();  b |= (uint16_t)Wire.read() << 8;
}

// The classification stays here because it is a sensor reading. The DEBOUNCE
// and the "is this the corner gate" decision are the Pi's now.
uint8_t classifyColor() {
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

// ============================================================
// HEADING PID - the reason this file still exists
// ============================================================
       float HEAD_KP        = 2.0;
       float HEAD_KI        = 0.0;
       float HEAD_KD        = 0.0;
       float YAW_FILT_ALPHA = 0.35;
       float SERVO_SLEW     = 2.5;     // servo deg per IMU update (~100 Hz)
       float INTEGRAL_CLAMP = 300.0;

unsigned long pidPrevTime  = 0;
float         pidIntegral  = 0.0;
float         yawFilt      = 0.0;
float         prevServoCmd = 76.5;

void resetHeadingPid() {
  pidPrevTime  = millis();
  pidIntegral  = 0.0;
  yawFilt      = 0.0;
  prevServoCmd = SERVO_TRUE_STRAIGHT;
}

void holdHeading(float target) {
  if (!gImuFresh) return;
  unsigned long now = millis();
  yawFilt += YAW_FILT_ALPHA * (gYawRate - yawFilt);
  float dt = (now - pidPrevTime) / 1000.0;
  if (dt <= 0.0) dt = 0.001;

  float error = wrapDeg(target - gHeading);
  pidIntegral += error * dt;
  pidIntegral  = clampf(pidIntegral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
  float correction = HEAD_KP * error + HEAD_KI * pidIntegral - HEAD_KD * yawFilt;

  float want = SERVO_TRUE_STRAIGHT - correction;        // below straight = left
  float dcmd = clampf(want - prevServoCmd, -SERVO_SLEW, SERVO_SLEW);
  float cmd  = clampf(prevServoCmd + dcmd, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  if (cmd <= SERVO_MAX_LEFT || cmd >= SERVO_MAX_RIGHT) pidIntegral -= error * dt;
  setServoAngle(cmd);
  prevServoCmd = cmd;
  pidPrevTime  = now;
}

// ============================================================
// THE ARC - eased 90 deg turn, terminated on the IMU
// ============================================================
       float TURN_KP        = 2.5;
       float TURN_MIN_STEER = 8.0;
       float TURN_STOP_DEG  = 15.0;   // hand back this close to the target:
                                      // the heading PID finishes the last
                                      // degrees while the Pi's planner is
                                      // already live again
bool arcDone = false;

void arcStep(float target, float lockFrac, uint8_t pwm) {
  if (!gImuFresh) return;
  float err = wrapDeg(target - gHeading);
  if (fabs(err) < TURN_STOP_DEG) { arcDone = true; setMotorSpeed(pwm); return; }
  arcDone = false;
  float maxSteer = (err > 0) ? (SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT) * lockFrac
                             : (SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT) * lockFrac;
  float steer = clampf(TURN_KP * fabs(err), TURN_MIN_STEER, maxSteer);
  setServoAngle((err > 0) ? (SERVO_TRUE_STRAIGHT - steer)
                          : (SERVO_TRUE_STRAIGHT + steer));
  setMotorSpeed(pwm);
}

// ============================================================
// WALL PANIC - the one reflex that is still ours
// ============================================================
// The trigger is a distance the Pi sent us and the response is one servo
// write plus one motor write. Routing that through the link would add a frame
// of latency to the only behaviour whose entire point is to be faster than
// the crash.
       uint16_t WALL_PANIC_MM     = 200;
       uint16_t WALL_CLEAR_MM     = 350;
       float    RECOVER_MAX_CM    = 30.0;
       int      RECOVER_MAX_TRIES = 3;

bool recovering    = false;
bool recoverCapped = false;
int  recoverTries  = 0;
long recoverBase   = 0;

void enterRecovery(uint8_t pwm) {
  recovering    = true;
  recoverCapped = false;
  recoverBase   = readEncoder();
  setMotorSpeed(0);
  setServoAngle(2.0 * SERVO_TRUE_STRAIGHT - lastServoCmd);   // mirror the lock
  setMotorSpeed(-(int)pwm);
  Serial.print(F("# RECOVER front=")); Serial.println(frontMm());
}

void recoverStep(uint8_t pwm) {
  bool clear     = lidarOk() && (frontMm() >= WALL_CLEAR_MM);
  bool backedFar = absEnc(readEncoder() - recoverBase)
                   >= (long)(RECOVER_MAX_CM * TICKS_PER_CM);
  if (!clear && !backedFar) { setMotorSpeed(-(int)pwm); return; }

  setMotorSpeed(0);
  recovering = false;
  if (clear) { recoverTries = 0; recoverCapped = false;
               Serial.println(F("# recover clear")); }
  else       { recoverTries++;   recoverCapped = true;
               Serial.print(F("# recover capped, try "));
               Serial.println(recoverTries); }
  resetHeadingPid();
}

// ============================================================
// DRIVE FRAME -> ACTUATORS
// ============================================================
void applyDrive() {
  // Link gone: stop. The old firmware carried on under its own FSM; there is
  // no FSM here any more, so silence means stop.
  if (linkStale || !linkEverUp) {
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    prevServoCmd = SERVO_TRUE_STRAIGHT;
    recovering = false;
    return;
  }

  uint8_t pwm = dSpeed;

  // A Stop wins over the reflex, and is checked BEFORE it. Otherwise a Stop
  // arriving mid-recovery would hand recoverStep() a speed of zero: the car
  // would "reverse" at a standstill, never reach WALL_CLEAR_MM or the
  // distance cap, and sit in recovery ignoring every command that followed.
  if (!enabled() || mode() == MODE_STOP || pwm == 0) {
    if (recovering) Serial.println(F("# recovery abandoned: STOP from Pi"));
    recovering = false;
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    prevServoCmd = SERVO_TRUE_STRAIGHT;
    pidIntegral = 0.0;
    arcDone = false;
    return;
  }

  if (recovering) { recoverStep(pwm); return; }

  // The reflex, armed only while the Pi says it is driving and the ranges can
  // be trusted, and stood down while the camera has a pillar - a short front
  // reading is then that pillar, and the planner is already steering round it.
  if (lidarOk() && !pillarSeen() &&
      recoverTries < RECOVER_MAX_TRIES && frontMm() <= WALL_PANIC_MM) {
    enterRecovery(pwm);
    return;
  }

  int drive = (dFlags & F_REVERSE) ? -(int)pwm : (int)pwm;
  float target = dHeading / 10.0f;

  switch (mode()) {
    case MODE_HEADING:
      arcDone = false;
      holdHeading(target);
      setMotorSpeed(drive);
      break;
    case MODE_DIRECT:
      arcDone = false;
      setServoAngle(servoForSteer(dSteer / 10.0f));
      prevServoCmd = lastServoCmd;     // so a later PID hand-back does not jump
      setMotorSpeed(drive);
      break;
    case MODE_ARC:
      arcStep(target, clampf(dArcLock / 100.0f, 0.2f, 1.0f), pwm);
      break;
    default:
      setMotorSpeed(0);
      break;
  }
}

// ============================================================
// TELEM
// ============================================================
uint8_t telemSeq = 0;
bool    paramsPushed = false;
uint32_t bootId = 0;
const unsigned long TELEM_PERIOD_MS = 20;
unsigned long telemLastMs = 0;

void sendTelem() {
  // Leave room so a TELEM frame can never be half-written into a full CDC
  // buffer and split by a log line.
  if ((uint16_t)Serial.availableForWrite() < TELEM_LEN + 8) return;

  uint8_t status = 0;
  if (enabled() && !linkStale)         status |= S_ENABLED;
  if (gImuSeen)                        status |= S_IMU_OK;
  if (tcsOk)                           status |= S_COLOUR_OK;
  if (arcDone)                         status |= S_ARC_DONE;
  if (recovering)                      status |= S_RECOVERING;
  if (recoverCapped)                   status |= S_RECOVER_CAPPED;
  if (linkStale)                       status |= S_LINK_STALE;
  if (paramsPushed)                    status |= S_PARAMS_PUSHED;

  int16_t  heading = (int16_t)lroundf(clampf(gHeading * 10.0f, -32768, 32767));
  int16_t  yaw     = (int16_t)lroundf(clampf(gYawRate * 10.0f, -32768, 32767));
  int32_t  odo     = (int32_t)readEncoder();
  int16_t  servo   = (int16_t)lroundf(clampf(lastServoCmd * 10.0f, -32768, 32767));
  uint8_t  floorC  = gRawColor;
  uint8_t  tries   = (uint8_t)recoverTries;

  uint8_t buf[TELEM_LEN];
  buf[0] = 0x55; buf[1] = 0xAA;
  uint8_t *p = buf + 2;
  p[0] = telemSeq++;
  p[1] = status;
  memcpy(p + 2,  &heading, 2);
  memcpy(p + 4,  &yaw,     2);
  memcpy(p + 6,  &odo,     4);
  memcpy(p + 10, &servo,   2);
  p[12] = floorC;
  p[13] = tries;
  memcpy(p + 14, &bootId,  4);
  buf[TELEM_LEN - 1] = xor8(p, TELEM_PAYLOAD);
  Serial.write(buf, TELEM_LEN);
}

// ============================================================
// PARAMETER TABLE  (Pi-owned: RAM only, re-pushed after every reset)
// ============================================================
// 20 entries, down from 91. Everything that described planning or passing
// went to the Pi with the logic that read it; what is left describes this
// board's hardware and the loops it still runs.
//
// WIRE FORMAT (newline-terminated ASCII, sharing the port with the frames)
//   Pi -> us   P<id> <v> | N <name> <v> | ?P | ?V | C
//   us -> Pi   !V <ver> <count> <boot> | !P <id> <name> <type> <val> <lo> <hi> <group>
//              !p <id> <val> | !E <what> | !C
//
// The dump is STREAMED: ?P arms a cursor and loop() emits at most
// PARAM_DUMP_PER_LOOP lines a pass, and only while the CDC buffer has room.

const uint16_t PARAM_VERSION       = 4;   // bumped: the table changed shape
const uint8_t  PARAM_DUMP_PER_LOOP = 2;
const uint16_t PARAM_TX_HEADROOM   = 96;

enum PType : uint8_t { PT_F, PT_I, PT_U32, PT_U16, PT_U8, PT_B };
enum PGroup : uint8_t { G_DRIVE, G_TURN, G_SAFE, G_LINK };

struct ParamDesc { const char *name; void *ptr; PType type; float lo, hi; uint8_t group; };

#define PF(n, g, lo, hi) { #n, (void *)&n, PT_F,   lo, hi, g }
#define PI_(n, g, lo, hi){ #n, (void *)&n, PT_I,   lo, hi, g }
#define PL(n, g, lo, hi) { #n, (void *)&n, PT_U32, lo, hi, g }
#define PS(n, g, lo, hi) { #n, (void *)&n, PT_U16, lo, hi, g }

const ParamDesc PARAMS[] = {
  // ---- hardware + the heading loop ----
  PF(TICKS_PER_CM,          G_DRIVE,   1,   100),
  PF(SERVO_TRUE_STRAIGHT,   G_DRIVE,  40,   120),
  PF(SERVO_MAX_LEFT,        G_DRIVE,   0,    90),
  PF(SERVO_MAX_RIGHT,       G_DRIVE,  90,   180),
  PF(STEER_LOCK_DEG,        G_DRIVE,   5,    60),
  PF(IMU_YAW_SIGN,          G_DRIVE,  -1,     1),
  PF(HEAD_KP,               G_DRIVE,   0,    10),
  PF(HEAD_KI,               G_DRIVE,   0,     5),
  PF(HEAD_KD,               G_DRIVE,   0,     5),
  PF(YAW_FILT_ALPHA,        G_DRIVE,   0,     1),
  PF(SERVO_SLEW,            G_DRIVE, 0.2,    30),
  PF(INTEGRAL_CLAMP,        G_DRIVE,   0,  2000),

  // ---- the arc's inner loop ----
  PF(TURN_KP,               G_TURN,  0.2,    10),
  PF(TURN_MIN_STEER,        G_TURN,    0,    45),
  PF(TURN_STOP_DEG,         G_TURN,    0,    45),

  // ---- the panic reflex ----
  PS(WALL_PANIC_MM,         G_SAFE,    0,  1000),
  PS(WALL_CLEAR_MM,         G_SAFE,   50,  1500),
  PF(RECOVER_MAX_CM,        G_SAFE,    5,   100),
  PI_(RECOVER_MAX_TRIES,    G_SAFE,    0,    20),

  // ---- link ----
  PL(LINK_STALE_MS,         G_LINK,   50,  2000),
};
const int PARAM_COUNT = (int)(sizeof(PARAMS) / sizeof(PARAMS[0]));

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
  paramsPushed = true;
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

int paramDumpIdx = -1;
void paramDumpStart() { paramDumpIdx = 0; }

void paramDumpStep() {
  if (paramDumpIdx < 0) return;
  for (uint8_t k = 0; k < PARAM_DUMP_PER_LOOP; k++) {
    if (paramDumpIdx >= PARAM_COUNT) { paramDumpIdx = -1; reportVersion(); return; }
    if ((uint16_t)Serial.availableForWrite() < PARAM_TX_HEADROOM) return;
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

bool parseTuning(char *s) {
  if (s[0] == '?' && s[1] == 'P' && s[2] == '\0') { paramDumpStart(); return true; }
  if (s[0] == '?' && s[1] == 'V' && s[2] == '\0') { reportVersion();  return true; }
  if (s[0] == 'C' && s[1] == '\0') { Serial.println(F("!C")); return true; }

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
    if (id < 0)                       { Serial.println(F("!E name"));  return true; }
    if (!paramSet(id, strtof(sp, 0))) { Serial.println(F("!E range")); return true; }
    Serial.print(F("!p ")); Serial.print(id);
    Serial.print(' ');      Serial.println(paramGet(id), 4);
    return true;
  }
  return false;
}

// ============================================================
// LINK SERVICE - one byte stream, two kinds of content
// ============================================================
// 0xAA while idle starts a candidate frame; anything else accumulates into an
// ASCII line until '\n'. 0xAA cannot appear inside a parameter line, so the
// two never interleave.

enum RxState { RX_IDLE, RX_SYNC2, RX_BODY };
RxState  rxState = RX_IDLE;
uint8_t  rxBuf[DRIVE_PAYLOAD + 1];
uint8_t  rxLen = 0;
char     lineBuf[96];
uint8_t  lineLen = 0;

void acceptDrive() {
  const uint8_t *p = rxBuf;
  if (xor8(p, DRIVE_PAYLOAD) != rxBuf[DRIVE_PAYLOAD]) return;   // bad frame
  dSeq   = p[0];
  dFlags = p[1];
  memcpy(&dHeading, p + 2, 2);
  memcpy(&dSteer,   p + 4, 2);
  dSpeed   = p[6];
  dArcLock = p[7];
  memcpy(&dLeft,  p + 8,  2);
  memcpy(&dFront, p + 10, 2);
  memcpy(&dRight, p + 12, 2);
  dRev = p[14];
  uint8_t cmd = p[15];

  // A command acts only after CMD_CONFIRM_FRAMES identical frames, so one
  // corrupt or mis-synced frame can never reset the car.
  if (cmd == cmdLast && cmd != CMD_NONE) {
    if (cmdRun < 255) cmdRun++;
  } else {
    cmdRun = (cmd == CMD_NONE) ? 0 : 1;
  }
  cmdLast = cmd;
  if (cmd == CMD_REBOOT && cmdRun >= CMD_CONFIRM_FRAMES) {
    Serial.println(F("# REBOOT from Pi"));
    Serial.flush();
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    delay(20);
    NVIC_SystemReset();
  }

  linkLastMs  = millis();
  linkStale   = false;
  linkEverUp  = true;
  driveFrames++;
}

void serviceLink() {
  while (Serial.available()) {
    uint8_t c = (uint8_t)Serial.read();
    switch (rxState) {
      case RX_IDLE:
        if (c == 0xAA) { rxState = RX_SYNC2; }
        else if (c == '\n' || c == '\r') {
          if (lineLen) { lineBuf[lineLen] = '\0'; parseTuning(lineBuf); lineLen = 0; }
        } else if (lineLen < sizeof(lineBuf) - 1) {
          lineBuf[lineLen++] = (char)c;
        } else {
          lineLen = 0;                        // no newline for ages: resync
        }
        break;
      case RX_SYNC2:
        if (c == 0x55) { rxState = RX_BODY; rxLen = 0; }
        else if (c == 0xAA) { /* stay: AA AA -> the second may start it */ }
        else { rxState = RX_IDLE; }
        break;
      case RX_BODY:
        rxBuf[rxLen++] = c;
        if (rxLen >= DRIVE_PAYLOAD + 1) { acceptDrive(); rxState = RX_IDLE; }
        break;
    }
  }
  if (millis() - linkLastMs > LINK_STALE_MS) linkStale = true;
}

// ============================================================
// SENSORS, once per loop
// ============================================================
void serviceSensors() {
  gImuFresh = false;
  if (myIMU.wasReset()) myIMU.enableGameRotationVector();
  if (myIMU.getSensorEvent() &&
      myIMU.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
    gImuFresh = true;
    gImuSeen  = true;
    float h = readHeading();
    unsigned long now = millis();
    float dt = (now - gPrevHT) / 1000.0;
    if (dt > 0.0) gYawRate = wrapDeg(h - gPrevH) / dt;
    gPrevH = h; gPrevHT = now;
    gHeading = h;
  }

  gRawColor = classifyColor();
  digitalWrite(LED2_PIN, gRawColor == COLOR_ORANGE ? HIGH : LOW);
  digitalWrite(LED3_PIN, gRawColor == COLOR_BLUE   ? HIGH : LOW);
}

// ============================================================
// INIT
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
// MAIN
// ============================================================
void setup() {
  Serial.begin(115200);
  bootId = (uint32_t)millis() ^ 0x5A5A0000u;   // a value the Pi has not seen
  initHardware();
  resetHeadingPid();
  reportVersion();
  Serial.println(F("# executor ready - waiting for DRIVE frames"));
}

void loop() {
  serviceLink();
  serviceSensors();
  applyDrive();
  paramDumpStep();

  unsigned long now = millis();
  if (now - telemLastMs >= TELEM_PERIOD_MS) {
    telemLastMs = now;
    sendTelem();
  }

  // LED1: no link yet = slow blink, link stale = fast blink, driving = solid
  if (!linkEverUp)      digitalWrite(LED1_PIN, ((now / 500) & 1) ? HIGH : LOW);
  else if (linkStale)   digitalWrite(LED1_PIN, ((now / 100) & 1) ? HIGH : LOW);
  else                  digitalWrite(LED1_PIN, enabled() ? HIGH : (((now / 250) & 1) ? HIGH : LOW));
}
