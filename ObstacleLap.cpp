// ============================================================================
// ObstacleLap.cpp — WRO Future Engineers obstacle round, LAP ONLY.
// (No parallel parking, no magenta, no start button, no rear ToF,
//  no colour sensor.)
//
// This is OpenRound.cpp with ONE state added. Everything that was already
// calibrated on this car — servo trim and limits, TICKS_PER_CM, the heading
// PID, the eased 90 deg arc, the wall recovery — is unchanged and still runs
// the car. Do not port this to ObstacleExecutor.cpp: that file's pin map
// (PB9/PB8 motor, TIM3 encoder, IMU on PB5/4/3) is from an earlier build and
// does not match this hardware.
//
// SPLIT OF WORK
//   Pi      camera + lidar, pillar tracking, and the avoidance solve. It sends
//           a solved manoeuvre: "hold this absolute heading for this many mm".
//   STM32   this file: the state machine, the heading PID at IMU rate, the
//           turn arc, odometry, the lidar/link staleness logic. Every fast
//           loop is here because a 50 Hz link cannot close a heading loop.
//           NOTE: there is no link-loss motor cut. If the Pi goes quiet the
//           car carries on under its own logic; to halt it the Pi sends
//           CMD = STOP (the dashboard's "Stop car").
//
// STATES
//   BOOT     nothing happens until the Pi says its lidar AND camera are up.
//   HEADING  hold the lane heading. Watch for a corner, a pillar, the finish.
//   AVOID    hold the heading the Pi solved, for the distance it asked for.
//   TURN90   eased 90 deg arc, terminated on IMU heading.
//   FINISH   stopped. A RERUN from the Pi sends it back to BOOT.
//   RECOVER  overlay: something is too close in front; back off and resume.
//   STOPPED  halted by CMD = STOP from any moving state. Motor off, steering
//            centred. A RERUN sends it back to BOOT, exactly as from FINISH.
//
// DRIVING DIRECTION  (lidar sides only — the colour sensor is not used)
//   Until the direction is locked, BOTH lidar sides are watched. The outer
//   wall never ends, so the first side to read past SIDE_OPEN_MM (1500) for
//   SIDE_OPEN_REVS revolutions is the inner side:
//       right opens first -> CLOCKWISE      left opens first -> COUNTER-CW
//   That same trigger is the first corner's turn. After the lock only the
//   locked side is watched, exactly as before. Two guards on the lock: a side
//   must have shown a wall at least once this run, and both sides opening
//   together locks nothing. LOCK_NEEDS_REAL_RETURN covers the one case they
//   do not (an outer-side dropout).
//
// LINK  (docs/OBSTACLE_LAP.md)
//   Pi  -> us   PERCEPT  17 bytes, sync AA 55, 50 Hz
//   us  -> Pi   TELEM    22 bytes, sync 55 AA, 50 Hz   (layout unchanged)
//   us  -> Pi   STATUS   61 bytes, sync 55 A5, 10 Hz   FSM internals for the
//                                                      dashboard; see sendStatus()
//   The '#' log lines below share the port. 0xAA and 0xA5 are not ASCII, so a
//   log line can never contain the TELEM or STATUS sync word and the Pi
//   separates the three cleanly.
//
//   PERCEPT (little-endian)
//     0-1    AA 55         sync
//     2      seq
//     3      flags         LIDAR_OK 01, CAM_OK 02, AVOID 0C, GREEN 10, HELLO 20
//     4-5    left   mm     u16
//     6-7    front  mm     u16
//     8-9    right  mm     u16
//     10     lidar revolution counter
//     11-12  avoid heading, deg x10   i16
//     13-14  avoid leg, mm            u16   (see AVOID's own comment: this is
//                                            only ever a backstop cap now, not
//                                            a leg the car counts down to)
//     15     CMD           0 = none, 1 = RERUN, 2 = STOP, 3 = REBOOT
//     16     base speed    u8 PWM for the straights (0 = unset, keep default);
//                          clamped to [SPEED_MIN, SPEED_MAX]. AVOID keeps a
//                          fixed fraction of it. Set live from the dashboard.
//     17     XOR of bytes 2..16
//
//   RERUN acts only on a 0 -> 1 change seen while the car is FINISHED (or
//   STOPPED). A 1 already being sent when the run ends does nothing, and
//   holding 1 gives one rerun, never run after run. The Pi can drop CMD back
//   to 0 as soon as TELEM shows state BOOT.
//
//   STOP and REBOOT act only once CMD_CONFIRM_FRAMES consecutive good frames
//   carry the same value, so one corrupt or mis-synced frame can never halt
//   or reset the car. STOP: any state but FINISH/STOPPED -> STOPPED; the Pi
//   holds 2 until TELEM shows STOPPED. REBOOT: motor off, then
//   NVIC_SystemReset() — USB drops and re-enumerates, setup() re-zeroes the
//   yaw, so the car must be still. A command frame need not carry ranges:
//   with LIDAR_OK clear it refreshes the link, never the lidar clock.
//
// LEDS
//   LED1  slow blink = waiting on Pi, faster = countdown, solid = running,
//         fast = lidar stale
//   LED2  direction locked CLOCKWISE
//   LED3  direction locked COUNTER-CLOCKWISE
//   all three solid = FINISHED
//   LED1 off, LED2/LED3 alternating = STOPPED
//
// BENCH-VERIFIED (unchanged from the open round)
//   motor    PA2 forward, PA3 reverse
//   encoder  TIM5 PA0/PA1, negated so forward counts up
//   IMU      BNO08x SPI1 ~100 Hz, clockwise = negative yaw
//   servo    500-2500 us, straight 76.5, left stop 20, right stop 140
//            turning radius at full lock: left 27 cm, right 25 cm
// ============================================================================

#include <Arduino.h>
#include <Servo.h>
#include <SPI.h>
#include <SparkFun_BNO08x_Arduino_Library.h>

// Must match STATE_NAMES / ST_* in control/percept_link.py.
enum RobotState {
  STATE_BOOT = 0,
  STATE_HEADING,
  STATE_AVOID,
  STATE_TURN90,
  STATE_FINISH,
  STATE_RECOVER,
  STATE_STOPPED
};

// Why the car is standing still at the end. Reported in STATUS only.
enum FinishReason { FINISH_NONE = 0, FINISH_BY_WALL, FINISH_BY_ODO, FINISH_BY_STOP_CMD };

// ============================================================================
// HARDWARE PINS & OBJECTS
// ============================================================================
const int MOT_RPWM_PIN = PA2;     // forward  (TIM2_CH3; TIM5 is the encoder)
const int MOT_LPWM_PIN = PA3;     // reverse  (TIM2_CH4)
const int SERVO_PIN    = PA8;

const int IMU_CS_PIN  = PA4;
const int IMU_INT_PIN = PB0;
const int IMU_RST_PIN = PB1;

const int LED1_PIN = PB12;   // slow blink = waiting on Pi; solid = running; fast = lidar stale
const int LED2_PIN = PB13;   // direction locked CLOCKWISE
const int LED3_PIN = PB14;   // direction locked COUNTER-CLOCKWISE
const int BTN_START_PIN = PB15;   // not used this round

// The colour sensor (TCS34725 behind the TCA9548A on PB6/PB7) is not read at
// all, so it can stay mounted or come off. Nothing in this file touches I2C.

// ---- calibration ----
const float TICKS_PER_CM        = 14.853;
const float TICKS_PER_MM        = TICKS_PER_CM / 10.0;
const float MM_PER_TICK         = 10.0 / TICKS_PER_CM;

const int   SERVO_MIN_PULSE_US  = 500;
const int   SERVO_MAX_PULSE_US  = 2500;
const float SERVO_TRUE_STRAIGHT = 76.5;
const float SERVO_MAX_LEFT      = 20.0;   // below straight steers LEFT
const float SERVO_MAX_RIGHT     = 140.0;  // above straight steers RIGHT
const float IMU_YAW_SIGN        = 1.0;

const int BASE_SPEED  = 70;    // default straight speed (until the Pi sends one)
const int AVOID_SPEED = 60;    // default pillar-threading speed (slower buys accuracy)
const int SPEED_MIN   = 40;    // a Pi-commanded base speed is clamped to this band:
const int SPEED_MAX   = 150;   //   below SPEED_MIN the car stalls; above is unsafe
// Runtime speeds. Start at the defaults above; the dashboard's speed slider
// overwrites baseSpeed via PERCEPT byte 16, and avoidSpeed keeps the same
// fraction of base the two defaults have (60/70). TURN90 and RECOVER run their
// own PWM laws and are not touched by the slider.
int baseSpeed  = BASE_SPEED;
int avoidSpeed = AVOID_SPEED;

const unsigned long START_DELAY_MS = 5000;

SPIClass SPI_IMU(PA7, PA6, PA5);  // MOSI, MISO, SCLK
Servo steeringServo;
BNO08x myIMU;

float initialYawOffset = 0.0;

// ============================================================================
// LINK — PERCEPT in, TELEM out
// ============================================================================
const uint8_t PERCEPT_SYNC0 = 0xAA, PERCEPT_SYNC1 = 0x55;
const uint8_t TELEM_SYNC0   = 0x55, TELEM_SYNC1   = 0xAA;
const uint8_t PERCEPT_LEN = 18, TELEM_LEN = 22;

const uint8_t P_LIDAR_OK = 0x01;
const uint8_t P_CAM_OK   = 0x02;
const uint8_t P_AVOID_MASK = 0x0C, P_AVOID_SHIFT = 2;
const uint8_t P_GREEN    = 0x10;
const uint8_t P_HELLO    = 0x20;

// AVOID_COMMIT (2) is still a valid value on the wire but the Pi no longer
// sends it (see control/supervisor.py) — the only actions that arrive now are
// NONE and TRACK. Left in place so an old Pi build, or a future one, cannot
// desync the enum; this firmware treats a stray COMMIT exactly like TRACK.
const uint8_t AVOID_NONE = 0, AVOID_TRACK = 1, AVOID_COMMIT = 2;

// PERCEPT byte 15. Only these mean anything; every other value reads as NONE.
const uint8_t CMD_NONE = 0, CMD_RERUN = 1, CMD_STOP = 2, CMD_REBOOT = 3;
const uint8_t CMD_CONFIRM_FRAMES = 3;   // STOP / REBOOT need this many in a row

const uint8_t S_RUNNING = 0x01, S_LIDAR_STALE = 0x02, S_LIDAR_DEAD = 0x04;
const uint8_t S_DIR_LOCKED = 0x08, S_CLOCKWISE = 0x10, S_IMU_OK = 0x20;
const uint8_t S_COLOUR_OK = 0x40;   // never set any more: no colour sensor
const uint8_t S_RECOVERING = 0x80;

// ---- STATUS (us -> Pi): FSM internals TELEM does not carry. 10 Hz. ----
// Reporting only: nothing in here feeds a decision. Layout in sendStatus();
// must match parse_status() in control/percept_link.py and tests/test_wire.py.
const uint8_t  STATUS_SYNC0 = 0x55, STATUS_SYNC1 = 0xA5;
const uint8_t  STATUS_LEN = 61;
const uint8_t  STATUS_VERSION = 1;
const uint32_t STATUS_PERIOD_MS = 100;
// flags
const uint8_t SF_FSM_STARTED = 0x01;    // a PERCEPT frame has arrived since power-up
const uint8_t SF_BOOT_READY  = 0x02;    // BOOT: Pi ready, 5 s countdown running
const uint8_t SF_LIDAR_HOLD  = 0x04;    // HEADING parked the car: lidar dead
const uint8_t SF_LINK_STALE  = 0x08;    // no PERCEPT for LINK_STALE_MS
const uint8_t SF_BLIND       = 0x10;    // this run's GO happened with the camera down
const uint8_t SF_RERUN_ARMED = 0x20;    // FINISH/STOPPED: a RERUN would be taken now
const uint8_t SF_WALL_SEEN_L = 0x40;
const uint8_t SF_WALL_SEEN_R = 0x80;
// flags2
const uint8_t SF2_REAL_OPEN_L   = 0x01;
const uint8_t SF2_REAL_OPEN_R   = 0x02;
const uint8_t SF2_LOCK_NEEDS_REAL = 0x04;   // the LOCK_NEEDS_REAL_RETURN constant
// pflags: the last PERCEPT, as this side understood it
const uint8_t PF_HELLO = 0x01, PF_LIDAR_OK = 0x02, PF_CAM_OK = 0x04, PF_GREEN = 0x08;
const uint8_t PF_ACTION_SHIFT = 4;      // bits 4-5

// Two thresholds on purpose. STALE (short) just stops us acting on old
// distances. DEAD (long) means the lidar is gone: HEADING parks the car until
// it is back. A single dropped batch of frames must not be read as "gone".
const unsigned long LIDAR_STALE_MS     = 200;
const unsigned long LIDAR_DEAD_MS      = 1000;
const unsigned long LINK_STALE_MS      = 250;    // Pi quiet: stop trusting TRACK
const uint16_t      LIDAR_MAX_VALID_MM = 3500;   // mat diagonal
const uint16_t      LIDAR_FAR          = 9999;   // internal "nothing there"
const uint32_t      TELEM_PERIOD_MS    = 20;     // 50 Hz

// A beam with no return must read FAR, never near. A dropout treated as 0 mm
// would latch the wall-panic recovery on permanently.
uint16_t lidarSanitize(uint16_t v) {
  if (v == 0 || v == 0xFFFF || v > LIDAR_MAX_VALID_MM) return LIDAR_FAR;
  return v;
}

uint16_t      lidarL = LIDAR_FAR, lidarF = LIDAR_FAR, lidarR = LIDAR_FAR;
uint8_t       lidarRev = 0, lastRev = 0;
// Two clocks, not one. lidarLastMs only advances on frames the Pi marked
// LIDAR_OK, so a frame carrying a committed leg but no ranges cannot clear the
// stale flag and make LIDAR_FAR side readings look like an open corner.
unsigned long lidarLastMs = 0, linkLastMs = 0;
bool          lidarStale = true, lidarDead = true, linkStale = true;
uint32_t      perceptFrames = 0;
uint8_t       rxSeq = 0;

// latest solved manoeuvre from the Pi
uint8_t pAction = AVOID_NONE;
bool    pGreen = false, pLidarOk = false, pCamOk = false, pHello = false;
float   pHeadingDeg = 0.0;
uint16_t pLegMm = 0;
uint8_t pCmd = CMD_NONE;          // latest CMD byte (RERUN: FINISH/STOPPED only)
uint8_t pCmdRun = 0;              // consecutive good frames carrying pCmd
uint8_t pBaseSpeed = 0;           // latest base-speed byte (0 = unset; see applyPercept)

uint8_t rxBuf[PERCEPT_LEN];
uint8_t rxLen = 0;

static inline uint8_t xor8(const uint8_t *p, uint8_t n) {
  uint8_t x = 0; while (n--) x ^= *p++;
  return x;
}

void applyPercept(const uint8_t *f) {
  rxSeq          = f[2];
  uint8_t flags  = f[3];
  pLidarOk       = flags & P_LIDAR_OK;
  pCamOk         = flags & P_CAM_OK;
  pHello         = flags & P_HELLO;
  pGreen         = flags & P_GREEN;
  pAction        = (flags & P_AVOID_MASK) >> P_AVOID_SHIFT;

  uint16_t l, fr, r, leg; int16_t head_dd;
  memcpy(&l,       f + 4,  2);
  memcpy(&fr,      f + 6,  2);
  memcpy(&r,       f + 8,  2);
  lidarRev = f[10];
  memcpy(&head_dd, f + 11, 2);
  memcpy(&leg,     f + 13, 2);
  pCmdRun  = (f[15] == pCmd) ? (uint8_t)min(255, pCmdRun + 1) : 1;
  pCmd     = f[15];

  // Base speed (byte 16). 0 means "unset" — keep whatever we have (the default,
  // or the last value the Pi sent), so a command-only frame (which carries 0)
  // never zeroes the speed. Otherwise clamp to the safe band and scale AVOID by
  // the same fraction the two defaults have.
  pBaseSpeed = f[16];
  if (pBaseSpeed != 0) {
    baseSpeed  = constrain((int)pBaseSpeed, SPEED_MIN, SPEED_MAX);
    avoidSpeed = (int)(baseSpeed * (AVOID_SPEED / (float)BASE_SPEED) + 0.5);
  }

  pHeadingDeg = head_dd / 10.0;
  pLegMm      = leg;

  if (pLidarOk) {
    lidarL = lidarSanitize(l);
    lidarF = lidarSanitize(fr);
    lidarR = lidarSanitize(r);
    lidarLastMs = millis();
    lidarStale = lidarDead = false;
  }
  linkLastMs = millis();
  linkStale = false;
  perceptFrames++;
}

void serviceLink() {
  while (Serial.available()) {
    uint8_t c = Serial.read();
    if (rxLen == 0) { if (c == PERCEPT_SYNC0) rxBuf[rxLen++] = c; continue; }
    if (rxLen == 1) {
      if (c == PERCEPT_SYNC1) rxBuf[rxLen++] = c;
      else rxLen = (c == PERCEPT_SYNC0) ? 1 : 0;   // resync without losing a sync
      continue;
    }
    rxBuf[rxLen++] = c;
    if (rxLen == PERCEPT_LEN) {
      if (xor8(rxBuf + 2, PERCEPT_LEN - 3) == rxBuf[PERCEPT_LEN - 1]) applyPercept(rxBuf);
      rxLen = 0;                                   // bad checksum: drop silently
    }
  }
  unsigned long age = millis() - lidarLastMs;
  if (age > LIDAR_STALE_MS) lidarStale = true;
  if (age > LIDAR_DEAD_MS)  lidarDead  = true;
  if (millis() - linkLastMs > LINK_STALE_MS) linkStale = true;
}

// ============================================================================
// RUN CONSTANTS
// ============================================================================
const uint16_t SIDE_OPEN_MM     = 1500;  // side above this = inner wall gone (before the lock: either side)
const uint8_t  SIDE_OPEN_REVS   = 3;     // consecutive lidar REVOLUTIONS, not frames

// A no-return reads FAR, which is past SIDE_OPEN_MM, so by default it counts as
// "open" for the direction lock exactly as it does for every later corner.
// The catch: an OUTER-side dropout lasting SIDE_OPEN_REVS revolutions before
// the first corner would lock the wrong way. If the lock log shows a real
// distance (not 9999) for the open side at the first corner, set this true:
// the locking side must then return an actual distance past SIDE_OPEN_MM at
// least once in its streak, and a dropout alone can never lock.
const bool     LOCK_NEEDS_REAL_RETURN = false;

const uint16_t WALL_PANIC_MM     = 200;
const uint16_t WALL_CLEAR_MM     = 350;
const int      RECOVER_PWM       = 90;
const float    RECOVER_MAX_CM    = 30.0;
const int      RECOVER_MAX_TRIES = 3;

const int   TARGET_CORNERS         = 12;
const float SEARCH_SAFETY_CM       = 400.0;
const float POST_CORNER_LOCKOUT_CM = 50.0;
// Was sized for the old leg-driven avoid (which could run for several seconds
// past the pillar). Now that AVOID releases the instant the Pi does — see the
// AVOID state's own comment — the car spends far less time off-lane, so this
// only needs to cover the heading PID's own settle time. Re-tune on the mat;
// if a corner right after a pillar is still missed, try this lower before
// touching anything else.
const float POST_AVOID_LOCKOUT_CM  = 15.0;

const float    AVOID_MAX_LEG_CM   = 150.0;   // backstop cap; see AVOID's comment
const uint32_t AVOID_TIMEOUT_MS   = 5000;    // absolute backstop; see AVOID's comment
const uint16_t FINISH_TOL_MM      = 60;      // front-distance match tolerance
const float    FINAL_FALLBACK_CM  = 100.0;

// ============================================================================
// FSM DATA
// ============================================================================
bool fsmStarted = false;
unsigned long firstFrameMs = 0;
const unsigned long CAMERA_GRACE_MS = 15000;  // start without the camera after this

RobotState    currentState = STATE_BOOT;
bool          entered = false;
unsigned long phaseT0 = 0;

// Reporting only (STATUS frame); none of these drive a decision.
RobotState    gTrackedState = STATE_BOOT;
unsigned long gStateSinceMs = 0;          // when currentState last changed
uint8_t       finishReason  = FINISH_NONE;
bool          startedBlind  = false;      // this run's GO had the camera down
uint8_t       runNumber     = 0;          // GOs since power-up (1 = first run)

// Driving direction, locked at the first corner by whichever lidar side opens
// first. clockwiseMode means nothing until dirLocked is set.
bool dirLocked     = false;
bool clockwiseMode = true;
int  cornerCount   = 0;

// A side may only lock the direction once it has shown a wall this run. A
// blocked or broken sector reads FAR (> SIDE_OPEN_MM) from the start, and
// without this it would look like "the inner wall ended" the moment we moved.
bool wallSeenL = false, wallSeenR = false;

// RERUN edge detector: cleared on entry to FINISH, set there by any CMD that
// is not RERUN. See finishStep().
bool rerunArmed = false;

float targetHeading = 0.0;
float laneHeading   = 0.0;

uint16_t frontAtStartMm = LIDAR_FAR;   // the finish line, measured not guessed

// cached sensors, refreshed every loop
bool          gImuFresh = false;
float         gHeading = 0.0, gYawRate = 0.0, gPrevH = 0.0;
unsigned long gPrevHT = 0;

// Odometry that NEVER resets. zeroEncoder() still zeroes TIM5 for the open
// round's per-segment measurements, so the Pi needs its own monotonic count.
long gOdoTicks = 0, gLastRawEnc = 0;
float gSpeedMmps = 0.0;

// ============================================================================
// HELPERS
// ============================================================================
float wrapDeg(float a) {
  while (a > 180.0)  a -= 360.0;
  while (a < -180.0) a += 360.0;
  return a;
}

int gMotorPwm = 0;   // last commanded PWM, reported in STATUS

void setMotorSpeed(int speed) {
  speed = constrain(speed, -255, 255);
  gMotorPwm = speed;
  if (speed > 0)      { analogWrite(MOT_RPWM_PIN, speed); analogWrite(MOT_LPWM_PIN, 0); }
  else if (speed < 0) { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, -speed); }
  else                { analogWrite(MOT_RPWM_PIN, 0);     analogWrite(MOT_LPWM_PIN, 0); }
}

float lastServoCmd = SERVO_TRUE_STRAIGHT;

void setServoAngle(float angleDeg) {
  angleDeg = constrain(angleDeg, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  lastServoCmd = angleDeg;
  int pulse = (int)((angleDeg / 180.0) * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US))
              + SERVO_MIN_PULSE_US;
  steeringServo.writeMicroseconds(pulse);
}

long readEncoder() { return -(int32_t)TIM5->CNT; }
long absEnc(long v) { return v < 0 ? -v : v; }

void serviceOdo() {
  long raw = readEncoder();
  gOdoTicks += (raw - gLastRawEnc);
  gLastRawEnc = raw;
}
void zeroEncoder() { TIM5->CNT = 0; gLastRawEnc = 0; }

// Clockwise means the inner wall is on the right, so the RIGHT side gives way
// at the corner. Counter-clockwise, the left. Only meaningful once dirLocked.
uint16_t turnSideMm() { return clockwiseMode ? lidarR : lidarL; }

// ---- IMU ----
float readYaw() {
  float qI = myIMU.getQuatI(), qJ = myIMU.getQuatJ();
  float qK = myIMU.getQuatK(), qReal = myIMU.getQuatReal();
  if (qI == 0.0f && qJ == 0.0f && qK == 0.0f && qReal == 0.0f) return 0.0f;
  return atan2(2.0f * (qI * qJ + qReal * qK),
               (qReal * qReal + qI * qI - qJ * qJ - qK * qK)) * (180.0 / PI);
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

void serviceSensors() {
  serviceLink();
  serviceOdo();

  gImuFresh = false;
  if (myIMU.wasReset()) myIMU.enableGameRotationVector();
  if (myIMU.getSensorEvent() &&
      myIMU.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
    gImuFresh = true;
    float h = readHeading();
    unsigned long now = millis();
    float dt = (now - gPrevHT) / 1000.0;
    if (dt > 0.0) gYawRate = wrapDeg(h - gPrevH) / dt;
    gPrevH = h; gPrevHT = now; gHeading = h;
  }

  // Latched in every state, so an avoid at the start cannot keep a side from
  // ever proving it has a wall.
  if (!lidarStale) {
    if (lidarL <= SIDE_OPEN_MM) wallSeenL = true;
    if (lidarR <= SIDE_OPEN_MM) wallSeenR = true;
  }
}

// ============================================================================
// TELEMETRY
// ============================================================================
long  avoidBaseTicks = 0, avoidLegTicks = 0;

void sendTelemetry() {
  static long prevOdo = 0;
  static uint32_t prevT = 0;
  uint32_t now = millis();
  if (prevT && now > prevT)
    gSpeedMmps = (gOdoTicks - prevOdo) * MM_PER_TICK * 1000.0 / (now - prevT);
  prevOdo = gOdoTicks; prevT = now;

  uint8_t f[TELEM_LEN];
  f[0] = TELEM_SYNC0; f[1] = TELEM_SYNC1;
  f[2] = rxSeq;

  uint8_t st = 0;
  if (fsmStarted && currentState != STATE_FINISH &&
      currentState != STATE_STOPPED) st |= S_RUNNING;
  if (lidarStale)                st |= S_LIDAR_STALE;
  if (lidarDead)                 st |= S_LIDAR_DEAD;
  if (dirLocked)                 st |= S_DIR_LOCKED;
  if (clockwiseMode)             st |= S_CLOCKWISE;
  if (gPrevHT != 0)              st |= S_IMU_OK;
  if (currentState == STATE_RECOVER) st |= S_RECOVERING;
  f[3] = st;

  int32_t  odoMm   = (int32_t)(gOdoTicks * MM_PER_TICK);
  int16_t  spd     = (int16_t)gSpeedMmps;
  int16_t  head_dd = (int16_t)(gHeading * 10.0);
  int16_t  yaw_dd  = (int16_t)(gYawRate * 10.0);
  uint16_t frontMm = (lidarF == LIDAR_FAR) ? 0xFFFF : lidarF;
  uint8_t  state   = (uint8_t)currentState;
  uint8_t  corners = (uint8_t)cornerCount;
  long     rem     = (currentState == STATE_AVOID)
                     ? (avoidLegTicks - (gOdoTicks - avoidBaseTicks)) : 0;
  uint16_t remMm   = (rem > 0) ? (uint16_t)min((long)0xFFFE, (long)(rem * MM_PER_TICK)) : 0;
   memcpy(f + 4,  &odoMm,   4);
  memcpy(f + 8,  &spd,     2);
  memcpy(f + 10, &head_dd, 2);
  memcpy(f + 12, &yaw_dd,  2);
  memcpy(f + 14, &frontMm, 2);
  f[16] = state;
  f[17] = corners;
  memcpy(f + 18, &remMm,   2);
  f[20] = 0;                           // floor colour: always none now
  f[21] = xor8(f + 2, TELEM_LEN - 3);

  Serial.write(f, TELEM_LEN);
}

// ============================================================================
// HEADING PID  (unchanged from the open round)
// ============================================================================
const float HEAD_KP        = 2.0;
const float YAW_FILT_ALPHA = 0.35;
const float SERVO_SLEW     = 2.5;
const float INTEGRAL_CLAMP = 300.0;
const float HEAD_KI        = 0.0;
const float HEAD_KD        = 0.0;

unsigned long pidPrevTime = 0;
float pidIntegral = 0.0, yawFilt = 0.0, prevServoCmd = SERVO_TRUE_STRAIGHT;

void resetHeadingPid() {
  pidPrevTime = millis();
  pidIntegral = 0.0; yawFilt = 0.0;
  prevServoCmd = SERVO_TRUE_STRAIGHT;
}

void updateHeadingPid(float heading) {
  if (!gImuFresh) return;
  unsigned long now = millis();
  yawFilt += YAW_FILT_ALPHA * (gYawRate - yawFilt);
  float dt = (now - pidPrevTime) / 1000.0;
  if (dt <= 0.0) dt = 0.001;
  float error = wrapDeg(heading - gHeading);
  pidIntegral += error * dt;
  pidIntegral  = constrain(pidIntegral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP);
  float correction = HEAD_KP * error + HEAD_KI * pidIntegral - HEAD_KD * yawFilt;
  float want = SERVO_TRUE_STRAIGHT - correction;
  float dcmd = constrain(want - prevServoCmd, -SERVO_SLEW, SERVO_SLEW);
  float cmd  = constrain(prevServoCmd + dcmd, SERVO_MAX_LEFT, SERVO_MAX_RIGHT);
  if (cmd <= SERVO_MAX_LEFT || cmd >= SERVO_MAX_RIGHT) pidIntegral -= error * dt;
  setServoAngle(cmd);
  prevServoCmd = cmd;
  pidPrevTime  = now;
}

// ============================================================================
// EASED TURN LAW  (unchanged from the open round)
// ============================================================================
const float TURN_KP              = 2.5;
const float TURN_MAX_STEER_LEFT  = SERVO_TRUE_STRAIGHT - SERVO_MAX_LEFT;   // 56.5
const float TURN_MAX_STEER_RIGHT = SERVO_MAX_RIGHT - SERVO_TRUE_STRAIGHT;  // 63.5
const float TURN_MIN_STEER       = 8.0;
const float TURN_KV              = 3.5;
const int   TURN_MAX_PWM         = 130;
const int   TURN_MIN_PWM         = 100;
const float TURN_STOP_DEG        = 0.3;

long turnStartTicks = 0, turnCapTicks = 0;

bool turnArcStep(float target) {
  if (absEnc(readEncoder() - turnStartTicks) >= turnCapTicks) return true;
  if (gImuFresh) {
    float err = wrapDeg(target - gHeading);
    if (fabs(err) < TURN_STOP_DEG) return true;
    float mag      = fabs(err);
    float maxSteer = (err > 0) ? TURN_MAX_STEER_LEFT : TURN_MAX_STEER_RIGHT;
    float steer    = constrain(TURN_KP * mag, TURN_MIN_STEER, maxSteer);
    int   pwm      = (int)constrain(TURN_KV * mag, (float)TURN_MIN_PWM, (float)TURN_MAX_PWM);
    setServoAngle((err > 0) ? (SERVO_TRUE_STRAIGHT - steer) : (SERVO_TRUE_STRAIGHT + steer));
    setMotorSpeed(pwm);
  }
  return false;
}

// ============================================================================
// STATE HELPERS + working vars
// ============================================================================
void goState(RobotState s) { currentState = s; entered = false; }

long    dcBaseTicks, dcLockoutTicks, dcSafetyTicks;
uint8_t openRevsL = 0, openRevsR = 0;   // consecutive revolutions each side has read open
bool    realOpenL = false, realOpenR = false;   // that streak included a real distance, not just FAR
bool    lidarHold = false;              // HEADING has parked the car: lidar dead

// Set right before AVOID hands back to HEADING, so headingStep()'s entry does
// NOT zero the open-revolution counters — see the note there and in AVOID.
bool    suppressCornerReset = false;

// One lidar revolution for one side: extend or break its open streak.
static inline void countOpenRev(uint8_t &n, bool &real, uint16_t mm) {
  if (mm <= SIDE_OPEN_MM) { n = 0; real = false; return; }
  if (n < 250) n++;
  if (mm != LIDAR_FAR) real = true;
}

float turnTarget, turnAmount;

long  avoidLockoutFrom = -1000000;
float avoidTargetHeading = 0.0;

// Open-round segment measurement, kept as the finish FALLBACK only. Avoidance
// zig-zags inflate encoder distance relative to straight-line distance, so the
// front-wall match below is the primary terminator.
float firstSegmentCm = 0.0, fullStartStraightCm = 0.0;
bool  haveFullStraight = false;
float finalDistanceCm = FINAL_FALLBACK_CM;
long  fsTargetTicks = (long)(FINAL_FALLBACK_CM * TICKS_PER_CM);   // never 0: 0 finishes instantly

// ============================================================================
// RECOVER  (overlay: interrupts a moving state and returns to it)
// ============================================================================
RobotState recoverReturnState = STATE_HEADING;
bool       recoverReturnEntered = false;
long       recoverBaseTicks = 0;
int        recoverTries = 0;

void enterRecovery() {
  recoverReturnState   = currentState;
  recoverReturnEntered = entered;
  currentState  = STATE_RECOVER;
  entered = false;
}

void recoverStep() {
  if (!entered) {
    entered = true;
    setMotorSpeed(0);
    // Counter-steer while backing: keeps rotating the car the way it was
    // already turning instead of retracing the arc it came in on.
    setServoAngle(2.0 * SERVO_TRUE_STRAIGHT - lastServoCmd);
    setMotorSpeed(-RECOVER_PWM);
    recoverBaseTicks = gOdoTicks;
    Serial.print(F("# RECOVER front=")); Serial.println(lidarF);
  }

  bool clear     = !lidarStale && (lidarF >= WALL_CLEAR_MM);
  bool backedFar = absEnc(gOdoTicks - recoverBaseTicks) >= (long)(RECOVER_MAX_CM * TICKS_PER_CM);
  if (!clear && !backedFar) return;   // stale => clear is false => the cap decides

  setMotorSpeed(0);
  if (clear) { recoverTries = 0; Serial.println(F("# recover clear")); }
  else       { recoverTries++;   Serial.print(F("# recover capped, try ")); Serial.println(recoverTries); }

  currentState = recoverReturnState;
  entered      = recoverReturnEntered;
  resetHeadingPid();
  if (currentState == STATE_HEADING)    setMotorSpeed(baseSpeed);
  else if (currentState == STATE_AVOID) setMotorSpeed(avoidSpeed);
  // TURN90 sets its own PWM each step.
}

// ============================================================================
// RUN RESET
//
// Everything one run leaves behind. Called on every entry to BOOT, so a rerun
// starts from the same state a power-up does. The yaw is deliberately NOT
// re-zeroed: laneHeading is taken from wherever the car points at GO, and the
// heading frame the Pi solves in stays continuous across runs.
// ============================================================================
void resetRun() {
  dirLocked     = false;
  clockwiseMode = true;
  wallSeenL = wallSeenR = false;
  cornerCount   = 0;
  recoverTries  = 0;
  lidarHold     = false;
  openRevsL = openRevsR = 0;
  realOpenL = realOpenR = false;
  suppressCornerReset = false;
  avoidLockoutFrom    = gOdoTicks - 1000000L;   // far in the past: no lockout
  firstSegmentCm      = 0.0;
  fullStartStraightCm = 0.0;
  haveFullStraight    = false;
  finalDistanceCm     = FINAL_FALLBACK_CM;
  fsTargetTicks       = (long)(FINAL_FALLBACK_CM * TICKS_PER_CM);
  finishReason        = FINISH_NONE;                    // reporting only
  startedBlind        = false;
}

// ============================================================================
// STATE: BOOT
//
// Nothing happens until the Pi has sent one well-formed PERCEPT frame saying
// its lidar AND camera are up. That is the whole handshake: the car cannot
// start moving while perception is still coming up, and it cannot silently run
// an obstacle round blind. If the camera never comes up but the lidar does, we
// start anyway after CAMERA_GRACE_MS — an open-round lap scores more than a
// car that never moves — and say so in the log.
//
// A RERUN comes back through here too: same handshake, same 5 s countdown.
// ============================================================================
void bootStep() {
  if (!entered) {
    entered = true;
    phaseT0 = 0;
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    resetRun();
    Serial.println(F("# BOOT waiting for Pi"));
  }

  if (phaseT0 == 0) {
    digitalWrite(LED1_PIN, ((millis() / 500) & 1) ? HIGH : LOW);
    bool ready = pHello ||
                 (pLidarOk && (millis() - firstFrameMs > CAMERA_GRACE_MS));
    if (!ready) return;
    if (!pCamOk) Serial.println(F("# WARN camera down - lap will run blind to pillars"));
    startedBlind = !pCamOk;
    phaseT0 = millis();
    Serial.println(F("# Pi ready - 5s countdown"));
  }

  digitalWrite(LED1_PIN, ((millis() / 250) & 1) ? HIGH : LOW);
  if (millis() - phaseT0 < START_DELAY_MS) return;

  digitalWrite(LED1_PIN, HIGH);
  laneHeading   = gHeading;
  targetHeading = laneHeading;
  frontAtStartMm = lidarF;            // the finish line, measured here
  zeroEncoder();
  if (runNumber < 255) runNumber++;
  Serial.print(F("# GO  frontAtStart=")); Serial.println(frontAtStartMm);
  goState(STATE_HEADING);
}

// ============================================================================
// STATE: HEADING  (maintain heading — every other state converges back here)
//
// Priority, and why: corner beats pillar. Missing a pillar costs points;
// missing a corner ends the run. Avoidance is checked before the post-corner
// lockout because a pillar can sit immediately after a turn, but the lockout
// still gates the CORNER trigger so the corner just turned cannot re-fire.
// ============================================================================
void headingStep() {
  if (!entered) {
    entered = true;
    // Coming back from AVOID keeps whatever open-revolution progress the
    // locked side already had going into it — see AVOID's own comment. Every
    // other way of reaching HEADING (a fresh GO, or just having finished a
    // turn) really is a new corridor, so it resets as before.
    if (!suppressCornerReset) {
      openRevsL = openRevsR = 0;
      realOpenL = realOpenR = false;
    }
    suppressCornerReset = false;
    lidarHold = false;
    resetHeadingPid();
    dcBaseTicks = gOdoTicks;
    setMotorSpeed(baseSpeed);
    dcLockoutTicks = (cornerCount > 0) ? (long)(POST_CORNER_LOCKOUT_CM * TICKS_PER_CM) : 0;
    dcSafetyTicks  = (long)(SEARCH_SAFETY_CM * TICKS_PER_CM);
    Serial.println(F("# HEADING"));
  }

  updateHeadingPid(targetHeading);

  // ---- pillar: the Pi has solved a manoeuvre ----
  // Checked before everything else including the finish, because the last
  // section has pillars in it too and a hit there costs more than a late stop.
  if (pAction != AVOID_NONE && !linkStale) { goState(STATE_AVOID); return; }

  // ---- the finish: same distance from the front wall as where we started ----
  // The wall test is primary because it is geometric. The open round's L - A
  // encoder arithmetic assumes path length equals straight-line length, and
  // avoidance zig-zags break that assumption; it is kept only as the fallback.
  if (cornerCount >= TARGET_CORNERS) {
    long run = absEnc(readEncoder());
    bool byWall = (frontAtStartMm != LIDAR_FAR) && !lidarStale &&
                  (lidarF != LIDAR_FAR) && (run > (long)(10.0 * TICKS_PER_CM)) &&
                  (lidarF <= frontAtStartMm + FINISH_TOL_MM);
    bool byOdo  = run >= fsTargetTicks;
    if (byWall || byOdo) {
      Serial.print(F("# FINISH by ")); Serial.println(byWall ? F("wall") : F("odo"));
      finishReason = byWall ? FINISH_BY_WALL : FINISH_BY_ODO;
      goState(STATE_FINISH);
    }
    return;                            // no corners left, just the run-out
  }

  // ---- lidar gone: park until it is back ----
  // Corners come only from the lidar now. The old fallback turned on the floor
  // line; without it, driving on blind means missing the next corner and
  // pushing into the outer wall with the wall panic disabled. So stop, and
  // pick the straight up again from scratch when the lidar returns. (The
  // final run-out above keeps going: it has the odometry fallback.)
  if (lidarDead) {
    if (!lidarHold) {
      lidarHold = true;
      setMotorSpeed(0);
      Serial.println(F("# lidar dead - holding"));
    }
    return;
  }
  if (lidarHold) {
    Serial.println(F("# lidar back"));
    entered = false;                   // re-enter: speed, PID, counters
    return;
  }

  // Genuinely cruising now (not parked, not held): re-assert the base speed
  // every tick so the dashboard's slider takes effect mid-straight, not only
  // on the next HEADING entry. Placed AFTER the dead/hold returns above so it
  // can never override their motor-off.
  setMotorSpeed(baseSpeed);

  long straightTicks = absEnc(gOdoTicks - dcBaseTicks);
  if (straightTicks >= dcSafetyTicks) {
    Serial.println(F("# WARN no turn trigger within safety distance, retrying"));
    entered = false;
    return;
  }

  // Corner lockouts: one from the turn just made, one from the avoid just made.
  if (absEnc(readEncoder()) <= dcLockoutTicks) return;
  if (absEnc(gOdoTicks - avoidLockoutFrom) <= (long)(POST_AVOID_LOCKOUT_CM * TICKS_PER_CM)) return;

  // ---- side openings: counted per lidar REVOLUTION ----
  // Each bearing gets at most one new sample per revolution (~10 Hz on the C1),
  // so counting per frame at 50 Hz would "confirm" on one measurement resent.
  // Both sides are always counted; before the lock both matter, after it only
  // the locked side does.
  if (lidarStale) {
    openRevsL = openRevsR = 0;
    realOpenL = realOpenR = false;
  } else if (lidarRev != lastRev) {
    lastRev = lidarRev;
    countOpenRev(openRevsL, realOpenL, lidarL);
    countOpenRev(openRevsR, realOpenR, lidarR);
  }

  if (!dirLocked) {
    // First corner. The outer wall is continuous, so only the inner side can
    // open: the first one to do so gives the direction. Both open at once is
    // a blind lidar, not a corner — wait for one of them to see a wall again.
    bool openL = wallSeenL && (openRevsL >= SIDE_OPEN_REVS) && (realOpenL || !LOCK_NEEDS_REAL_RETURN);
    bool openR = wallSeenR && (openRevsR >= SIDE_OPEN_REVS) && (realOpenR || !LOCK_NEEDS_REAL_RETURN);
    if (openL == openR) return;
    dirLocked     = true;
    clockwiseMode = openR;
    Serial.print(clockwiseMode ? F("# LOCKED CW (right opened first)  L=")
                               : F("# LOCKED CCW (left opened first)  L="));
    Serial.print(lidarL); Serial.print(F(" R=")); Serial.println(lidarR);
  } else if ((clockwiseMode ? openRevsR : openRevsL) < SIDE_OPEN_REVS) {
    return;
  }

  Serial.print(F("# turn: side open ")); Serial.println(turnSideMm());

  // Fallback finish measurement only — see the note at the declaration.
  float segCm = absEnc(readEncoder()) / TICKS_PER_CM;
  if (cornerCount == 0)                            firstSegmentCm = segCm;
  else if (cornerCount == 4)                     { fullStartStraightCm = segCm; haveFullStraight = true; }
  else if (cornerCount == 8 && haveFullStraight)   fullStartStraightCm = 0.5 * (fullStartStraightCm + segCm);
  if (haveFullStraight) {
    finalDistanceCm = max(0.0f, fullStartStraightCm - firstSegmentCm);
    fsTargetTicks   = (long)(finalDistanceCm * TICKS_PER_CM);
  }

  goState(STATE_TURN90);
}

// ============================================================================
// STATE: AVOID
//
// The Pi has already done the geometry. This state holds the heading it was
// given and hands back to HEADING the instant the Pi says NONE — no distance,
// no timer, just that one condition. Release is entirely the Pi's call (see
// control/supervisor.py): it comes as soon as the pillar is either confirmed
// clear or out of view, and updateHeadingPid(laneHeading) back in HEADING is
// what closes the gap — an ordinary P-loop convergence, nothing scheduled.
//
//   TRACK   the only non-NONE action sent any more. While a fresh TRACK frame
//           keeps arriving, RE-BASE the odometry every tick: avoidBaseTicks
//           tracks "now", so travelled never accumulates and avoidLegTicks/
//           AVOID_MAX_LEG_CM/AVOID_TIMEOUT_MS are inert. They exist purely as
//           a last-resort backstop for a genuinely stuck link (TRACK frames
//           stop arriving fresh, or the link goes stale) — not as the normal
//           way out. Do not read pLegMm as a target to reach; the Pi does not
//           send one any more (see BACKSTOP_LEG_MM in supervisor.py).
//
// No corner watching here: the car is yawed, so the side beams are not square
// to the walls. A corner that opens during an avoid is still open when HEADING
// resumes, and is counted there — including whatever progress the locked
// side's open-revolution count already had before this avoid started; see the
// suppressCornerReset note in headingStep().
// ============================================================================
void avoidStep() {
  if (!entered) {
    entered = true;
    phaseT0 = millis();
    avoidTargetHeading = pHeadingDeg;
    avoidBaseTicks = gOdoTicks;
    avoidLegTicks  = (long)(pLegMm * TICKS_PER_MM);
    resetHeadingPid();
    setMotorSpeed(avoidSpeed);
    Serial.print(F("# AVOID ")); Serial.print(pGreen ? F("GREEN->left ") : F("RED->right "));
    Serial.print(avoidTargetHeading); Serial.println(F(" deg"));
  }

  if (pAction == AVOID_NONE) {                 // the Pi says we are clear
    Serial.println(F("# avoid released"));
    avoidLockoutFrom = gOdoTicks;
    targetHeading = laneHeading;
    suppressCornerReset = true;        // don't lose progress made before this avoid
    goState(STATE_HEADING);
    return;
  }

  // A stale link means the last TRACK heading is not being refreshed any more.
  // Stop re-basing and let the backstop cap/timeout below take over, rather
  // than holding a frozen heading indefinitely with no one watching.
  if (pAction == AVOID_TRACK && !linkStale) {
    avoidTargetHeading = pHeadingDeg;
    avoidLegTicks = (long)(pLegMm * TICKS_PER_MM);
    avoidBaseTicks = gOdoTicks;
  }

  updateHeadingPid(avoidTargetHeading);
  setMotorSpeed(avoidSpeed);          // live: a mid-avoid slider change applies too

  long travelled = absEnc(gOdoTicks - avoidBaseTicks);
  bool capped  = travelled >= (long)(AVOID_MAX_LEG_CM * TICKS_PER_CM);
  bool timeout = (millis() - phaseT0) > AVOID_TIMEOUT_MS;

  if (capped || timeout) {
    if (capped)  Serial.println(F("# avoid capped (link not re-basing)"));
    if (timeout) Serial.println(F("# avoid timed out"));
    avoidLockoutFrom = gOdoTicks;
    targetHeading = laneHeading;
    suppressCornerReset = true;
    goState(STATE_HEADING);
  }
}

// ============================================================================
// STATE: TURN90  (eased arc, no settle, no lane correction)
// ============================================================================
void turn90Step() {
  if (!entered) {
    entered = true;
    cornerCount++;
    Serial.print(F("# TURN ")); Serial.print(cornerCount);
    Serial.print('/'); Serial.println(TARGET_CORNERS);
    turnAmount     = clockwiseMode ? 90.0 : -90.0;
    turnTarget     = wrapDeg(laneHeading - turnAmount);
    turnStartTicks = readEncoder();
    turnCapTicks   = (long)(120.0 * TICKS_PER_CM);
  }

  if (turnArcStep(turnTarget)) {
    laneHeading   = wrapDeg(laneHeading - turnAmount);
    targetHeading = laneHeading;
    zeroEncoder();                 // segment origin = this turn corner
    Serial.print(F("# lane heading ")); Serial.println(laneHeading);
    goState(STATE_HEADING);
  }
}

// ============================================================================
// STATE: FINISH  (stopped; the Pi can send the car round again)
//
// RERUN has to RISE while we sit here. rerunArmed is cleared on the way in and
// set by any CMD that is not RERUN, so:
//   - a 1 the Pi was already sending when the run ended does nothing until it
//     drops and comes back,
//   - holding 1 forever gives exactly one rerun, never run after run.
// The rerun goes through BOOT like a power-up: HELLO, 5 s countdown, and the
// finish line re-measured at GO from wherever the car sits now.
// ============================================================================
void finishStep() {
  if (!entered) {
    entered = true;
    rerunArmed = false;
    Serial.println(F("# FINISHED"));
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    digitalWrite(LED1_PIN, HIGH);
    digitalWrite(LED2_PIN, HIGH);
    digitalWrite(LED3_PIN, HIGH);
  }

  if (pCmd != CMD_RERUN) { rerunArmed = true; return; }
  if (!rerunArmed || linkStale) return;

  Serial.println(F("# RERUN from Pi"));
  goState(STATE_BOOT);
}

// ============================================================================
// STATE: STOPPED  (the Pi said STOP; RERUN leaves, exactly as from FINISH)
//
// Entered only through serviceCommands(). The motor was already cut there, the
// moment the command confirmed; this holds it and shows it on the LEDs. RERUN
// uses the same rising-edge rule as FINISH (rerunArmed is shared: only one of
// the two states can be current), and goes back through BOOT: HELLO, 5 s
// countdown, finish line re-measured at GO.
// ============================================================================
void stoppedStep() {
  if (!entered) {
    entered = true;
    rerunArmed = false;
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    digitalWrite(LED1_PIN, LOW);
    Serial.println(F("# STOPPED - RERUN to go again"));
  }
  // LED2/LED3 alternate: tells STOPPED apart from FINISH (all solid) at a glance.
  bool ph = ((millis() / 250) & 1);
  digitalWrite(LED2_PIN, ph ? HIGH : LOW);
  digitalWrite(LED3_PIN, ph ? LOW : HIGH);

  if (pCmd != CMD_RERUN) { rerunArmed = true; return; }
  if (!rerunArmed || linkStale) return;

  Serial.println(F("# RERUN from Pi"));
  goState(STATE_BOOT);
}

// ============================================================================
// COMMANDS  (PERCEPT byte 15; RERUN is handled inside finishStep/stoppedStep)
// ============================================================================
void serviceCommands() {
  if (pCmdRun < CMD_CONFIRM_FRAMES) return;

  if (pCmd == CMD_REBOOT) {
    setMotorSpeed(0);
    setServoAngle(SERVO_TRUE_STRAIGHT);
    Serial.println(F("# REBOOT from Pi"));
    Serial.flush();
    delay(50);                             // let the line leave; servo settles
    NVIC_SystemReset();                    // never returns
  }

  if (pCmd == CMD_STOP &&
      currentState != STATE_STOPPED && currentState != STATE_FINISH) {
    setMotorSpeed(0);                      // now, not on the next state step
    setServoAngle(SERVO_TRUE_STRAIGHT);
    Serial.println(F("# STOP from Pi"));
    finishReason = FINISH_BY_STOP_CMD;
    currentState = STATE_STOPPED;          // also abandons RECOVER: no return
    entered = false;
  }
}

// ============================================================================
// STATUS  (10 Hz, reporting only)
//
//  off type field                      off type field
//   2  u8   version (1)                 32  u16  lidar L mm (0xFFFF = far)
//   3  u8   state                       34  u16  lidar R mm (0xFFFF = far)
//   4  u8   flags   SF_*                36  u16  straight_mm      (HEADING)
//   5  u8   flags2  SF2_*               38  u16  corner_lockout_mm left
//   6  u8   pflags  PF_*                40  u16  avoid_lockout_mm left
//   7  u8   pCmd                        42  u16  segment_mm (since GO / turn)
//   8  u8   pCmdRun                     44  u16  finish_target_mm (odo fallback)
//   9  u8   runNumber                   46  u16  front_at_start_mm (0xFFFF)
//  10  u16  state_ms                    48  u16  phase_mm (AVOID/TURN90/RECOVER)
//  12  u16  countdown_ms  (BOOT)        50  u16  avoid_leg_mm     (AVOID, now
//  14  u16  grace_ms      (BOOT)                 always the backstop cap, not
//  16  u16  avoid_ms      (AVOID)                a target — see AVOID's comment)
//  18  i16  lane_heading   x10          52  u16  link_age_ms  (0xFFFF never)
//  20  i16  target_heading x10          54  u16  lidar_age_ms (0xFFFF never)
//  22  i16  servo_cmd x10 (servo deg)   56  u32  percept_frames
//  24  i16  motor_pwm                   60  u8   xor8 over 2..59
//  26  u8   openRevsL      27 u8 openRevsR
//  28  u8   recoverTries   29 u8 recoverReturnState
//  30  u8   finishReason   31 u8 reserved (0)
// ============================================================================
static inline uint16_t sat16(long v) {
  return v <= 0 ? 0 : (v >= 0xFFFE ? 0xFFFE : (uint16_t)v);   // 0xFFFF = invalid
}
static inline uint16_t farToInvalid(uint16_t v) { return v == LIDAR_FAR ? 0xFFFF : v; }
static inline long     ticksToMm(long t) { return (long)(t * MM_PER_TICK); }
static inline int16_t  decideg(float deg) {
  float v = deg * 10.0f;
  if (v >  32767.0f) v =  32767.0f;
  if (v < -32768.0f) v = -32768.0f;
  return (int16_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

void sendStatus() {
  const unsigned long now = millis();
  const bool inBoot    = currentState == STATE_BOOT;
  const bool inHeading = currentState == STATE_HEADING;
  const bool inAvoid   = currentState == STATE_AVOID;
  const bool inTurn    = currentState == STATE_TURN90;
  const bool inRecover = currentState == STATE_RECOVER;
  const bool parked    = currentState == STATE_FINISH || currentState == STATE_STOPPED;

  uint8_t f[STATUS_LEN];
  memset(f, 0, sizeof(f));
  f[0] = STATUS_SYNC0; f[1] = STATUS_SYNC1;
  f[2] = STATUS_VERSION;
  f[3] = (uint8_t)currentState;

  uint8_t fl = 0;
  if (fsmStarted)                  fl |= SF_FSM_STARTED;
  if (inBoot && phaseT0 != 0)      fl |= SF_BOOT_READY;
  if (lidarHold)                   fl |= SF_LIDAR_HOLD;
  if (linkStale)                   fl |= SF_LINK_STALE;
  if (startedBlind)                fl |= SF_BLIND;
  if (parked && rerunArmed)        fl |= SF_RERUN_ARMED;
  if (wallSeenL)                   fl |= SF_WALL_SEEN_L;
  if (wallSeenR)                   fl |= SF_WALL_SEEN_R;
  f[4] = fl;
  uint8_t fl2 = 0;
  if (realOpenL)                   fl2 |= SF2_REAL_OPEN_L;
  if (realOpenR)                   fl2 |= SF2_REAL_OPEN_R;
  if (LOCK_NEEDS_REAL_RETURN)      fl2 |= SF2_LOCK_NEEDS_REAL;
  f[5] = fl2;
  uint8_t pf = 0;
  if (pHello)                      pf |= PF_HELLO;
  if (pLidarOk)                    pf |= PF_LIDAR_OK;
  if (pCamOk)                      pf |= PF_CAM_OK;
  if (pGreen)                      pf |= PF_GREEN;
  pf |= (pAction & 0x03) << PF_ACTION_SHIFT;
  f[6] = pf;
  f[7] = pCmd;
  f[8] = pCmdRun;
  f[9] = runNumber;

  uint16_t stateMs   = sat16((long)(now - gStateSinceMs));
  uint16_t countdown = (inBoot && phaseT0 != 0)
                       ? sat16((long)START_DELAY_MS - (long)(now - phaseT0)) : 0;
  uint16_t grace     = (inBoot && fsmStarted && phaseT0 == 0)
                       ? sat16((long)CAMERA_GRACE_MS - (long)(now - firstFrameMs)) : 0;
  uint16_t avoidMs   = inAvoid ? sat16((long)(now - phaseT0)) : 0;

  float   target  = inAvoid ? avoidTargetHeading : (inTurn ? turnTarget : targetHeading);
  int16_t laneDd  = decideg(laneHeading);
  int16_t tgtDd   = decideg(target);
  int16_t servoDd = decideg(lastServoCmd);
  int16_t pwm     = (int16_t)gMotorPwm;

  f[26] = openRevsL;
  f[27] = openRevsR;
  f[28] = (uint8_t)recoverTries;
  f[29] = (uint8_t)recoverReturnState;
  f[30] = finishReason;

  uint16_t lidL       = farToInvalid(lidarL);
  uint16_t lidR       = farToInvalid(lidarR);
  uint16_t straightMm = inHeading ? sat16(ticksToMm(absEnc(gOdoTicks - dcBaseTicks))) : 0;
  uint16_t cornerLock = inHeading ? sat16(ticksToMm(dcLockoutTicks - absEnc(readEncoder()))) : 0;
  uint16_t avoidLock  = sat16(ticksToMm((long)(POST_AVOID_LOCKOUT_CM * TICKS_PER_CM)
                                        - absEnc(gOdoTicks - avoidLockoutFrom)));
  uint16_t segmentMm  = sat16(ticksToMm(absEnc(readEncoder())));
  uint16_t finTarget  = sat16(ticksToMm(fsTargetTicks));
  uint16_t frontStart = farToInvalid(frontAtStartMm);
  uint16_t phaseMm = 0, legMm = 0;
  if (inAvoid) {
    phaseMm = sat16(ticksToMm(absEnc(gOdoTicks - avoidBaseTicks)));
    legMm   = sat16(ticksToMm(avoidLegTicks));
  } else if (inTurn) {
    phaseMm = sat16(ticksToMm(absEnc(readEncoder() - turnStartTicks)));
  } else if (inRecover && entered) {
    phaseMm = sat16(ticksToMm(absEnc(gOdoTicks - recoverBaseTicks)));
  }
  uint16_t linkAge  = perceptFrames ? sat16((long)(now - linkLastMs))  : 0xFFFF;
  uint16_t lidarAge = lidarLastMs   ? sat16((long)(now - lidarLastMs)) : 0xFFFF;
  uint32_t frames   = perceptFrames;

  memcpy(f + 10, &stateMs,    2);
  memcpy(f + 12, &countdown,  2);
  memcpy(f + 14, &grace,      2);
  memcpy(f + 16, &avoidMs,    2);
  memcpy(f + 18, &laneDd,     2);
  memcpy(f + 20, &tgtDd,      2);
  memcpy(f + 22, &servoDd,    2);
  memcpy(f + 24, &pwm,        2);
  memcpy(f + 32, &lidL,       2);
  memcpy(f + 34, &lidR,       2);
  memcpy(f + 36, &straightMm, 2);
  memcpy(f + 38, &cornerLock, 2);
  memcpy(f + 40, &avoidLock,  2);
  memcpy(f + 42, &segmentMm,  2);
  memcpy(f + 44, &finTarget,  2);
  memcpy(f + 46, &frontStart, 2);
  memcpy(f + 48, &phaseMm,    2);
  memcpy(f + 50, &legMm,      2);
  memcpy(f + 52, &linkAge,    2);
  memcpy(f + 54, &lidarAge,   2);
  memcpy(f + 56, &frames,     4);
  f[60] = xor8(f + 2, STATUS_LEN - 3);

  Serial.write(f, STATUS_LEN);
}

void trackState() {
  if (currentState != gTrackedState) {
    gTrackedState = currentState;
    gStateSinceMs = millis();
  }
}

// ============================================================================
// MAIN
// ============================================================================
void initHardware() {
  pinMode(MOT_RPWM_PIN, OUTPUT);
  pinMode(MOT_LPWM_PIN, OUTPUT);
  setMotorSpeed(0);

  pinMode(LED1_PIN, OUTPUT); pinMode(LED2_PIN, OUTPUT); pinMode(LED3_PIN, OUTPUT);
  pinMode(BTN_START_PIN, INPUT_PULLUP);
  digitalWrite(LED1_PIN, LOW); digitalWrite(LED2_PIN, LOW); digitalWrite(LED3_PIN, LOW);

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
  gLastRawEnc = readEncoder();

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

void setup() {
  Serial.begin(115200);
  initHardware();
}

void loop() {
  serviceSensors();
  serviceCommands();                      // STOP / REBOOT from PERCEPT byte 15

  static uint32_t lastTelem = 0, lastStatus = 0;
  uint32_t now = millis();
  if (now - lastTelem >= TELEM_PERIOD_MS) { lastTelem = now; sendTelemetry(); }
  if (now - lastStatus >= STATUS_PERIOD_MS) { lastStatus = now; sendStatus(); }

  if (!fsmStarted) {
    if (perceptFrames == 0) {
      digitalWrite(LED1_PIN, ((now / 500) & 1) ? HIGH : LOW);
      return;
    }
    fsmStarted = true;
    firstFrameMs = now;
    Serial.println(F("# first PERCEPT frame received"));
  }

  // Wall panic overlay. Disabled when the lidar is stale — there is no front
  // distance to act on, and a frozen value would latch it.
  if (!lidarStale &&
      currentState != STATE_RECOVER && currentState != STATE_BOOT &&
      currentState != STATE_FINISH && currentState != STATE_STOPPED &&
      recoverTries < RECOVER_MAX_TRIES &&
      lidarF <= WALL_PANIC_MM) {
    enterRecovery();
  }

  if (currentState != STATE_BOOT && currentState != STATE_FINISH &&
      currentState != STATE_STOPPED) {
    digitalWrite(LED1_PIN, lidarStale ? (((now / 100) & 1) ? HIGH : LOW) : HIGH);
  }
  if (currentState != STATE_FINISH && currentState != STATE_STOPPED) {
    digitalWrite(LED2_PIN, (dirLocked &&  clockwiseMode) ? HIGH : LOW);
    digitalWrite(LED3_PIN, (dirLocked && !clockwiseMode) ? HIGH : LOW);
  }

  switch (currentState) {
    case STATE_BOOT:    bootStep();    break;
    case STATE_HEADING: headingStep(); break;
    case STATE_AVOID:   avoidStep();   break;
    case STATE_TURN90:  turn90Step();  break;
    case STATE_RECOVER: recoverStep(); break;
    case STATE_FINISH:  finishStep();  break;
    case STATE_STOPPED: stoppedStep(); break;
  }

  trackState();
}
