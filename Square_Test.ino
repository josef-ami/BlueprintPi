// Square with IMU-controlled 90 degree turns. ESP32 DevKit V1, SparkFun BNO08x library, 2x BTS7960.
// Straight sides are still timed (DRIVE_MS); only the turns use the IMU yaw.
// Serial Monitor: 115200. Press the start button to run; press again to repeat. Floor, with room around it.
//
// IMU wiring (same as the SPI example): SCK 18, MISO 19, MOSI 23, CS 22, INT 21, RST 13
// Motors: left RPWM/LPWM 26/25 (swapped, it was reversed), right 27/33

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
const int PIN_LED = 4;    // LED: blinks = ready, solid = running

const int L_FWD = 26, L_BACK = 25;
const int R_FWD = 27, R_BACK = 33;

// ---- tune these
const int DRIVE_SPEED = 65;            // 0..255 (half of before; raise if the robot does not move)
const uint32_t DRIVE_MS = 1000;        // side length (half of before)
const bool YAW_RISES_TURNING_RIGHT = false;  // if the first turn runs away, the sketch aborts and tells you to flip this
const float TURN_DEG = 90.0f;
const float KP = 3.0f;                 // duty per degree of error
const float KD = 0.35f;                // duty per (degree/second): brakes the turn before the target
const int TURN_MAX = 130;
const int TURN_MIN = 80;               // lowest duty that still turns the wheels
const float DEADBAND_DEG = 3.0f;       // inside this the motors are OFF and it is never corrected again
const float STOP_RATE_DPS = 20.0f;     // ...once it has also slowed below this
const uint32_t TURN_TIMEOUT_MS = 4000;

BNO08x imu;
float yaw = 0;                         // degrees, from the game rotation vector (no magnetometer, drifts slowly)
bool haveYaw = false;
float yawRate = 0;                     // deg/s, filtered
uint32_t yawT = 0;
float wrap(float a);

void updateYaw() {
  if (imu.wasReset()) { Serial.println("IMU reset, re-enabling reports"); imu.enableGameRotationVector(20); }
  for (int n = 0; n < 10 && imu.getSensorEvent(); n++) {
    if (imu.getSensorEventID() == SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      float qi = imu.getQuatI(), qj = imu.getQuatJ(), qk = imu.getQuatK(), qr = imu.getQuatReal();
      float prev = yaw; uint32_t nowT = millis();
      yaw = atan2f(2.0f * (qr * qk + qi * qj), 1.0f - 2.0f * (qj * qj + qk * qk)) * RAD_TO_DEG;
      if (haveYaw && nowT > yawT) yawRate = 0.7f * yawRate + 0.3f * (wrap(yaw - prev) * 1000.0f / (nowT - yawT));
      yawT = nowT; haveYaw = true;
    }
  }
}

float wrap(float a) { while (a > 180) a -= 360; while (a < -180) a += 360; return a; }

void side(int fwdPin, int backPin, int v) {
  if (v >= 0) { analogWrite(backPin, 0); analogWrite(fwdPin, v); }
  else        { analogWrite(fwdPin, 0);  analogWrite(backPin, -v); }
}
void motors(int l, int r) { side(L_FWD, L_BACK, l); side(R_FWD, R_BACK, r); }

void driveTimed(int v, uint32_t ms) {
  motors(v, v);
  uint32_t t = millis();
  while (millis() - t < ms) { updateYaw(); delay(2); }
  motors(0, 0);
  uint32_t p = millis();
  while (millis() - p < 400) { updateYaw(); delay(2); }
}

// Turn right by deg (negative = left). PD control with a deadband; returns false if it had to give up.
bool turnRight(float deg) {
  const float s = YAW_RISES_TURNING_RIGHT ? 1.0f : -1.0f;
  float target = wrap(yaw + s * deg);
  uint32_t t0 = millis();
  bool wasPositive = (deg > 0);                         // true while we still have to turn right
  Serial.printf("turn: yaw %.1f -> target %.1f\n", yaw, target);
  while (millis() - t0 < TURN_TIMEOUT_MS) {
    updateYaw();
    float e = wrap(target - yaw);
    float r = e * s;                                    // >0 means "still need to turn right"
    float rate = yawRate * s;                           // >0 means "turning right now"
    if (millis() - t0 > 800 && fabsf(e) > fabsf(deg) + 25) {
      motors(0, 0);
      Serial.println("ABORT: the error is growing. Flip YAW_RISES_TURNING_RIGHT and run again.");
      return false;
    }
    bool crossed = (wasPositive && r <= 0) || (!wasPositive && r >= 0);    // reached or passed the target
    if (fabsf(r) < DEADBAND_DEG || crossed) {
      motors(0, 0);                                     // stop driving; no more corrections, so it cannot hunt
      if (fabsf(rate) < STOP_RATE_DPS) {
        delay(150);
        updateYaw();
        Serial.printf("turn done, yaw %.1f (error %.1f)\n", yaw, wrap(target - yaw));
        return true;
      }
    } else {
      float u = KP * r - KD * rate;                     // PD: the D term brakes as it speeds up
      int v = constrain((int)fabsf(u), TURN_MIN, TURN_MAX);
      if (u > 0) motors(v, -v); else motors(-v, v);
    }
    delay(5);
  }
  motors(0, 0);
  Serial.printf("turn timed out, yaw %.1f\n", yaw);
  return false;
}

void setup() {
  Serial.begin(115200);
  int pins[] = {L_FWD, L_BACK, R_FWD, R_BACK};
  for (int p : pins) analogWrite(p, 0);
  delay(1000);

  Serial.println("\n=== square_imu ===");
  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, -1);
  Serial.println("step 1: beginSPI");
  if (!imu.beginSPI(PIN_CS, PIN_INT, PIN_RST, 3000000, SPI)) {
    Serial.println("FAIL: BNO08x not found over SPI, stopping.");
    while (true) delay(1000);
  }
  Serial.println("step 2: beginSPI ok");
  delay(500);
  imu.enableGameRotationVector(20);     // same rate as the version that worked
  delay(100);
  imu.getSensorEvent();
  uint32_t t = millis();
  while (!haveYaw && millis() - t < 3000) { updateYaw(); delay(5); }
  if (!haveYaw) { Serial.println("FAIL: no yaw from the IMU in 3 s, stopping."); while (true) delay(1000); }
  Serial.printf("step 3: yaw %.1f, IMU running\n", yaw);

  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);
}

void waitForButton() {
  Serial.println("ready: press the start button (keep the robot still)");
  while (digitalRead(PIN_BTN) == LOW) { updateYaw(); delay(5); }       // wait for any old press to end
  while (true) {
    updateYaw();
    static uint32_t lastP = 0;
    if (millis() - lastP > 500) { lastP = millis(); Serial.printf("yaw %.1f\n", yaw); }   // watch it change when you rotate the robot
    digitalWrite(PIN_LED, (millis() / 300) % 2);                       // blink while waiting
    if (digitalRead(PIN_BTN) == LOW) {
      delay(25);                                                       // debounce
      if (digitalRead(PIN_BTN) == LOW) break;
    }
    delay(5);
  }
  while (digitalRead(PIN_BTN) == LOW) { updateYaw(); delay(5); }       // wait for release
  digitalWrite(PIN_LED, HIGH);
  uint32_t t = millis();
  while (millis() - t < 1000) { updateYaw(); delay(5); }               // let the robot settle after the press
}

void runSquare() {
  for (int i = 1; i <= 4; i++) {
    Serial.printf("side %d\n", i);
    driveTimed(DRIVE_SPEED, DRIVE_MS);
    if (!turnRight(TURN_DEG)) { Serial.println("stopped"); return; }
    uint32_t p = millis();
    while (millis() - p < 300) { updateYaw(); delay(2); }
  }
  Serial.println("done");
}

void loop() {
  waitForButton();
  runSquare();
  motors(0, 0);
  digitalWrite(PIN_LED, LOW);
}
