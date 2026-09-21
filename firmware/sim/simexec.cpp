// simexec.cpp - the REAL firmware/ObstacleRound.cpp, compiled for the host.
//
// The hardware is replaced by the stubs in this folder; the firmware source is
// #included unmodified, so what this exercises is what gets flashed. The C
// functions below are what tests/test_firmware_sim.py drives through ctypes:
// it feeds DRIVE frames and parameter lines in, and reads back the exact byte
// stream the firmware wrote - TELEM frames and ASCII lines interleaved, as
// they would arrive at the Pi.
//
//   ./build.sh            -> libexec.so
#include "Arduino.h"
#include "Servo.h"
#include "Wire.h"
#include "SPI.h"
#include "SparkFun_BNO08x_Arduino_Library.h"
#include "Adafruit_TCS34725.h"

std::string g_tx;
int g_reset_requested = 0;

static unsigned long g_T = 0;
static std::string g_rx;
static size_t g_pos = 0;
static float g_heading = 0;
static bool  g_imu = false;
static bool  g_imu_streaming = false;   // setup(): the IMU free-runs
static int   g_floor = 0;
static float g_servo = 76.5;
static int   g_pwmF = 0, g_pwmR = 0;
static int   g_tx_room = 4096;

unsigned long millis() { return g_T; }
void delay(unsigned long ms) { g_T += ms; }
void pinMode(int, int) {}
void digitalWrite(int, int) {}
void analogWrite(int p, int v) { if (p == PA2) g_pwmF = v; if (p == PA3) g_pwmR = v; }

int SerialT::available() { return g_pos < g_rx.size(); }
int SerialT::read() { return (uint8_t)g_rx[g_pos++]; }
int SerialT::availableForWrite() { return g_tx_room; }
SerialT Serial;
TIM_TypeDef tim5; TIM_TypeDef *TIM5 = &tim5; void *GPIOA;

void Servo::attach(int, int, int) {}
void Servo::writeMicroseconds(int us) { g_servo = (us - 500) / 2000.0f * 180.0f; }
SPIClass::SPIClass(int, int, int) {}
void SPIClass::begin() {}

// colour sensor over Wire: C, R, G, B little-endian. floor 1 = orange,
// 2 = blue, anything else = the white mat - the bench readings from the
// firmware header (white pR 47 / orange pR 69 / blue pB 27).
TwoWire Wire; static uint8_t g_wb[8]; static int g_wi = 8;
void TwoWire::beginTransmission(uint8_t) {}
void TwoWire::write(uint8_t) {}
int  TwoWire::endTransmission() { return 0; }
void TwoWire::requestFrom(uint8_t, uint8_t) {
  uint16_t c = 1000, r, g, b;
  if (g_floor == 1)      { r = 690; g = 200; b = 110; }
  else if (g_floor == 2) { r = 300; g = 350; b = 350; }
  else                   { r = 470; g = 340; b = 190; }
  uint16_t v[4] = {c, r, g, b};
  for (int i = 0; i < 4; i++) { g_wb[2 * i] = v[i] & 0xff; g_wb[2 * i + 1] = v[i] >> 8; }
  g_wi = 0;
}
int  TwoWire::available() { return 8 - g_wi; }
int  TwoWire::read() { return g_wb[g_wi++]; }
void TwoWire::setSCL(int) {}
void TwoWire::setSDA(int) {}
void TwoWire::begin() {}
void TwoWire::setClock(long) {}

bool  BNO08x::wasReset() { return false; }
void  BNO08x::enableGameRotationVector() {}
bool  BNO08x::getSensorEvent() { bool e = g_imu || g_imu_streaming; g_imu = false; return e; }
int   BNO08x::getSensorEventID() { return SENSOR_REPORTID_GAME_ROTATION_VECTOR; }
float BNO08x::getQuatI() { return 0; }
float BNO08x::getQuatJ() { return 0; }
float BNO08x::getQuatK() { return sinf(g_heading * PI / 360.0f); }
float BNO08x::getQuatReal() { return cosf(g_heading * PI / 360.0f); }
bool  BNO08x::beginSPI(int, int, int, long, SPIClass &) { g_imu = true; return true; }
Adafruit_TCS34725::Adafruit_TCS34725(int, int) {}
bool Adafruit_TCS34725::begin() { return true; }

#include "../ObstacleRound.cpp"

extern "C" {

void fw_setup() {
  g_T = 0; g_heading = 0; g_imu = true; g_tx.clear(); g_rx.clear(); g_pos = 0;
  g_reset_requested = 0; TIM5->CNT = 0;
  g_imu_streaming = true;      // a real BNO08x keeps reporting while setup() waits
  setup();
  g_imu_streaming = false;
}

// One loop() pass at time t (ms). heading is the IMU's absolute yaw (the
// firmware applies its own sign and zero), dticks the encoder counts moved
// forward since the last call, imu=1 means a fresh IMU event this pass.
// rx/rx_len are bytes arriving from the Pi - binary-safe.
void fw_step(unsigned long t, float heading, long dticks, int floor, int imu,
             const uint8_t *rx, int rx_len) {
  g_T = t; g_heading = heading; g_imu = imu; g_floor = floor;
  TIM5->CNT = (uint32_t)((int32_t)TIM5->CNT - (int32_t)dticks);
  if (rx && rx_len > 0) {
    g_rx.erase(0, g_pos); g_pos = 0;
    g_rx.append((const char *)rx, rx_len);
  }
  loop();
}

// Everything written since the last call. *n gets the length (binary-safe).
const uint8_t *fw_tx(int *n) {
  static std::string s;
  s = g_tx; g_tx.clear();
  *n = (int)s.size();
  return (const uint8_t *)s.data();
}

void  fw_set_tx_room(int n) { g_tx_room = n; }
float fw_servo()       { return g_servo; }
int   fw_motor()       { return g_pwmF - g_pwmR; }
int   fw_resets()      { return g_reset_requested; }
int   fw_recovering()  { return recovering ? 1 : 0; }
int   fw_link_stale()  { return linkStale ? 1 : 0; }

}
