// lostexec.cpp - the lost-pillar chain out of the REAL firmware, on the host.
//
// simexec.cpp drives the whole executor firmware over its binary wire and no
// longer builds: firmware/ObstacleRound.cpp is now the build that runs the FSM
// on the STM32. Rewriting that harness for this architecture is a job of its
// own. This is the small version: it includes the same source unmodified and
// exposes just the ANTi lost-pillar chain, so tests/test_lost_chain.py can
// check the ported logic against the code that actually gets flashed rather
// than against a transcription of it.
//
//   ./build_lost.sh       -> liblost.so

#include "Arduino.h"
#include "Servo.h"
#include "Wire.h"
#include "SPI.h"
#include "SparkFun_BNO08x_Arduino_Library.h"
#include "Adafruit_TCS34725.h"

// ---- the hardware, doing nothing ----
// The chain reads no sensor and drives nothing, so every stub here can be
// inert. It exists only because the headers declare these out of line; a
// harness that exercises the FSM would have to model them properly, and
// simexec.cpp is where that belongs.

std::string g_tx;
int g_reset_requested = 0;
SerialT Serial;
TwoWire Wire;
static TIM_TypeDef g_tim5 = {0};
TIM_TypeDef *TIM5 = &g_tim5;
void *GPIOA = nullptr;

static unsigned long g_ms = 0;

unsigned long millis() { return g_ms; }
void delay(unsigned long ms) { g_ms += ms; }
void pinMode(int, int) {}
void digitalWrite(int, int) {}
void analogWrite(int, int) {}

int SerialT::available() { return 0; }
int SerialT::read() { return -1; }
int SerialT::availableForWrite() { return 256; }

void TwoWire::beginTransmission(uint8_t) {}
void TwoWire::write(uint8_t) {}
int  TwoWire::endTransmission() { return 0; }
void TwoWire::requestFrom(uint8_t, uint8_t) {}
int  TwoWire::available() { return 0; }
int  TwoWire::read() { return 0; }
void TwoWire::setSCL(int) {}
void TwoWire::setSDA(int) {}
void TwoWire::begin() {}
void TwoWire::setClock(long) {}

void Servo::attach(int, int, int) {}
void Servo::writeMicroseconds(int) {}

SPIClass::SPIClass(int, int, int) {}
void SPIClass::begin() {}

bool  BNO08x::wasReset() { return false; }
void  BNO08x::enableGameRotationVector() {}
bool  BNO08x::getSensorEvent() { return false; }
int   BNO08x::getSensorEventID() { return 0; }
float BNO08x::getQuatI() { return 0.0f; }
float BNO08x::getQuatJ() { return 0.0f; }
float BNO08x::getQuatK() { return 0.0f; }
float BNO08x::getQuatReal() { return 1.0f; }
bool  BNO08x::beginSPI(int, int, int, long, SPIClass &) { return false; }

Adafruit_TCS34725::Adafruit_TCS34725(int, int) {}
bool Adafruit_TCS34725::begin() { return false; }

#include "../ObstacleRound.cpp"

extern "C" {

void lost_reset() { clearLost(); }

// One new Pi frame carrying `colour` (VIS_RED / VIS_GREEN / VIS_NONE).
void lost_feed(int colour) {
    visColor = colour;
    updateLost();
}

int   lost_phase()  { return lostPhase; }
int   lost_frames() { return (int)lostFrames; }
int   lost_colour() { return lostColor; }
float lost_yaw()    { return lostYaw(); }

void lost_tune(int search, int commit, int abandon, float sdeg, float cdeg) {
    LOST_SEARCH_FRAMES  = (uint8_t)search;
    LOST_COMMIT_FRAMES  = (uint8_t)commit;
    LOST_ABANDON_FRAMES = (uint8_t)abandon;
    LOST_SEARCH_DEG     = sdeg;
    LOST_COMMIT_DEG     = cdeg;
}

// The firmware's own enum values, so the test never restates them.
int k_vis_red()      { return VIS_RED; }
int k_vis_green()    { return VIS_GREEN; }
int k_vis_none()     { return VIS_NONE; }
int k_lost_none()    { return LOST_NONE; }
int k_lost_search()  { return LOST_SEARCH; }
int k_lost_commit()  { return LOST_COMMIT; }

}
