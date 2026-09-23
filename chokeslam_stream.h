// ============================================================
// chokeslam_stream.h  -  portable chokeSLAM STM32 -> Pi stream
//
// Drop this file next to any STM32duino sketch (same folder as the .ino)
// and add three lines. It does ONE thing: sends the agreed protocol line
// (docs/CHANGES.md section 9.1, decisions #17, #18, #49)
//
//     $IMU,<seq>,<t_ms>,<enc>,<yaw>\n        100 Hz
//
// It touches no pins, timers, SPI, IMU or encoder hardware. Your sketch
// already reads those; you just hand the values over.
//
// ---------------- HOW TO USE ----------------
//   #include "chokeslam_stream.h"
//
//   void setup() {
//     ...
//     Serial.begin(115200);            // USB CDC (baud ignored)
//   }
//
//   void loop() {
//     ...your code...
//     chokeslamStreamPoll(encoderCount, yawDeg, yawValid);
//   }
//
//   Call it every loop pass (faster than 100 Hz). It paces itself and
//   sends at most one line per 10 ms. It never blocks.
//
// ---------------- WHAT YOU MUST PASS ----------------
//   encoderCount  int32  cumulative count since power-on, forward = +,
//                        NEVER reset (not per lap, not per turn)
//   yawDeg        float  Game Rotation Vector yaw in degrees AS THE CHIP
//                        REPORTS IT: no zero offset, no sign flip, no
//                        unwrapping (-180..180 is fine). The Pi applies
//                        IMU_YAW_SIGN = -1 itself.
//   yawValid      bool   true only while yawDeg is a real, recent reading
//                        (e.g. a report arrived in the last 100 ms). While
//                        false, NO line is sent, so the Pi sees the link go
//                        STALE instead of integrating a frozen heading.
//
// ---------------- RULES FOR THE HOST SKETCH ----------------
//   * Nothing else may be written to the stream port (default Serial):
//     no Serial.print debug, no '#' lines (decision #49). The Pi counts any
//     other line as bad. Move debug output to another port or remove it.
//   * USB support must be "CDC (generic 'Serial' supersede U(S)ART)" so that
//     Serial is the native USB port (/dev/ttyACM* on the Pi).
//   * To use a different port, define CHOKESLAM_PORT before the include:
//         #define CHOKESLAM_PORT SerialUSB
//         #include "chokeslam_stream.h"
//
// ---------------- LINE FIELDS ----------------
//   seq   uint32  +1 every due line. A line dropped because USB was busy
//                 still uses its seq, so the Pi counts it as lost. Lines
//                 not sent because yawValid was false do NOT use a seq.
//   t_ms  uint32  HAL_GetTick() at the moment the line is built
//   enc   int32   as passed
//   yaw   2 decimals, formatted with integers (no printf-float needed)
// ============================================================
#ifndef CHOKESLAM_STREAM_H
#define CHOKESLAM_STREAM_H

#include <Arduino.h>
#include <math.h>

#ifndef CHOKESLAM_PORT
#define CHOKESLAM_PORT Serial
#endif

#ifndef CHOKESLAM_PERIOD_MS
#define CHOKESLAM_PERIOD_MS 10u        // 100 Hz
#endif

namespace chokeslam_stream_detail {
  static uint32_t seq        = 0;
  static uint32_t nextSendMs = 0;
  static bool     started    = false;
  static uint32_t lastSentMs = 0;
  static bool     everSent   = false;
}

// Builds one protocol line into buf. Returns its length, or 0 on failure.
static inline int chokeslamFormatLine(char *buf, size_t size, uint32_t seq,
                                      uint32_t t_ms, int32_t enc, float yawDeg) {
  int32_t  y100 = (int32_t)lroundf(yawDeg * 100.0f);
  const char *sign = (y100 < 0) ? "-" : "";
  uint32_t ay = (uint32_t)(y100 < 0 ? -y100 : y100);
  int n = snprintf(buf, size, "$IMU,%lu,%lu,%ld,%s%lu.%02lu\n",
                   (unsigned long)seq, (unsigned long)t_ms, (long)enc, sign,
                   (unsigned long)(ay / 100), (unsigned long)(ay % 100));
  return (n > 0 && n < (int)size) ? n : 0;
}

// Call every loop pass. Returns true if a line actually went out.
static inline bool chokeslamStreamPoll(int32_t encoderCount, float yawDeg, bool yawValid) {
  using namespace chokeslam_stream_detail;
  uint32_t now = millis();
  if (!started) { nextSendMs = now; started = true; }
  if ((int32_t)(now - nextSendMs) < 0) return false;          // not due yet

  nextSendMs += CHOKESLAM_PERIOD_MS;
  // a long stall resyncs instead of bursting catch-up lines
  if ((int32_t)(now - nextSendMs) > (int32_t)CHOKESLAM_PERIOD_MS)
    nextSendMs = now + CHOKESLAM_PERIOD_MS;

  if (!yawValid || !isfinite(yawDeg)) return false;           // no line, no seq

  seq++;                                                      // used even if dropped
  char line[64];
  int n = chokeslamFormatLine(line, sizeof(line), seq, HAL_GetTick(), encoderCount, yawDeg);
  if (n == 0) return false;
  // never block: if the Pi isn't reading, drop the line (the seq gap shows it)
  if (CHOKESLAM_PORT && CHOKESLAM_PORT.availableForWrite() >= n) {
    CHOKESLAM_PORT.write((const uint8_t *)line, n);
    lastSentMs = millis();
    everSent = true;
    return true;
  }
  return false;
}

// Optional: true if a line went out within the last `withinMs` (for a
// status LED). Not needed for the protocol.
static inline bool chokeslamStreamHostOk(uint32_t withinMs = 200) {
  using namespace chokeslam_stream_detail;
  return everSent && (millis() - lastSentMs) <= withinMs;
}

#endif  // CHOKESLAM_STREAM_H
