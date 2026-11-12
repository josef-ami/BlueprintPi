// RPLIDAR C1 -> USB serial stream, ESP32 DevKit V1. Pair it with lidar_view.py on the PC.
// Do NOT open the Arduino Serial Monitor with this sketch: the data is binary, at 921600 baud.
//
// Wiring: lidar TX (yellow) -> GPIO16 (RX2)   lidar RX (green) -> GPIO17 (TX2)
//         lidar VCC (red) -> 5 V   lidar GND (black) -> GND
//
// Output, one frame per revolution (all numbers little-endian):
//   0xAA 0x55 | count u16 | count x { angle u16 (0.1 degree), distance u16 (mm, 0 = no return) } | xor u8
//   The xor byte is the XOR of the count bytes and all point bytes.
// Commands from the PC: 'x' stops the scan and the motor, 'g' starts it again.

#include <Arduino.h>

HardwareSerial& L = Serial2;
const int LIDAR_RX = 16, LIDAR_TX = 17;
const uint32_t LIDAR_BAUD = 460800;
const uint32_t PC_BAUD = 921600;

const int MAX_PTS = 900;                 // the C1 gives about 510 per revolution at 10 rev/s
struct Pt { uint16_t ang, dist; };
Pt pts[MAX_PTS];
int nPts = 0;
uint8_t frameBuf[4 + MAX_PTS * 4 + 1];

uint8_t pk[5]; int pn = 0;
bool scanning = false;
uint32_t lastData = 0, lastTry = 0;

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

void sendFrame() {
  uint16_t n = nPts;
  frameBuf[0] = 0xAA; frameBuf[1] = 0x55;
  frameBuf[2] = n & 0xFF; frameBuf[3] = n >> 8;
  int k = 4;
  for (int i = 0; i < n; i++) {
    frameBuf[k++] = pts[i].ang & 0xFF;  frameBuf[k++] = pts[i].ang >> 8;
    frameBuf[k++] = pts[i].dist & 0xFF; frameBuf[k++] = pts[i].dist >> 8;
  }
  uint8_t x = 0;
  for (int i = 2; i < k; i++) x ^= frameBuf[i];
  frameBuf[k++] = x;
  Serial.write(frameBuf, k);
}

void setup() {
  Serial.setTxBufferSize(4096);           // must come before begin()
  Serial.begin(PC_BAUD);
  L.setRxBufferSize(4096);                // must come before begin()
  L.begin(LIDAR_BAUD, SERIAL_8N1, LIDAR_RX, LIDAR_TX);
  delay(1500);
  scanning = startScan();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'x') { sendCmd(0x25); scanning = false; }
    if (c == 'g' && !scanning) scanning = startScan();
  }
  if (!scanning) return;

  while (L.available()) {
    pk[pn++] = L.read();
    if (pn < 5) continue;
    bool S = pk[0] & 1, nS = (pk[0] >> 1) & 1, C = pk[1] & 1;
    if (S != nS && C) {                                   // valid 5-byte sample
      uint8_t quality = pk[0] >> 2;
      uint16_t angQ6 = (((uint16_t)pk[2] << 8) | pk[1]) >> 1;      // degrees * 64
      uint16_t distQ2 = ((uint16_t)pk[4] << 8) | pk[3];            // mm * 4
      if (S) {                                            // start of a new revolution
        if (nPts >= 20) sendFrame();                      // skip the partial first one
        nPts = 0;
      }
      if (nPts < MAX_PTS) {
        pts[nPts].ang = (uint16_t)((angQ6 * 10UL + 32) / 64);
        pts[nPts].dist = (quality > 0) ? (distQ2 + 2) / 4 : 0;
        nPts++;
      }
      lastData = millis();
      pn = 0;
    } else {                                              // lost sync: slide by one byte
      memmove(pk, pk + 1, 4); pn = 4;
    }
  }
  if (millis() - lastData > 3000 && millis() - lastTry > 3000) {   // data stopped: restart the scan
    lastTry = millis();
    scanning = startScan();
  }
}
