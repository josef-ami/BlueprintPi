#pragma once
#include <stdint.h>
struct TwoWire{void beginTransmission(uint8_t); void write(uint8_t); int endTransmission(); void requestFrom(uint8_t,uint8_t); int available(); int read(); void setSCL(int); void setSDA(int); void begin(); void setClock(long);}; extern TwoWire Wire;
