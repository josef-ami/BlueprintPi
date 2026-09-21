#pragma once
#include "SPI.h"
#define SENSOR_REPORTID_GAME_ROTATION_VECTOR 8
struct BNO08x{bool wasReset(); void enableGameRotationVector(); bool getSensorEvent(); int getSensorEventID(); float getQuatI(); float getQuatJ(); float getQuatK(); float getQuatReal(); bool beginSPI(int,int,int,long,SPIClass&);};
