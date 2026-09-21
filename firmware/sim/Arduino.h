// Host stand-in for the Arduino / STM32duino core - just enough of it for
// firmware/ObstacleRound.cpp to compile with g++ and run on a PC.
//
// Adapted from the obstacle-round simulator's stubs. Additions for the
// executor firmware: a binary Serial.write() (TELEM frames), Serial.flush(),
// and NVIC_SystemReset(), which records the reset instead of performing one.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <sstream>
#define F(x) x
#define PI 3.14159265f
#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT_PULLUP 2
enum { PA0,PA1,PA2,PA3,PA4,PA5,PA6,PA7,PA8,PB0,PB1,PB6,PB7,PB8,PB12,PB13,PB14,PB15 };
template<class T,class L,class H> T constrain(T x,L l,H h){return x<l?l:(x>h?h:x);}
unsigned long millis(); unsigned long micros();
void delay(unsigned long); void pinMode(int,int);
void digitalWrite(int,int); void analogWrite(int,int);

// Everything the firmware sends - ASCII lines and binary frames alike - lands
// in one byte string, in order, exactly as it would on the USB CDC port.
extern std::string g_tx;
struct SerialT {
  void begin(long) {}
  int available();
  int read();
  template<class T> void print(T v) { std::ostringstream o; o << v; g_tx += o.str(); }
  template<class T> void println(T v) { print(v); g_tx += "\n"; }
  void println() { g_tx += "\n"; }
  void print(float v, int d) { std::ostringstream o; o.setf(std::ios::fixed); o.precision(d); o << v; g_tx += o.str(); }
  void print(double v, int d) { print((float)v, d); }
  void println(float v, int d) { print(v, d); g_tx += "\n"; }
  void println(double v, int d) { print((float)v, d); g_tx += "\n"; }
  size_t write(const uint8_t *b, size_t n) { g_tx.append((const char *)b, n); return n; }
  size_t write(uint8_t c) { g_tx.push_back((char)c); return 1; }
  void flush() {}
  int availableForWrite();
};
extern SerialT Serial;

extern int g_reset_requested;
inline void NVIC_SystemReset() { g_reset_requested++; }

struct TIM_TypeDef { uint32_t CNT; }; extern TIM_TypeDef *TIM5;
#define __HAL_RCC_GPIOA_CLK_ENABLE()
#define __HAL_RCC_TIM5_CLK_ENABLE()
struct GPIO_InitTypeDef { uint32_t Pin, Mode, Pull, Speed, Alternate; };
enum { GPIO_PIN_0 = 1, GPIO_PIN_1 = 2, GPIO_MODE_AF_PP, GPIO_PULLUP, GPIO_SPEED_FREQ_HIGH,
       GPIO_AF2_TIM5, TIM_COUNTERMODE_UP, TIM_ENCODERMODE_TI12, TIM_ICPOLARITY_RISING,
       TIM_ICSELECTION_DIRECTTI, TIM_CHANNEL_ALL };
extern void *GPIOA; inline void HAL_GPIO_Init(void *, GPIO_InitTypeDef *) {}
struct TIM_Encoder_InitTypeDef { uint32_t EncoderMode, IC1Polarity, IC1Selection, IC2Polarity, IC2Selection; };
struct TIM_InitT { uint32_t Prescaler, CounterMode, Period; };
struct TIM_HandleTypeDef { TIM_TypeDef *Instance; TIM_InitT Init; };
inline void HAL_TIM_Encoder_Init(TIM_HandleTypeDef *, TIM_Encoder_InitTypeDef *) {}
inline void HAL_TIM_Encoder_Start(TIM_HandleTypeDef *, int) {}
