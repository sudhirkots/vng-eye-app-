// Balance Band firmware — Seeed XIAO nRF52840 Sense (2026-09-23)
// NOT YET COMPILED OR RUN ON A BOARD. See ../README.md for setup.
//
// Reads the on-board 6-axis IMU (LSM6DS3TR-C) at 100 Hz and streams every sample over Bluetooth LE
// using the Nordic UART Service (NUS), device name "BalanceBand".
//
// Each sample is one 20-byte frame, little-endian:
//   0xA5 0x5A | seq u8 | t_us u32 | ax ay az i16 (milli-g) | gx gy gz i16 (0.01 deg/s) | checksum u8
//   checksum = sum of bytes seq..gz, & 0xFF.   seq counts 0..255 so the app can count lost samples.
// Frames are sent in batches of 10 (10 notifications per second). The app parses them as a byte stream,
// so it does not matter how Bluetooth splits the packets.
//
// LED: green blink = waiting for the app; blue = connected and streaming; red = IMU not found.

#include <bluefruit.h>
#include <LSM6DS3.h>
#include <Wire.h>

static const uint32_t SAMPLE_US = 10000;      // 100 Hz
static const uint8_t  BATCH = 10;             // frames per notification
static const uint8_t  FRAME_LEN = 20;

LSM6DS3 imu(I2C_MODE, 0x6A);
BLEUart bleuart;

uint8_t  buf[BATCH * FRAME_LEN];
uint8_t  nbuf = 0;
uint8_t  seq = 0;
uint32_t next_us = 0;

static void led(bool r, bool g, bool b) {     // XIAO LEDs are active LOW
  digitalWrite(LED_RED, r ? LOW : HIGH);
  digitalWrite(LED_GREEN, g ? LOW : HIGH);
  digitalWrite(LED_BLUE, b ? LOW : HIGH);
}

static int16_t clip16(float v) {
  if (v > 32767.0f) return 32767;
  if (v < -32768.0f) return -32768;
  return (int16_t)lroundf(v);
}

static void put16(uint8_t *p, int16_t v) { p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF; }

void setup() {
  pinMode(LED_RED, OUTPUT); pinMode(LED_GREEN, OUTPUT); pinMode(LED_BLUE, OUTPUT);
  led(false, false, false);

#ifdef PIN_LSM6DS3TR_C_POWER                  // the IMU has its own power switch on this board
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(20);
#endif

  imu.settings.accelRange = 4;                // +-4 g
  imu.settings.accelSampleRate = 104;         // Hz (sensor runs slightly above our 100 Hz read rate)
  imu.settings.gyroRange = 245;               // +-245 deg/s: small angles, fine resolution
  imu.settings.gyroSampleRate = 104;
  if (imu.begin() != 0) {                     // 0 = IMU_SUCCESS
    while (true) { led(true, false, false); delay(200); led(false, false, false); delay(200); }
  }

  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);   // larger packets
  Bluefruit.begin();
  Bluefruit.setName("BalanceBand");
  Bluefruit.setTxPower(4);
  bleuart.begin();

  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);

  next_us = micros();
}

void loop() {
  bool streaming = Bluefruit.connected() && bleuart.notifyEnabled();

  if ((int32_t)(micros() - next_us) < 0) {    // not yet time for the next sample
    if (!streaming) led(false, (millis() / 500) % 2 == 0, false);
    return;
  }
  uint32_t t = next_us;
  next_us += SAMPLE_US;
  if ((int32_t)(micros() - next_us) > (int32_t)(5 * SAMPLE_US)) next_us = micros();   // fell far behind: resync

  if (!streaming) { nbuf = 0; return; }
  led(false, false, true);

  uint8_t *f = buf + nbuf * FRAME_LEN;
  f[0] = 0xA5; f[1] = 0x5A; f[2] = seq++;
  put32(f + 3, t);
  put16(f + 7,  clip16(imu.readFloatAccelX() * 1000.0f));
  put16(f + 9,  clip16(imu.readFloatAccelY() * 1000.0f));
  put16(f + 11, clip16(imu.readFloatAccelZ() * 1000.0f));
  put16(f + 13, clip16(imu.readFloatGyroX() * 100.0f));
  put16(f + 15, clip16(imu.readFloatGyroY() * 100.0f));
  put16(f + 17, clip16(imu.readFloatGyroZ() * 100.0f));
  uint8_t sum = 0;
  for (int i = 2; i < FRAME_LEN - 1; i++) sum += f[i];
  f[FRAME_LEN - 1] = sum;

  if (++nbuf == BATCH) {
    bleuart.write(buf, nbuf * FRAME_LEN);
    nbuf = 0;
  }
}
