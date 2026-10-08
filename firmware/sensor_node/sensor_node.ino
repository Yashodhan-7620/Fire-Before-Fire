/*
  =====================================================================
  Fire / Short-Circuit Early-Warning System — Sensor Node Firmware
  Board: ESP32 (NOT Arduino Uno)
  =====================================================================
  IMPORTANT COMPATIBILITY NOTE
  -----------------------------------------------------------------
  Your two original sketches targeted different hardware:
    1) The DS18B20 sketch is generic and runs on either an Uno or ESP32.
    2) The Blynk + EmonLib sketch includes <WiFi.h> and
       BlynkSimpleEsp32.h, which only exist for the ESP32 core.
       An Arduino Uno has no Wi-Fi radio and cannot compile or run it.
  Because of this, both sketches are merged below into ONE ESP32
  sketch. If you actually have an Uno + a separate ESP32/ESP8266 for
  Wi-Fi, tell me and I'll split this back into two files (Uno does
  sensing over Serial/I2C, ESP32 only handles Wi-Fi/Blynk).

  BUGS FIXED FROM YOUR ORIGINAL CODE
  -----------------------------------------------------------------
  - emon.calcV(...) does not exist in EmonLib -> changed to
    emon.calcVI(20, 2000), which is what actually computes Vrms/Irms.
  - emon.current(34, currCalibration) had a stray comment referencing
    a non-existent "currect" typo — pin/args cleaned up.
  - delay(2000) inside loop() blocked Blynk.run()/timer.run() and
    would eventually disconnect Blynk. Replaced with BlynkTimer
    intervals so nothing in loop() blocks.
  - Both sketches each had their own setup()/loop(); merged into a
    single setup()/loop() (a sketch can only have one of each).

  ADDED
  -----------------------------------------------------------------
  - 16x2 I2C LCD showing live temp / voltage / current / power.
  - RGB LED status indicator (green = normal, amber = warning,
    red = threshold exceeded) — used remaining free GPIOs.
  - HTTP POST of every reading, as JSON, to your local ingestion
    API (src/ingestion/api.py) so it lands in the database that
    the MLOps pipeline trains on.
  =====================================================================
*/

#define BLYNK_TEMPLATE_ID "TMPL3O1qyEX83"
#define BLYNK_TEMPLATE_NAME "Fire before Fire"
#define BLYNK_AUTH_TOKEN "gpqJk7PFQ14o585iIlgO5DHBDNYeoCHB"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <HTTPClient.h>
#include <BlynkSimpleEsp32.h>
#include "EmonLib.h"
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>

#include <Adafruit_RGBLCDShield.h>
#include <utility/Adafruit_MCP23017.h>
// ---------------- Pin map ----------------
#define ONE_WIRE_BUS   4     // DS18B20 data pin
#define VOLT_PIN       35    // ZMPT voltage sensor (ADC1, input-only pin)
#define CURR_PIN       34    // SCT current sensor (ADC1, input-only pin)
#define SDA_PIN 22
#define SCL_PIN 23
// I2C LCD uses the default ESP32 bus: SDA=21, SCL=22 (free, unused above)

// ---------------- Calibration ----------------
#define V_CALIBRATION   3.3
#define I_CALIBRATION   0.50

// ---------------- Alert thresholds (tune to your wiring) ----------------
#define TEMP_ALERT_C     34.0
#define POWER_ALERT_W    1500.0
#define CURRENT_ALERT_A  10.0

// ---------------- Wi-Fi / Blynk ----------------
char auth[] = BLYNK_AUTH_TOKEN;
char ssid[] = "Differentiationnn";
char pass[] = "calculus-eAsY-xn_nxn-1";

// Point this at the machine running src/ingestion/api.py on your LAN
const char* INGEST_URL = "http://192.168.29.84:8000/ingest";
const char* NODE_ID    = "node-01";
Adafruit_RGBLCDShield lcd = Adafruit_RGBLCDShield();
// ---------------- Objects ----------------
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature tempSensor(&oneWire);
EnergyMonitor emon;
BlynkTimer timer;
 // if the LCD stays blank, try 0x3F instead of 0x27

float kWh = 0;
unsigned long lastMillis = 0;
float lastTempC = 0, lastVrms = 0, lastIrms = 0, lastPower = 0;

// void setRGB(int r, int g, int b) {
//   analogWrite(RGB_RED_PIN, r);
//   analogWrite(RGB_GREEN_PIN, g);
//   analogWrite(RGB_BLUE_PIN, b);
// }

void readTemperature() {
  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  if (t == -127.0) {
    Serial.println("Temp sensor error: not found");
    return;
  }
  lastTempC = t;
  Blynk.virtualWrite(V4, lastTempC);
}

void sendToIngestAPI() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  http.begin(INGEST_URL);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"node_id\":\"" + String(NODE_ID) + "\",";
  payload += "\"temperature_c\":" + String(lastTempC, 2) + ",";
  payload += "\"voltage_v\":" + String(lastVrms, 2) + ",";
  payload += "\"current_a\":" + String(lastIrms, 4) + ",";
  payload += "\"power_w\":" + String(lastPower, 4) + ",";
  payload += "\"kwh\":" + String(kWh, 4);
  payload += "}";

  int code = http.POST(payload);
  if (code <= 0) {
    Serial.print("Ingest POST failed: ");
    Serial.println(http.errorToString(code));
  }
  http.end();
}

void readPowerAndPush() {
  emon.calcVI(20, 2000);
  lastVrms = emon.Vrms;
  lastIrms = emon.Irms;
  lastPower = emon.apparentPower;
  kWh += lastPower * (millis() - lastMillis) / 3600000000.0;
  lastMillis = millis();

  Serial.printf("Vrms: %.2fV  Irms: %.4fA  Power: %.4fW  kWh: %.4f  Temp: %.2fC\n",
                lastVrms, lastIrms, lastPower, kWh, lastTempC);

  Blynk.virtualWrite(V0, lastVrms);
  Blynk.virtualWrite(V1, lastIrms);
  Blynk.virtualWrite(V2, lastPower);
  Blynk.virtualWrite(V3, kWh);

  sendToIngestAPI();
}

void updateLcdAndRGB() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("T:"); lcd.print(lastTempC, 1); lcd.print("C V:"); lcd.print(lastVrms, 0);
  lcd.setCursor(0, 1);
  lcd.print("I:"); lcd.print(lastIrms, 2); lcd.print("A P:"); lcd.print(lastPower, 0);

  bool alert = (lastTempC > TEMP_ALERT_C) || (lastPower > POWER_ALERT_W) || (lastIrms > CURRENT_ALERT_A);
  bool warn  = (lastTempC > TEMP_ALERT_C * 0.8) || (lastPower > POWER_ALERT_W * 0.8);

  if (alert) {
    lcd.setBacklight(0x1);
    Blynk.logEvent("fire_risk_alert", "Threshold exceeded on sensor node");
  } else if (warn) {
    lcd.setBacklight(0x2);
  } else {
    lcd.setBacklight(0x4);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("---- Fire/Short-Circuit Sensor Node ----");

  // pinMode(RGB_RED_PIN, OUTPUT);
  // pinMode(RGB_GREEN_PIN, OUTPUT);
  // pinMode(RGB_BLUE_PIN, OUTPUT);

  lcd.begin(16,2);
  lcd.setBacklight(0x7);
  lcd.setCursor(0, 0);
  lcd.print("Booting...");

  tempSensor.begin();
  emon.voltage(VOLT_PIN, V_CALIBRATION, 1.7);
  emon.current(CURR_PIN, I_CALIBRATION);

  Blynk.begin(auth, ssid, pass);

  timer.setInterval(2000L, readTemperature);
  timer.setInterval(5000L, readPowerAndPush);
  timer.setInterval(2000L, updateLcdAndRGB);

  lastMillis = millis();
}

void loop() {
  Blynk.run();
  timer.run();
}
