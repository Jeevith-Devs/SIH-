/*
 * ASTRA-Track: Physical Hardware-in-the-Loop (HIL) Gimbal Controller
 * Smart India Hackathon 2026 — Problem Statement 26169
 * 
 * Target Hardware: Arduino Uno / Nano / ESP32 / RP2040
 * Actuators: Dual 9g Micro Servos (SG90, MG90S, MG996R)
 *   - Pin 9:  Pan Servo (Azimuth axis)
 *   - Pin 10: Tilt Servo (Elevation axis)
 * 
 * Communication Protocol: 115200 Baud Serial
 * Commands:
 *   PING           -> Returns "PONG ASTRA_GIMBAL_V1"
 *   SET_PT <p> <t> -> Sets Pan and Tilt target angles (-45.0 to +45.0 deg relative to boresight)
 *   CENTER         -> Returns servos to optical boresight (0.0, 0.0)
 *   STATUS         -> Returns "STATUS <cur_pan> <cur_tilt>"
 */

#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN = 9;
const int TILT_PIN = 10;

// Mechanical Calibration offsets (Degrees)
const float PAN_CENTER_SERVO_DEG = 90.0;
const float TILT_CENTER_SERVO_DEG = 90.0;

// Hardware Travel Limits (+/- 45 degrees from boresight)
const float PAN_MIN_DEG = -45.0;
const float PAN_MAX_DEG = 45.0;
const float TILT_MIN_DEG = -30.0;
const float TILT_MAX_DEG = 30.0;

// Slew rate smoothing
float current_pan = 0.0;
float current_tilt = 0.0;
float target_pan = 0.0;
float target_tilt = 0.0;

const float MAX_SLEW_PER_CYCLE = 2.0; // Smooth motion step

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  // Home to optical boresight
  updateServoOutputs(0.0, 0.0);

  Serial.println("ASTRA_GIMBAL_READY 115200");
}

void updateServoOutputs(float p, float t) {
  // Clamp within safety limits
  p = constrain(p, PAN_MIN_DEG, PAN_MAX_DEG);
  t = constrain(t, TILT_MIN_DEG, TILT_MAX_DEG);

  // Convert relative optical angles to absolute servo PWM angles (0 - 180)
  int pan_pwm = (int)round(PAN_CENTER_SERVO_DEG + p);
  int tilt_pwm = (int)round(TILT_CENTER_SERVO_DEG - t); // Invert tilt for standard elevation

  panServo.write(pan_pwm);
  tiltServo.write(tilt_pwm);

  current_pan = p;
  current_tilt = t;
}

void loop() {
  // Read incoming serial commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("PING")) {
      Serial.println("PONG ASTRA_GIMBAL_V1");
    }
    else if (cmd.startsWith("CENTER")) {
      target_pan = 0.0;
      target_tilt = 0.0;
      updateServoOutputs(0.0, 0.0);
      Serial.println("OK CENTERED");
    }
    else if (cmd.startsWith("SET_PT")) {
      // Format: SET_PT <pan_float> <tilt_float>
      int firstSpace = cmd.indexOf(' ');
      int secondSpace = cmd.indexOf(' ', firstSpace + 1);
      if (firstSpace > 0 && secondSpace > 0) {
        float p = cmd.substring(firstSpace + 1, secondSpace).toFloat();
        float t = cmd.substring(secondSpace + 1).toFloat();
        target_pan = p;
        target_tilt = t;
        updateServoOutputs(target_pan, target_tilt);
        Serial.print("OK PT ");
        Serial.print(current_pan, 2);
        Serial.print(" ");
        Serial.println(current_tilt, 2);
      } else {
        Serial.println("ERR INVALID_ARGS");
      }
    }
    else if (cmd.startsWith("STATUS")) {
      Serial.print("STATUS ");
      Serial.print(current_pan, 2);
      Serial.print(" ");
      Serial.println(current_tilt, 2);
    }
  }
}
