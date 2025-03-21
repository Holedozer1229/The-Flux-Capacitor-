// Flux Capacitor Arduino Code
// Controls an 8-track player's stepper motor and electromagnet, reads Hall sensor

#include <Stepper.h>

// Pin definitions
const int STEPPER_PIN1 = 2;  // IN1 on stepper driver
const int STEPPER_PIN2 = 3;  // IN2
const int STEPPER_PIN3 = 4;  // IN3
const int STEPPER_PIN4 = 5;  // IN4
const int ELECTROMAGNET_PIN = 9;  // PWM pin for electromagnet
const int HALL_SENSOR_PIN = A0;   // Analog pin for Hall effect sensor

// Stepper motor setup (adjust steps per revolution based on your NEMA 17)
const int STEPS_PER_REV = 200;
Stepper stepper(STEPS_PER_REV, STEPPER_PIN1, STEPPER_PIN2, STEPPER_PIN3, STEPPER_PIN4);

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Set pin modes
  pinMode(ELECTROMAGNET_PIN, OUTPUT);
  pinMode(HALL_SENSOR_PIN, INPUT);
  
  // Set initial stepper speed (RPM)
  stepper.setSpeed(60);  // Adjust as needed
  
  // Wait for serial connection
  while (!Serial) {
    delay(10);
  }
  Serial.println("Flux Capacitor Arduino Ready");
}

void loop() {
  if (Serial.available() > 0) {
    // Read PWM value from Python (0-255)
    int pwmValue = Serial.read();
    
    // Control electromagnet with PWM
    analogWrite(ELECTROMAGNET_PIN, pwmValue);
    
    // Adjust stepper speed based on PWM value (map 0-255 to 0-120 RPM)
    int motorSpeed = map(pwmValue, 0, 255, 0, 120);
    stepper.setSpeed(motorSpeed);
    
    // Step the motor (one step per signal)
    stepper.step(1);
    
    // Read Hall effect sensor (0-1023 range, mapped to voltage)
    int hallRaw = analogRead(HALL_SENSOR_PIN);
    float hallVoltage = (hallRaw / 1023.0) * 5.0;  // Convert to volts (5V reference)
    
    // Send Hall sensor reading back to Python
    Serial.println(hallVoltage, 2);  // 2 decimal places
    
    // Small delay to match Python's sample rate (44100 Hz ~ 22.7 µs)
    delayMicroseconds(23);
  }
}ent overwhelming the serial buffer
    delay(1);
  }
}
