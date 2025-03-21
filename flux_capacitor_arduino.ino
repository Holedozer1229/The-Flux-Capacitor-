// Flux Capacitor Arduino Module
// Controls a stepper motor, electromagnet, and hall effect sensor for the 8-track player

#include <Stepper.h>

// Pin definitions
const int STEPPER_PIN_1 = 8;  // Stepper motor pins (adjust based on your wiring)
const int STEPPER_PIN_2 = 9;
const int STEPPER_PIN_3 = 10;
const int STEPPER_PIN_4 = 11;
const int ELECTROMAGNET_PIN = 5;  // PWM pin for electromagnet (adjust as needed)
const int HALL_SENSOR_PIN = A0;   // Analog pin for hall effect sensor (e.g., A1302)

// Stepper motor configuration
const int STEPS_PER_REVOLUTION = 200;  // Adjust based on your stepper motor
Stepper myStepper(STEPS_PER_REVOLUTION, STEPPER_PIN_1, STEPPER_PIN_2, STEPPER_PIN_3, STEPPER_PIN_4);

// Base speed for the 8-track (3.75 ips)
const float BASE_SPEED = 3.75;  // Inches per second
const float TAPE_WIDTH = 0.25;  // Tape width in inches (1/4 inch)
const float BASE_STEPS_PER_SECOND = (BASE_SPEED / (TAPE_WIDTH / STEPS_PER_REVOLUTION));  // Steps per second for base speed

// Electromagnet control
const int MAX_ELECTROMAGNET_VALUE = 255;  // Max PWM value for electromagnet

// Hall sensor variables
int hallValue = 0;
const int HALL_THRESHOLD = 512;  // Adjust based on your sensor's baseline (typically 512 for A1302)

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Set pin modes
  pinMode(ELECTROMAGNET_PIN, OUTPUT);
  pinMode(HALL_SENSOR_PIN, INPUT);
  
  // Initialize stepper motor
  myStepper.setSpeed(60);  // Set initial speed (RPM, will be adjusted dynamically)
  
  // Initial message
  Serial.println("Flux Capacitor Arduino Module Initialized");
}

void loop() {
  // Check for incoming serial data from Python
  if (Serial.available() > 0) {
    // Read the incoming byte (0-255, scaled from the flux signal)
    int signalValue = Serial.read();
    
    // Map the signal value to stepper speed (modulate around base speed)
    float speedVariation = map(signalValue, 0, 255, -50, 50);  // ±50 steps/second variation
    float adjustedStepsPerSecond = BASE_STEPS_PER_SECOND + speedVariation;
    
    // Convert steps per second to RPM for the stepper motor
    float rpm = (adjustedStepsPerSecond * 60.0) / STEPS_PER_REVOLUTION;
    myStepper.setSpeed(max(1, min(rpm, 100)));  // Constrain RPM between 1 and 100
    
    // Step the motor
    myStepper.step(1);  // Step once per loop (adjust for smoother control if needed)
    
    // Map the signal value to electromagnet PWM (0-255)
    analogWrite(ELECTROMAGNET_PIN, signalValue);
    
    // Read the hall effect sensor
    hallValue = analogRead(HALL_SENSOR_PIN);
    
    // Send feedback to Python (e.g., magnetic field strength)
    Serial.print("Hall:");
    Serial.println(hallValue);
    
    // Small delay to prevent overwhelming the serial buffer
    delay(1);
  }
}
