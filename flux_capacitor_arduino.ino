#define STEPPER_PIN1 2
#define STEPPER_PIN2 3
#define STEPPER_PIN3 4
#define STEPPER_PIN4 5
#define ELECTROMAGNET_PIN 9
#define HALL_SENSOR_PIN A0

int stepSequence[4][4] = {
  {1, 0, 0, 1},
  {1, 1, 0, 0},
  {0, 1, 1, 0},
  {0, 0, 1, 1}
};
int stepIndex = 0;

void setup() {
  pinMode(STEPPER_PIN1, OUTPUT);
  pinMode(STEPPER_PIN2, OUTPUT);
  pinMode(STEPPER_PIN3, OUTPUT);
  pinMode(STEPPER_PIN4, OUTPUT);
  pinMode(ELECTROMAGNET_PIN, OUTPUT);
  pinMode(HALL_SENSOR_PIN, INPUT);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0) {
    int value = Serial.read();  // 0-255 from Python
    analogWrite(ELECTROMAGNET_PIN, value);  // PWM electromagnet
    stepMotor(value);  // Drive stepper
    int hallReading = analogRead(HALL_SENSOR_PIN);  // 0-1023
    float hallVoltage = hallReading * (5.0 / 1023.0);  // Convert to volts
    Serial.println(hallVoltage);  // Send feedback
    delay(1);  // Match sample rate pacing
  }
}

void stepMotor(int speed) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(STEPPER_PIN1, stepSequence[stepIndex][0]);
    digitalWrite(STEPPER_PIN2, stepSequence[stepIndex][1]);
    digitalWrite(STEPPER_PIN3, stepSequence[stepIndex][2]);
    digitalWrite(STEPPER_PIN4, stepSequence[stepIndex][3]);
    delay(max(1, 255 - speed));  // Speed inversely proportional to delay
    stepIndex = (stepIndex + 1) % 4;
  }
}
