const int buttonPin = 2;
const int buzzerPin = 8;
const unsigned long EYE_CLOSED_THRESHOLD = 5000;

unsigned long eyeClosedStartTime = 0;
bool eyeWasClosed = false;
bool alarmActive = false;

// Alarm tone state
unsigned long alarmToneTimer = 0;
bool toneOn = false;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(buzzerPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("System Ready. Hold button to simulate closed eyes.");
}

void loop() {
  bool eyeIsClosed = (digitalRead(buttonPin) == LOW);

  if (eyeIsClosed) {
    if (!eyeWasClosed) {
      eyeClosedStartTime = millis();
      eyeWasClosed = true;
      Serial.println("Eye closed - timer started...");
    }

    unsigned long closedDuration = millis() - eyeClosedStartTime;

    static unsigned long lastPrint = 0;
    if (millis() - lastPrint >= 1000 && !alarmActive) {
      unsigned long secondsLeft = (EYE_CLOSED_THRESHOLD - closedDuration) / 1000 + 1;
      Serial.print("Alarm in: ");
      Serial.print(secondsLeft);
      Serial.println("s");
      lastPrint = millis();
    }

    if (closedDuration >= EYE_CLOSED_THRESHOLD) {
      if (!alarmActive) {
        alarmActive = true;
        Serial.println("!!! DROWSINESS ALERT !!!");
      }
    }

    // Non-blocking NORMAL beep pattern (steady on/off beep)
    if (alarmActive) {
      unsigned long now = millis();
      if (toneOn && now - alarmToneTimer >= 500) {        // beep ON for 500ms
        noTone(buzzerPin);
        toneOn = false;
        alarmToneTimer = now;
      } else if (!toneOn && now - alarmToneTimer >= 500) { // beep OFF for 500ms
        tone(buzzerPin, 1000);                             // 1000Hz = normal beep
        toneOn = true;
        alarmToneTimer = now;
      }
    }

  } else {
    // Eye opened — immediately stop everything
    if (eyeWasClosed) {
      Serial.println("Eye opened - all clear.\n");
    }
    eyeWasClosed = false;
    alarmActive = false;
    toneOn = false;
    eyeClosedStartTime = 0;
    noTone(buzzerPin);
  }
}