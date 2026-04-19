import cv2
import time

from config import CAM_INDEX, SERIAL_SEND_INTERVAL
from detector import DriverDetector
from Serial_comm import ArduinoComm

def main():
    # ── Setup ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    detector = DriverDetector()
    arduino  = ArduinoComm()

    last_serial_send = 0

    print("Running — press Q to quit.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        # Run detection
        frame, alert_active, alerts = detector.process(frame)

        # Send state to Arduino at fixed interval (not every frame)
        now = time.time()
        if now - last_serial_send >= SERIAL_SEND_INTERVAL:
            arduino.send_state(alert_active)
            last_serial_send = now

        # Arduino connection indicator (bottom-right corner)
        mode  = "ARDUINO CONNECTED" if arduino.connected else "NO ARDUINO"
        color = (0, 200, 100)       if arduino.connected else (100, 100, 100)
        h, w  = frame.shape[:2]
        cv2.putText(frame, mode, (w - 220, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow("Driver Alert System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    arduino.close()
    detector.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()