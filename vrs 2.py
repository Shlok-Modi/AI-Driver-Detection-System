import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance as dist

# ── Thresholds ────────────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.25     # Eye aspect ratio — below this = eyes closing
ALERT_TIME_THRESHOLD = 1.0      # Seconds a condition must persist before alerting
HEAD_PITCH_THRESHOLD = 30       # Degrees nodding down
HEAD_YAW_THRESHOLD   = 35       # Degrees turning sideways
MAR_THRESHOLD        = 0.65      # Mouth aspect ratio — above this = yawning

CAMERA_INDEX         = 0

# ── MediaPipe Landmark Indices ────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)
POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]

# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_ear(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def calc_mar(landmarks, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH]
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[7])
    C = dist.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C)

def calc_head_pose(landmarks, w, h):
    img_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in POSE_LANDMARKS],
        dtype=np.float64
    )
    focal = w
    cam_matrix = np.array([
        [focal, 0,     w / 2],
        [0,     focal, h / 2],
        [0,     0,     1    ]
    ], dtype=np.float64)
    _, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, img_pts, cam_matrix,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1]

# ── Condition Timer ───────────────────────────────────────────────────────────
# Tracks how long each condition has been continuously detected.
# Only fires an alert once the condition persists for ALERT_TIME_THRESHOLD seconds.

class ConditionTimer:
    def __init__(self, threshold=ALERT_TIME_THRESHOLD):
        self.threshold  = threshold
        self.start_time = {}   # condition -> time it started

    def update(self, condition: str, detected: bool):
        """
        Call every frame with whether the condition is currently detected.
        Returns (alert_active, elapsed_seconds).
        """
        if detected:
            if condition not in self.start_time:
                self.start_time[condition] = time.time()
            elapsed = time.time() - self.start_time[condition]
            return elapsed >= self.threshold, elapsed
        else:
            self.start_time.pop(condition, None)
            return False, 0.0

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    timers = ConditionTimer(ALERT_TIME_THRESHOLD)

    print("Running — press Q to quit.")
    print(f"Alerts fire after {ALERT_TIME_THRESHOLD}s of continuous detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alerts        = []   # confirmed alerts (past threshold)
        timer_display = {}   # condition -> elapsed seconds (for display)

        if not results.multi_face_landmarks:
            # No face is instant alert — no timer needed
            alerts.append("NO FACE")
        else:
            lm = results.multi_face_landmarks[0].landmark

            # — EAR (drowsy) —
            ear_val  = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2.0
            drowsy_now = ear_val < EAR_THRESHOLD
            alert, elapsed = timers.update("DROWSY", drowsy_now)
            if drowsy_now:
                timer_display["DROWSY"] = elapsed
            if alert:
                alerts.append("DROWSY")

            # — MAR (yawning) —
            mar_val   = calc_mar(lm, w, h)
            yawn_now  = mar_val > MAR_THRESHOLD
            alert, elapsed = timers.update("YAWNING", yawn_now)
            if yawn_now:
                timer_display["YAWNING"] = elapsed
            if alert:
                alerts.append("YAWNING")

            # — Head pose (distracted) —
            pitch, yaw = 0.0, 0.0
            try:
                pitch, yaw    = calc_head_pose(lm, w, h)
                distract_now  = abs(pitch) > HEAD_PITCH_THRESHOLD or abs(yaw) > HEAD_YAW_THRESHOLD
            except Exception:
                distract_now  = False
            alert, elapsed = timers.update("DISTRACTED", distract_now)
            if distract_now:
                timer_display["DISTRACTED"] = elapsed
            if alert:
                alerts.append("DISTRACTED")

            # HUD metrics
            cv2.putText(frame, f"EAR: {ear_val:.2f}  MAR: {mar_val:.2f}",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}",
                        (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # ── Alert banner + timers ─────────────────────────────────────────────
        if alerts:
            label = "  |  ".join(alerts)
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(frame, label, (12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "OK", (12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 80), 2)

        # Show a timer next to the banner for any condition being tracked
        # Even before it triggers (shows as a warning countdown)
        x_offset = 12
        for condition, elapsed in timer_display.items():
            confirmed = condition in alerts
            bar_color = (0, 0, 220) if confirmed else (0, 165, 255)  # red if alert, orange if pending
            label     = f"{condition}: {elapsed:.1f}s"
            cv2.putText(frame, label, (x_offset, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)
            x_offset += 220  # space multiple timers horizontally

        cv2.imshow("Driver Alert v1 — Laptop Only", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
