import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# ── Thresholds (tune these to your face) ─────────────────────────────────────
EAR_THRESHOLD        = 0.25
EAR_CONSEC_FRAMES    = 20
MAR_THRESHOLD        = 0.6
MAR_CONSEC_FRAMES    = 15
HEAD_PITCH_THRESHOLD = 30
HEAD_YAW_THRESHOLD   = 35

CAMERA_INDEX         = 0        # 0 = built-in webcam, 1 = first USB camera

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

    ear_counter = 0
    mar_counter = 0

    print("Running — press Q to quit.")
    print(f"Thresholds: EAR<{EAR_THRESHOLD} for {EAR_CONSEC_FRAMES} frames | "
          f"MAR>{MAR_THRESHOLD} for {MAR_CONSEC_FRAMES} frames | "
          f"Pitch>{HEAD_PITCH_THRESHOLD}° or Yaw>{HEAD_YAW_THRESHOLD}°")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        alerts = []

        if not results.multi_face_landmarks:
            alerts.append("NO FACE")
        else:
            lm = results.multi_face_landmarks[0].landmark

            # EAR — drowsiness
            ear_val = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2.0
            if ear_val < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= EAR_CONSEC_FRAMES:
                    alerts.append("DROWSY")
            else:
                ear_counter = 0

            # MAR — yawning
            mar_val = calc_mar(lm, w, h)
            if mar_val > MAR_THRESHOLD:
                mar_counter += 1
                if mar_counter >= MAR_CONSEC_FRAMES:
                    alerts.append("YAWNING")
            else:
                mar_counter = 0

            # Head pose — distracted
            pitch, yaw = 0.0, 0.0
            try:
                pitch, yaw = calc_head_pose(lm, w, h)
                if abs(pitch) > HEAD_PITCH_THRESHOLD or abs(yaw) > HEAD_YAW_THRESHOLD:
                    alerts.append("DISTRACTED")
            except Exception:
                pass

            # HUD
            cv2.putText(frame, f"EAR: {ear_val:.2f}  MAR: {mar_val:.2f}",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}",
                        (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Alert banner
        if alerts:
            label = "  |  ".join(alerts)
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(frame, label, (12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "OK", (12, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 80), 2)

        cv2.imshow("Driver Alert — v1 (Laptop Only)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
