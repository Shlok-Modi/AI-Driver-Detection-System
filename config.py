SERIAL_PORT = "COM3"
BAUD_RATE = 9600
SERIAL_SEND_INTERVAL = 0.1

# Eye Aspect Ratio — below this for N frames = drowsy
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20       # ~0.67s at 30fps

# Mouth Aspect Ratio — above this for N frames = yawning
MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES = 15

# Head pose angles (degrees)
HEAD_PITCH_THRESHOLD = 30    # Nodding down
HEAD_YAW_THRESHOLD = 35      # Turning sideways

CAM_INDEX = 0