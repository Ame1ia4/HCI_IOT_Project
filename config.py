# ---------------------------------------------------------------------------
# Camera source
# Change CAMERA_SOURCE to switch between input devices — nothing else needed.
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "phone"  # "phone" | "esp32"

SOURCES = {
    "phone": "http://10.54.148.17:8080/video",  # IP Webcam app (Android/iOS)
    "esp32": "http://192.168.1.105/stream",
}

# ---------------------------------------------------------------------------
# Frame settings
# All frames are resized to this resolution so thresholds generalise across
# cameras (ESP32-CAM outputs 640x480 at most).
# ---------------------------------------------------------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------
CARD_ASPECT_RATIO    = 1.585  # ISO/IEC 7810 ID-1: 85.6mm x 54mm
ASPECT_RATIO_TOL     = 0.3   # ± tolerance — loosened for real-world camera angles
MIN_CARD_AREA        = 5000  # minimum contour area in pixels
CANNY_THRESHOLD_LOW  = 30
CANNY_THRESHOLD_HIGH = 100

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
VALIDATION_SCORE_THRESHOLD = 0.65  # minimum weighted score to call VALID

VALIDATION_WEIGHTS = {
    "colour": 0.4,  # colour of green band — very reliable
    "text":   0.3,  # OCR keywords — less reliable on laminated cards under camera
    "layout": 0.3,  # layout zone checks — very reliable
}

# ---------------------------------------------------------------------------
# HSV colour ranges for known card types
# Format: (H_low, H_high, S_low, S_high, V_low, V_high)
# ---------------------------------------------------------------------------
CARD_COLOUR_RANGES = {
    # UL dark green #006B3C — OpenCV HSV: H≈77, S=255, V=107
    # H range 50-105 covers lighting variation; S lowered to 50 to handle
    # camera desaturation; V range 10-200 handles dim and bright/reflective lighting
    "ul_student": (50, 105, 50, 255, 10, 200),
}

# ---------------------------------------------------------------------------
# OCR keywords for known card types
# ---------------------------------------------------------------------------
CARD_KEYWORDS = {
    "ul_student": ["University of Limerick", "UL", "Student", "University"],
}

# ---------------------------------------------------------------------------
# Supabase
# Credentials are loaded from .env (never commit that file).
# Set SUPABASE_ENABLED = True once your project and students table are ready.
# Required table:  students (student_id TEXT PRIMARY KEY, name TEXT)
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_ENABLED = True
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Arduino serial communication
# ---------------------------------------------------------------------------
SERIAL_PORT    = "COM3"   # Windows: "COM3" etc. | Linux/Mac: "/dev/ttyUSB0"
SERIAL_BAUD    = 9600
SERIAL_ENABLED = False    # Set True once Arduino is connected

# ---------------------------------------------------------------------------
# IoT HTTP endpoint
# The CV pipeline POSTs scan results here as JSON.
# Run server/app.py on the same machine or any device on the network.
# ---------------------------------------------------------------------------
ENDPOINT_ENABLED = True
ENDPOINT_URL     = "http://127.0.0.1:5000/scan"  # change host if server is on another device
ENDPOINT_TIMEOUT = 2  # seconds — keeps the camera loop from stalling
