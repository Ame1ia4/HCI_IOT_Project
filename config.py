# ---------------------------------------------------------------------------
# Camera source
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "phone"  # "phone" | "esp32" | "pi"

SOURCES = {
    "phone": "http://10.54.155.148:8080/video",
    "esp32": "http://192.168.1.105/stream",
    "pi":    "/dev/video0",
}

# ---------------------------------------------------------------------------
# Frame settings
# ---------------------------------------------------------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------
CARD_ASPECT_RATIO    = 1.585  # ISO/IEC 7810 ID-1: 85.6mm x 54mm
ASPECT_RATIO_TOL     = 0.35   # ± tolerance — loosened for real-world camera angles
MIN_CARD_AREA        = 8000   # minimum contour area in pixels
CANNY_THRESHOLD_LOW  = 50
CANNY_THRESHOLD_HIGH = 150

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
VALIDATION_SCORE_THRESHOLD = 0.55

VALIDATION_WEIGHTS = {
    "colour": 0.40,  # colour of green band — very reliable
    "text":   0.20,  # OCR keywords — less reliable on laminated cards under camera
    "layout": 0.30,  # layout zone checks — very reliable
    "ml":     0.10,  # ORB feature matching — falls back to 0 if no reference image
}

# ---------------------------------------------------------------------------
# HSV colour ranges for known card types
# ---------------------------------------------------------------------------
CARD_COLOUR_RANGES = {
    # UL dark green #006B3C — OpenCV HSV: H≈77, S=255, V=107
    # Broad S/V ranges to handle camera desaturation and lighting variation.
    # Hue is slightly tighter than original (45-105 vs 40-110) but the main
    # discriminator against non-UL cards is the text gate in run_validators.
    "ul_student": (45, 105, 30, 255, 5, 220),
}

# ---------------------------------------------------------------------------
# OCR keywords for known card types
# ---------------------------------------------------------------------------
CARD_KEYWORDS = {
    "ul_student": ["University of Limerick", "UL", "Student", "University"],
}

# ---------------------------------------------------------------------------
# Supabase
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
SERIAL_PORT    = "/dev/ttyUSB0"
SERIAL_BAUD    = 9600
SERIAL_ENABLED = False

# ---------------------------------------------------------------------------
# IoT HTTP endpoint
# ---------------------------------------------------------------------------
ENDPOINT_ENABLED = True
ENDPOINT_URL     = "http://127.0.0.1:5000/scan"
ENDPOINT_TIMEOUT = 2
