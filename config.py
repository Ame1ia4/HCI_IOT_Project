import os
from dotenv import load_dotenv
load_dotenv()

# Camera source
CAMERA_SOURCE = "phone"  # "phone" | "esp32" | "pi"

SOURCES = {
    "phone": "http://10.54.155.148:8080/video",
    "esp32": "http://192.168.1.105/stream",
    "pi":    0, # Use 0 for local camera index if /dev/video0 fails
}

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Card detection
CARD_ASPECT_RATIO    = 1.585
ASPECT_RATIO_TOL     = 0.20   # Tightened: 0.45 was too loose
MIN_CARD_AREA        = 5000   # Increased: ignore small background objects
CANNY_THRESHOLD_LOW  = 50
CANNY_THRESHOLD_HIGH = 150

# Validation thresholds
VALIDATION_SCORE_THRESHOLD = 0.58

# Weights must sum to 1.0
VALIDATION_WEIGHTS = {
    "colour": 0.40,
    "text":   0.35,
    "layout": 0.20,
    "ml":     0.05,
}

# HSV colour ranges (STRICTER GREEN)
CARD_COLOUR_RANGES = {
    # Narrowed Hue to 50-90 to avoid yellowish or bluish greens
    "ul_student": (50, 95, 35, 255, 15, 210),
}

CARD_KEYWORDS = {
    "ul_student": ["University", "Limerick", "Student", "ID"],
}

# Integrations
SUPABASE_ENABLED = False
SERIAL_ENABLED   = False
ENDPOINT_ENABLED = True
ENDPOINT_URL     = "http://127.0.0.1:5000/scan"
ENDPOINT_TIMEOUT = 2