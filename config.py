import os
from dotenv import load_dotenv
load_dotenv()

# Camera source
CAMERA_SOURCE = "pi"  # "phone" | "esp32" | "pi"

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
VALIDATION_SCORE_THRESHOLD = 0.15 # Increased: Much stricter

VALIDATION_WEIGHTS = {
    "colour": 0.40,
    "text":   0.30, # Increased importance of text
    "layout": 0.20,
    "ml":     0.10,
}

# HSV colour ranges (STRICTER GREEN)
CARD_COLOUR_RANGES = {
    # Narrowed Hue to 50-90 to avoid yellowish or bluish greens
    "ul_student": (50, 90, 50, 255, 40, 200), 
}

CARD_KEYWORDS = {
    "ul_student": ["University", "Limerick", "Student", "ID"],
}

# Integrations
SUPABASE_ENABLED = False # Set to True if using
SERIAL_ENABLED = False
ENDPOINT_ENABLED = False # FIXED: Set to False to stop the Connection Refused errors
ENDPOINT_URL = "http://127.0.0.1:5000/scan"