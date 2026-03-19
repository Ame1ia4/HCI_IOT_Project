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
# --- config.py updates ---

# Standard ID card is 85.6mm x 53.98mm = ~1.585 aspect ratio
CARD_ASPECT_RATIO    = 1.585
ASPECT_RATIO_TOL     = 0.15   # Tightened from 0.20 to reject faces (which are rounder)

# Adjust this based on how close you hold the card to the camera
# 10% of a 640x480 frame is about 30,000. Let's set a healthy middle ground.
MIN_CARD_AREA        = 20000  # Increased from 5000 to ignore background clutter
# Add this to config.py
MAX_CARD_AREA = 150000  # Adjust this: it should be larger than a card but smaller than the whole screen
# Increase Canny thresholds to ignore soft edges (like facial features)
# and only catch sharp edges (like card borders)
CANNY_THRESHOLD_LOW  = 100 
CANNY_THRESHOLD_HIGH = 200

# Validation thresholds
VALIDATION_SCORE_THRESHOLD = 0.55

VALIDATION_WEIGHTS = {
    "colour": 0.40,  # Trust the green band more
    "text":   0.20,  # Lower text weight since it's failing to crop the full card
    "layout": 0.30,  # Zone checks are good
    "ml":     0.10,  
}

# HSV colour ranges (STRICTER GREEN)
CARD_COLOUR_RANGES = {
    # Narrowed Hue to 50-90 to avoid yellowish or bluish greens
    "ul_student": (35, 85, 40, 255, 30, 180),
}

CARD_KEYWORDS = {
    "ul_student": ["University", "Limerick", "Student", "ID"],
}

# Integrations
SUPABASE_ENABLED = True
SUPABASE_URL     = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY     = os.getenv("SUPABASE_KEY", "")
SERIAL_ENABLED   = False
ENDPOINT_ENABLED = True
ENDPOINT_URL     = "http://127.0.0.1:5000/scan"
ENDPOINT_TIMEOUT = 2