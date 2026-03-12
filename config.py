# ---------------------------------------------------------------------------
# Camera source
# Change CAMERA_SOURCE to switch between input devices — nothing else needed.
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "esp32"  # "phone" | "esp32"

SOURCES = {
    "phone": "http://192.168.1.100:8080/video",  # IP Webcam app (Android/iOS)
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
CARD_ASPECT_RATIO   = 1.585   # ISO/IEC 7810 ID-1: 85.6mm x 54mm
ASPECT_RATIO_TOL    = 0.15    # ± tolerance on aspect ratio check
MIN_CARD_AREA       = 8000    # minimum contour area in pixels
CANNY_THRESHOLD_LOW  = 50
CANNY_THRESHOLD_HIGH = 150

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
VALIDATION_SCORE_THRESHOLD = 0.75  # minimum weighted score to call VALID

VALIDATION_WEIGHTS = {
    "colour": 0.3,
    "text":   0.5,
    "layout": 0.2,
}

# ---------------------------------------------------------------------------
# HSV colour ranges for known card types
# Format: (H_low, H_high, S_low, S_high, V_low, V_high)
# ---------------------------------------------------------------------------
CARD_COLOUR_RANGES = {
    "parking_permit_ie": (15, 35, 150, 255, 150, 255),   # Orange/yellow — DDAI Ireland
    "parking_permit_eu": (100, 130, 80, 255, 100, 220),  # Light blue — EU standard
}

# ---------------------------------------------------------------------------
# OCR keywords for known card types
# ---------------------------------------------------------------------------
CARD_KEYWORDS = {
    "parking_permit": ["Disabled Parking", "DDAI", "Ireland"],
}

# ---------------------------------------------------------------------------
# Arduino serial communication
# ---------------------------------------------------------------------------
SERIAL_PORT = "COM3"   # Windows: "COM3" etc. | Linux/Mac: "/dev/ttyUSB0"
SERIAL_BAUD = 9600
SERIAL_ENABLED = False  # Set True once Arduino is connected
