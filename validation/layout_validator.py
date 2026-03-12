import cv2
import numpy as np

from config import CARD_COLOUR_RANGES


# ---------------------------------------------------------------------------
# Grid zone definitions
# The card is divided into a 3-column x 2-row grid (6 zones).
# Zones are referenced as (row, col) with 0-based indexing:
#
#   (0,0) | (0,1) | (0,2)
#   ------+-------+------
#   (1,0) | (1,1) | (1,2)
# ---------------------------------------------------------------------------

def _get_zone(card_img, row, col, rows=2, cols=3):
    """Return the sub-image for a grid zone (row, col)."""
    h, w  = card_img.shape[:2]
    zone_h = h // rows
    zone_w = w // cols
    y1, y2 = row * zone_h, (row + 1) * zone_h
    x1, x2 = col * zone_w, (col + 1) * zone_w
    return card_img[y1:y2, x1:x2]


def _colour_pixel_ratio(zone_img, hsv_range):
    """
    Return the fraction of pixels in zone_img that fall within the given
    HSV range. Used to detect whether a particular colour blob is present.
    """
    hsv   = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = hsv_range
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size


def _dark_pixel_ratio(zone_img, threshold=60):
    """
    Return the fraction of pixels darker than threshold (grayscale).
    High dark-pixel ratio in a zone indicates printed text or a dark symbol.
    """
    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray < threshold)) / gray.size


# ---------------------------------------------------------------------------
# Per-card-type layout rules
#
# Each rule is a dict:
#   zone        : (row, col) to inspect
#   check       : "colour" | "dark" | "light"
#   hsv_range   : used when check == "colour"
#   min_ratio   : minimum pixel fraction required to pass the check
#   weight      : contribution to the overall layout confidence score
# ---------------------------------------------------------------------------

# UL Student Card layout:
#   Top row  — dark green header band spanning the full width
#   Bottom-left  — student photo (lighter region)
#   Bottom-right — printed text (dark pixels on white)
#
#   (0,0) green | (0,1) green | (0,2) green
#   ------------+-------------+------------
#   (1,0) photo | (1,1) text  | (1,2) text

_RULES_UL = [
    {
        "description": "Green header present in top-left zone",
        "zone":        (0, 0),
        "check":       "colour",
        "hsv_range":   CARD_COLOUR_RANGES["ul_student"],
        "min_ratio":   0.25,
        "weight":      0.4,
    },
    {
        "description": "Green header present in top-right zone",
        "zone":        (0, 2),
        "check":       "colour",
        "hsv_range":   CARD_COLOUR_RANGES["ul_student"],
        "min_ratio":   0.25,
        "weight":      0.3,
    },
    {
        "description": "Printed text present in bottom-right zone",
        "zone":        (1, 2),
        "check":       "dark",
        "min_ratio":   0.03,
        "weight":      0.3,
    },
]

_LAYOUT_RULES = {
    "ul_student": _RULES_UL,
}


def validate_layout(card_img, card_type):
    """
    Check that key design elements appear in the expected grid zones
    for the given card type.

    Args:
      card_img:  BGR image of the cropped, perspective-corrected card.
      card_type: "ul_student".

    Returns:
      (is_valid: bool, confidence: float 0.0–1.0)
      is_valid is True when the weighted confidence exceeds 0.5.
    """
    rules = _LAYOUT_RULES.get(card_type)
    if rules is None:
        return False, 0.0

    score = 0.0

    for rule in rules:
        row, col = rule["zone"]
        zone     = _get_zone(card_img, row, col)

        if rule["check"] == "dark":
            ratio = _dark_pixel_ratio(zone)
        else:
            ratio = _colour_pixel_ratio(zone, rule["hsv_range"])

        if ratio >= rule["min_ratio"]:
            score += rule["weight"]

    confidence = round(score, 3)
    return confidence >= 0.5, confidence
