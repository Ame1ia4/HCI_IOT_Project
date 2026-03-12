import cv2
import numpy as np

from config import CARD_COLOUR_RANGES


def validate_colour(card_img, card_type):
    """
    Check whether the top header band of the card contains enough pixels
    within the expected HSV colour range for the given card type.

    The UL student card (and similar designs) has a coloured header across
    the top ~30% of the card; checking that region is far more reliable than
    sampling the centre, which is mostly white.

    Args:
      card_img:  BGR image of the cropped, perspective-corrected card.
      card_type: key into CARD_COLOUR_RANGES.

    Returns:
      (is_valid: bool, confidence: float 0.0–1.0)
    """
    if card_type not in CARD_COLOUR_RANGES:
        return False, 0.0

    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = CARD_COLOUR_RANGES[card_type]
    lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
    upper = np.array([h_hi, s_hi, v_hi], dtype=np.uint8)

    # Sample top 30% of card where the coloured header band sits
    card_h = card_img.shape[0]
    top_band = card_img[:int(card_h * 0.30), :]

    hsv  = cv2.cvtColor(top_band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / mask.size

    if ratio < 0.15:  # need at least 15% matching pixels in the header band
        return False, 0.0

    # Scale ratio to a 0–1 confidence (saturates at ~50% coverage)
    confidence = min(ratio * 2.0, 1.0)
    return True, round(confidence, 3)


def detect_card_type(card_img):
    """
    Try all known card types and return the best matching one.

    Returns:
      (card_type: str | None, confidence: float)
    """
    best_type       = None
    best_confidence = 0.0

    for card_type in CARD_COLOUR_RANGES:
        valid, confidence = validate_colour(card_img, card_type)
        if valid and confidence > best_confidence:
            best_type       = card_type
            best_confidence = confidence

    return best_type, best_confidence
