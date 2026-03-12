import cv2
import numpy as np

from config import CARD_COLOUR_RANGES


def get_dominant_colour(card_img):
    """
    Return the median HSV values of the centre 60% of the card.
    Using the centre avoids border noise and table/background bleed-in.

    Returns:
      (h, s, v) median values as a numpy array.
    """
    h, w = card_img.shape[:2]
    margin_y = int(h * 0.2)
    margin_x = int(w * 0.2)
    centre = card_img[margin_y:h - margin_y, margin_x:w - margin_x]

    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    median = np.median(hsv.reshape(-1, 3), axis=0)
    return median  # (H, S, V)


def validate_colour(card_img, card_type):
    """
    Check whether the card's dominant colour falls within the expected HSV
    range for the given card type.

    Args:
      card_img:  BGR image of the cropped, perspective-corrected card.
      card_type: key into CARD_COLOUR_RANGES, e.g. "parking_permit_ie".

    Returns:
      (is_valid: bool, confidence: float 0.0–1.0)
      Confidence reflects how centred the colour is within the expected range.
    """
    if card_type not in CARD_COLOUR_RANGES:
        return False, 0.0

    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = CARD_COLOUR_RANGES[card_type]
    h, s, v = get_dominant_colour(card_img)

    h_in = h_lo <= h <= h_hi
    s_in = s_lo <= s <= s_hi
    v_in = v_lo <= v <= v_hi

    if not (h_in and s_in and v_in):
        return False, 0.0

    # Confidence: average of how centred each channel is within its range
    def centredness(val, lo, hi):
        mid   = (lo + hi) / 2
        half  = (hi - lo) / 2
        return max(0.0, 1.0 - abs(val - mid) / half)

    confidence = (
        centredness(h, h_lo, h_hi) * 0.5 +
        centredness(s, s_lo, s_hi) * 0.3 +
        centredness(v, v_lo, v_hi) * 0.2
    )
    return True, round(float(confidence), 3)


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
