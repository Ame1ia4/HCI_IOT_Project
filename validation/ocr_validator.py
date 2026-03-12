import re
from datetime import datetime

import cv2
import pytesseract

from config import CARD_KEYWORDS


def preprocess_for_ocr(card_img):
    """
    Convert card to grayscale and apply adaptive thresholding to improve
    Tesseract accuracy on worn, laminated, or unevenly lit cards.
    """
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    return processed


def extract_text(card_img):
    """
    Run Tesseract OCR on the card image and return the raw text string.

    Args:
      card_img: BGR image of the cropped, perspective-corrected card.

    Returns:
      Raw OCR string (may contain noise — use validate_text to interpret it).
    """
    processed = preprocess_for_ocr(card_img)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 6",  # assume a single block of text
    )
    return text


def validate_text(text, card_type):
    """
    Check that the OCR text contains the expected keywords for the card type.
    Matching is case-insensitive.

    Args:
      text:      Raw string from extract_text().
      card_type: Key into CARD_KEYWORDS, e.g. "parking_permit_ie".

    Returns:
      (is_valid: bool, matched_keywords: list[str])
      is_valid is True when at least one expected keyword is matched.
    """
    if card_type not in CARD_KEYWORDS:
        return False, []

    text_upper    = text.upper()
    keywords      = CARD_KEYWORDS[card_type]
    matched       = [kw for kw in keywords if kw.upper() in text_upper]

    return len(matched) > 0, matched


def keyword_confidence(text, card_type):
    """
    Return a confidence score based on the fraction of expected keywords found.

    Returns:
      float 0.0–1.0
    """
    if card_type not in CARD_KEYWORDS:
        return 0.0

    keywords = CARD_KEYWORDS[card_type]
    if not keywords:
        return 0.0

    _, matched = validate_text(text, card_type)
    return round(len(matched) / len(keywords), 3)


def extract_expiry(text):
    """
    Search OCR text for a date in DD/MM/YYYY format.

    Returns:
      datetime object if found, otherwise None.
    """
    match = re.search(r"\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b", text)
    if not match:
        return None
    try:
        return datetime(
            int(match.group(3)),
            int(match.group(2)),
            int(match.group(1)),
        )
    except ValueError:
        return None


def is_expired(expiry_date):
    """
    Check whether the given expiry date has passed.

    Args:
      expiry_date: datetime object from extract_expiry(), or None.

    Returns:
      True if expired or no date found, False if still valid.
    """
    if expiry_date is None:
        return True  # treat missing expiry as expired (fail safe)
    return datetime.now() > expiry_date
