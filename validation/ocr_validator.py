import re

import cv2
import pytesseract

from config import CARD_KEYWORDS

# Point pytesseract at the default Windows install location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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


def extract_student_number(text):
    """
    Search OCR text for a student ID number (7–9 consecutive digits).

    UL student numbers are 7 digits (e.g. 21234567 or 1234567).

    Returns:
      The matched string if found, otherwise None.
    """
    match = re.search(r"\b(\d{7,9})\b", text)
    return match.group(1) if match else None


def has_name(text):
    """
    Return True if OCR text contains at least one line with two or more
    capitalised alphabetic words — a heuristic for a printed name.

    Examples that pass:  'John Smith', 'Mary O Brien'
    Examples that fail:  'UNIVERSITY OF LIMERICK', 'Student Card'
    """
    for line in text.splitlines():
        words = line.strip().split()
        # Count words that start with a capital and are all-alpha (no ALL-CAPS)
        name_words = [
            w for w in words
            if len(w) > 1 and w[0].isupper() and w[1:].islower() and w.isalpha()
        ]
        if len(name_words) >= 2:
            return True
    return False
