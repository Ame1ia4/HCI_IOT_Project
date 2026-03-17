import re

import cv2
import numpy as np
import pytesseract

from config import CARD_KEYWORDS

# Point pytesseract at the default Windows install location
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_for_ocr(card_img):
    """
    Convert card to grayscale and sharpen edges.

    Adaptive thresholding is intentionally avoided: it inverts white text on
    the dark green band into unreadable black blobs. Sharpening alone gives
    Tesseract enough contrast to read both the white-on-green text (university
    name / STUDENT) and the black-on-white text (name, ID number).
    """
    gray   = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def extract_text(card_img):
    """
    Run Tesseract OCR on the card image and return the raw text string.

    Args:
      card_img: BGR image of the cropped, perspective-corrected card.

    Returns:
      Raw OCR string (may contain noise — use validate_text to interpret it).
    """
    processed = preprocess_for_ocr(card_img)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    text = pytesseract.image_to_string(
        processed,
        config="--psm 11",  # sparse text — handles mixed layouts (name + green band + ID)
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


# Words that appear on the card itself and should not be mistaken for a name
_CARD_VOCAB = {
    "NAME", "AINM", "UNIVERSITY", "LIMERICK", "OLLSCOIL", "LUIMNIGH",
    "STUDENT", "MAC", "LEINN", "LÉINN", "ID", "NO", "UIMH", "AITH", "UL", "OF",
}


def has_name(text):
    """
    Return True if OCR text contains a line with at least two consecutive
    all-alpha words that are not part of the standard UL card vocabulary.

    Handles both ALL-CAPS and Title-Case printed names.

    Examples that pass:  'CONOR CLANCY', 'John Smith', 'MARY O BRIEN'
    Examples that fail:  'UNIVERSITY OF LIMERICK', 'STUDENT MAC LEINN'
    """
    for line in text.splitlines():
        words = [w.upper().strip(".,") for w in line.strip().split()]
        name_words = [w for w in words if w.isalpha() and len(w) >= 2 and w not in _CARD_VOCAB]
        if len(name_words) >= 2:
            return True
    return False
