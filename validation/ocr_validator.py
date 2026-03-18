import re

import cv2
import numpy as np
import pytesseract

from config import CARD_KEYWORDS

# Point pytesseract at the default Windows install location
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

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


def _preprocess_strip(region):
    """
    Prepare a cropped card region for Tesseract:
    grayscale → denoise → upscale 3x → Otsu threshold.
    Upscaling to ~300 dpi equivalent gives Tesseract much cleaner input.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_student_number(text, card_img=None):
    """
    Search for a student ID number (7–9 consecutive digits).

    When card_img is provided, first tries a targeted OCR pass on the bottom
    20% of the card (where the UL ID number is printed) using single text-line
    mode with a digit-only whitelist — more accurate than full-card sparse OCR.

    Falls back to a regex scan of the full OCR text.

    Returns:
      The matched string if found, otherwise None.
    """
    if card_img is not None:
        try:
            h = card_img.shape[0]
            # Bottom 20% — below the green band, white background with ID number
            bottom = card_img[int(h * 0.80):, :]
            thresh = _preprocess_strip(bottom)
            strip_text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=0123456789",
            )
            match = re.search(r"(\d{7,9})", strip_text)
            if match:
                return match.group(1)
        except Exception:
            pass

    # Fallback: regex over full OCR text
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


def _parse_name_from_text(text):
    """
    Parse a name out of raw OCR text using label-then-value and heuristic strategies.
    Returns a Title Case name string or None.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Strategy 1: line immediately after a NAME / AINM label
    for i, line in enumerate(lines[:-1]):
        if re.search(r"\b(NAME|AINM)\b", line, re.IGNORECASE):
            candidate = lines[i + 1]
            words = [w.upper().strip(".,") for w in candidate.split()]
            name_words = [w for w in words if w.isalpha() and len(w) >= 2 and w not in _CARD_VOCAB]
            if len(name_words) >= 2:
                return candidate.title()

    # Strategy 2: first line that looks like a name
    for line in lines:
        words = [w.upper().strip(".,") for w in line.split()]
        name_words = [w for w in words if w.isalpha() and len(w) >= 2 and w not in _CARD_VOCAB]
        if len(name_words) >= 2:
            return line.title()

    return None


def extract_name(text, card_img=None):
    """
    Attempt to extract the cardholder's name.

    When card_img is provided, first runs a targeted OCR pass on the top-right
    of the card — on the UL student card the name field sits in roughly the
    top 35% of the card, right of the photo (~40% from left).

    Falls back to parsing the full-card OCR text.

    Returns:
      The name string in Title Case, or None if not found.
    """
    if card_img is not None:
        try:
            h, w = card_img.shape[:2]
            # Top-right quadrant: top 35% of height, right 60% of width
            name_region = card_img[0:int(h * 0.35), int(w * 0.40):]
            thresh = _preprocess_strip(name_region)
            region_text = pytesseract.image_to_string(
                thresh,
                config="--psm 6",  # assume uniform block of text
            )
            result = _parse_name_from_text(region_text)
            if result:
                return result
        except Exception:
            pass

    # Fallback: parse full-card OCR text
    return _parse_name_from_text(text)
