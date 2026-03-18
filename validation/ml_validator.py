"""
ML validator — ORB feature matching

Compares the detected card against all reference photos in the models/ folder.
Confidence is the best match score across all references.

Place any number of clear card photos (jpg/jpeg) in models/ — more reference
images with varied lighting/angles improves robustness.
"""

import os
import glob

import cv2

MODELS_DIR  = "models"
IMG_SIZE    = (320, 200)
MIN_MATCHES = 10
GOOD_RATIO  = 0.75


def _load_references():
    """Load ORB descriptors for every jpg/jpeg in the models folder."""
    paths = glob.glob(os.path.join(MODELS_DIR, "*.jpg")) + \
            glob.glob(os.path.join(MODELS_DIR, "*.jpeg"))

    orb  = cv2.ORB_create(nfeatures=500)
    refs = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        if desc is not None and len(kp) >= MIN_MATCHES:
            refs.append(desc)
            print(f"[ML] Loaded reference: {os.path.basename(path)} ({len(kp)} keypoints)")

    if not refs:
        print("[ML] No valid reference images found in models/ — ORB matching disabled")

    return refs, orb


_references, _orb = _load_references()


def is_model_available():
    return len(_references) > 0


def predict(card_img):
    """
    Compare card_img against all reference images, return the best match score.

    Returns:
      (is_valid: bool, confidence: float 0.0–1.0)
    """
    if not _references:
        return False, 0.0

    img  = cv2.resize(card_img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, desc = _orb.detectAndCompute(gray, None)
    if desc is None or len(kp) < MIN_MATCHES:
        return False, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_confidence = 0.0

    for ref_desc in _references:
        raw_matches = matcher.knnMatch(desc, ref_desc, k=2)
        good = [m for m, n in raw_matches if m.distance < GOOD_RATIO * n.distance]
        confidence = min(len(good) / 50.0, 1.0)
        if confidence > best_confidence:
            best_confidence = confidence

    return best_confidence >= 0.4, round(best_confidence, 3)
