import cv2
import numpy as np

from config import (
    CARD_ASPECT_RATIO,
    ASPECT_RATIO_TOL,
    MIN_CARD_AREA,
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    CARD_COLOUR_RANGES,
)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(frame, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    dst_w, dst_h = 856, 540
    dst = np.array([
        [0,         0        ],
        [dst_w - 1, 0        ],
        [dst_w - 1, dst_h - 1],
        [0,         dst_h - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (dst_w, dst_h))


def _detect_by_green_band(frame):
    """
    Fallback detector: find the UL green band by colour, then derive the full
    card bounding box from it.

    The green band sits in roughly the middle 40% of the card height.
    So if the band has height B, the full card height is B / 0.4 and
    extends ~1.5*B above and below the band centre.

    Returns (warped, contour_pts) or (None, None).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = CARD_COLOUR_RANGES["ul_student"]
    mask = cv2.inRange(hsv,
                       np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                       np.array([h_hi, s_hi, v_hi], dtype=np.uint8))

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    band = max(contours, key=cv2.contourArea)
    if cv2.contourArea(band) < 2000:
        return None, None

    bx, by, bw, bh = cv2.boundingRect(band)

    # The band is ~40% of card height → full card height ≈ bh / 0.40
    card_h = int(bh / 0.40)
    card_w = int(card_h * CARD_ASPECT_RATIO)

    # Centre the card box horizontally on the band centre
    cx = bx + bw // 2
    x1 = max(cx - card_w // 2, 0)
    x2 = min(cx + card_w // 2, frame.shape[1])

    # Band centre vertically — card extends card_h * 0.30 above and 0.70 below
    band_cy = by + bh // 2
    y1 = max(band_cy - int(card_h * 0.55), 0)
    y2 = min(band_cy + int(card_h * 0.55), frame.shape[0])

    if (x2 - x1) < 40 or (y2 - y1) < 40:
        return None, None

    crop = frame[y1:y2, x1:x2]
    warped = cv2.resize(crop, (856, 540))

    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    contour_out = pts.reshape(-1, 1, 2)
    return warped, contour_out


def detect_card(frame, debug=False):
    """
    Locate a UL student card in a frame.

    Strategy:
      1. Contour detection (works well when card fills frame / camera moves)
      2. Green-band colour detection fallback (works well for held cards
         against complex backgrounds where contour detection struggles)
    """
    edges = None

    # --- Primary: contour detection ---
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray    = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < MIN_CARD_AREA:
            break

        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            pts = box.astype("float32")

        rect = order_points(pts)
        tl, tr, br, bl = rect

        width  = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

        if height == 0 or width == 0:
            continue

        aspect = width / height
        if abs(aspect - CARD_ASPECT_RATIO) > ASPECT_RATIO_TOL:
            if aspect == 0 or abs((1 / aspect) - CARD_ASPECT_RATIO) > ASPECT_RATIO_TOL:
                continue

        warped = perspective_transform(frame, pts)
        contour_out = approx if len(approx) == 4 else box.astype(np.int32).reshape(-1, 1, 2)
        if debug:
            return warped, contour_out, edges
        return warped, contour_out

    # --- Fallback: green-band colour detection ---
    warped, contour_out = _detect_by_green_band(frame)
    if warped is not None:
        if debug:
            return warped, contour_out, edges
        return warped, contour_out

    if debug:
        return None, None, edges
    return None, None
