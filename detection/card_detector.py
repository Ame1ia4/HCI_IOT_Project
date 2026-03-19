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
    dst_w, dst_h = 856, 540
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, M, (dst_w, dst_h))

def _detect_by_green_band(frame):
    """
    Fallback: Find the card based on the UL green band color.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get values from config
    vals = CARD_COLOUR_RANGES["ul_student"]
    mask = cv2.inRange(hsv, np.array(vals[0:5:2]), np.array(vals[1:6:2]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    band = max(contours, key=cv2.contourArea)
    if cv2.contourArea(band) < 1500:
        return None, None

    bx, by, bw, bh = cv2.boundingRect(band)
    card_h = int(bh / 0.40)
    card_w = int(card_h * CARD_ASPECT_RATIO)
    
    cx, cy = bx + bw // 2, by + bh // 2
    x1, y1 = max(cx - card_w // 2, 0), max(cy - card_h // 2, 0)
    x2, y2 = min(x1 + card_w, frame.shape[1]), min(y1 + card_h, frame.shape[0])

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None, None
    
    warped = cv2.resize(crop, (856, 540))
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    return warped, pts.reshape(-1, 1, 2)

def detect_card(frame, debug=False):
    """
    Main detection logic with heavy smoothing to ignore internal logos.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:8]:
        area = cv2.contourArea(contour)
        if area < MIN_CARD_AREA: break

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            rect_min = cv2.minAreaRect(contour)
            pts = cv2.boxPoints(rect_min).astype("float32")

        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        width = np.linalg.norm(tr - tl)
        height = np.linalg.norm(bl - tl)

        if height == 0 or width == 0: continue
        
        aspect = width / height
        if (abs(aspect - CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOL or 
            abs((1/aspect) - CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOL):
            
            warped = perspective_transform(frame, pts)
            contour_out = pts.astype(np.int32).reshape(-1, 1, 2)
            if debug: return warped, contour_out, edges
            return warped, contour_out

    # Call the local fallback function directly (no import needed)
    warped, contour_out = _detect_by_green_band(frame)
    
    if debug: return warped, contour_out, edges
    return warped, contour_out