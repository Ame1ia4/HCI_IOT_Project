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

def detect_card(frame, debug=False):
    """
    Locate a UL student card using improved smoothing to ignore internal patterns.
    """
    # 1. PRE-PROCESSING (Focus on big shapes, ignore logos/patterns)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Increase blur to wash out the small logo details and text
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    
    # Canny with slightly higher thresholds to ignore noise
    edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    # 2. MORPHOLOGY (Bridge the gaps in the outer border)
    # This makes the thin white lines in your 'Edges' view thicker so they connect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # 3. CONTOUR SEARCH
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:8]:
        area = cv2.contourArea(contour)
        if area < MIN_CARD_AREA:
            break

        # Approximate the shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # If it's not a perfect 4-point rect, force a bounding box around the shape
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            # Fallback: Get the minimum area rectangle (handles jagged edges better)
            rect_min = cv2.minAreaRect(contour)
            pts = cv2.boxPoints(rect_min).astype("float32")

        # 4. ASPECT RATIO VALIDATION
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        width = np.linalg.norm(tr - tl)
        height = np.linalg.norm(bl - tl)

        if height == 0 or width == 0: continue
        
        # Check both orientations (Landscape vs Portrait)
        aspect = width / height
        is_landscape = abs(aspect - CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOL
        is_portrait = abs((1/aspect) - CARD_ASPECT_RATIO) <= ASPECT_RATIO_TOL

        if is_landscape or is_portrait:
            warped = perspective_transform(frame, pts)
            # Final output pts for drawing
            contour_out = pts.astype(np.int32).reshape(-1, 1, 2)
            
            if debug: return warped, contour_out, edges
            return warped, contour_out

    # --- Fallback: green-band colour detection (existing logic) ---
    from .card_detector import _detect_by_green_band # ensure this is importable
    warped, contour_out = _detect_by_green_band(frame)
    
    if debug:
        return warped, contour_out, edges
    return warped, contour_out