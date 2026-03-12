import cv2
import numpy as np

from config import (
    CARD_ASPECT_RATIO,
    ASPECT_RATIO_TOL,
    MIN_CARD_AREA,
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
)


def order_points(pts):
    """
    Order 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    Required for a consistent perspective transform regardless of detection order.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective_transform(frame, pts):
    """
    Apply a four-point perspective transform to produce a flat, top-down crop.
    Output size is fixed to standard ID-1 card proportions (856 x 540 px).
    """
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


def detect_card(frame, debug=False):
    """
    Locate a parking permit card in a camera frame.

    Steps:
      1. Grayscale + Gaussian blur to reduce noise
      2. Canny edge detection
      3. Contour search — filter by area and aspect ratio (~1.585 for ID-1 cards)
      4. Perspective warp to produce a flat, top-down crop

    Args:
      frame: BGR camera frame.
      debug: When True, also return the Canny edge image for visualisation.

    Returns:
      (warped, contour, edges) when debug=True, or (warped, contour) otherwise.
      warped and contour are None if no card is found.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    # Dilate edges slightly to close small gaps in card borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours     = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:  # only check the 10 largest contours
        area = cv2.contourArea(contour)
        if area < MIN_CARD_AREA:
            break  # contours are sorted so no point checking further

        # Try approxPolyDP first; fall back to minimum-area rectangle for
        # laminated cards whose rounded corners produce more than 4 vertices.
        peri   = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            pts = box.astype("float32")
        rect = order_points(pts)
        tl, tr, br, bl = rect

        width  = max(
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl),
        )
        height = max(
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr),
        )

        if height == 0 or width == 0:
            continue

        aspect = width / height
        if abs(aspect - CARD_ASPECT_RATIO) > ASPECT_RATIO_TOL:
            # Try the flipped ratio in case card is held portrait
            if aspect == 0 or abs((1 / aspect) - CARD_ASPECT_RATIO) > ASPECT_RATIO_TOL:
                continue

        warped = perspective_transform(frame, pts)
        contour_out = approx if len(approx) == 4 else box.astype(np.int32).reshape(-1, 1, 2)
        if debug:
            return warped, contour_out, edges
        return warped, contour_out

    if debug:
        return None, None, edges
    return None, None
