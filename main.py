import cv2
import threading
import time
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

import config
from detection.card_detector import detect_card
from validation.colour_validator import detect_card_type
from validation.ocr_validator import (
    extract_text, keyword_confidence,
    extract_student_number, has_name, extract_name
)
from validation.layout_validator import validate_layout
from validation.ml_validator import predict as ml_predict, is_model_available

from comms.arduino_serial import send_result
from comms.http_client import post_result
from comms.blink import green_on
from comms.buzzer import beep

# ---------------- CAMERA SETUP ---------------- #

def get_camera():
    if config.CAMERA_SOURCE == "pi":
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not installed")
        cam = Picamera2()
        config_ = cam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        cam.configure(config_)
        cam.start()
        cam.set_controls({"AfMode": 2, "AwbEnable": True})
        return cam

    source = config.SOURCES[config.CAMERA_SOURCE]
    for attempt in range(5):
        cap = cv2.VideoCapture(source)
        time.sleep(1)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(2)

    raise RuntimeError(f"Could not open camera source '{config.CAMERA_SOURCE}': {source}")


# ---------------- VALIDATION ---------------- #

def run_validators(card_img):
    card_type, colour_conf = detect_card_type(card_img)

    if card_type is None:
        return None   # signal: not a UL card — discard silently

    text          = extract_text(card_img)
    text_conf     = keyword_confidence(text, card_type)
    student_number = extract_student_number(text, card_img)
    name_found    = has_name(text)
    name          = extract_name(text, card_img)

    layout_valid, layout_conf = validate_layout(card_img, card_type)
    _, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"]   +
        layout_conf * w["layout"] +
        (ml_conf * w.get("ml", 0.1) if is_model_available() else 0),
        3
    )

    is_valid = score >= config.VALIDATION_SCORE_THRESHOLD
    # Hard gate: zero keyword matches → not a UL card regardless of score
    if text_conf == 0.0:
        is_valid = False

    return {
        "card_type":    card_type,
        "colour_conf":  colour_conf,
        "text_conf":    text_conf,
        "layout_valid": layout_valid,
        "layout_conf":  layout_conf,
        "ml_conf":      ml_conf,
        "student_number": student_number,
        "name_found":   name_found,
        "name":         name,
        "score":        score,
        "is_valid":     is_valid,
    }


# ---------------- UI OVERLAY ---------------- #

def draw_overlay(frame, contour, results):
    if results is None or contour is None:
        return
    color = (0, 255, 0) if results["is_valid"] else (0, 0, 255)
    cv2.drawContours(frame, [contour], -1, color, 3)
    x, y, w, h = cv2.boundingRect(contour)
    label = "VALID" if results["is_valid"] else "INVALID"
    cv2.rectangle(frame, (x, y - 35), (x + 120, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    fh = frame.shape[0]
    debug_str = (f"Type: {results['card_type']} | Score: {results['score']:.2f} | "
                 f"C:{results['colour_conf']:.2f} T:{results['text_conf']:.2f} "
                 f"L:{results['layout_conf']:.2f} ML:{results['ml_conf']:.2f}")
    cv2.putText(frame, debug_str, (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# ---------------- MAIN LOOP ---------------- #

def main():
    cap = get_camera()

    last_card            = None
    last_contour         = None
    locked_contour       = None   # frozen at moment of valid scan
    no_detect            = 0
    last_results         = None
    ocr_frame            = 0
    frame_count          = 0
    already_triggered    = False
    validator_running    = False
    validation_attempts  = 0
    debug_mode           = False

    COAST_FRAMES = 15
    OCR_INTERVAL = 10
    FRAME_SKIP   = 2

    print("System Active. Press 'Q' to quit, 'D' for debug.")

    def run_async(card_img, snap_contour):
        nonlocal last_results, validator_running, already_triggered
        nonlocal locked_contour, validation_attempts
        results = run_validators(card_img)
        if results is None:          # not a UL card — ignore, don't show INVALID
            validator_running = False
            return
        validation_attempts += 1
        results["attempts"] = validation_attempts
        last_results = results
        if config.SERIAL_ENABLED:
            send_result(results["is_valid"])
        if config.ENDPOINT_ENABLED:
            post_result(results)
        if results["is_valid"] and not already_triggered:
            locked_contour      = snap_contour
            already_triggered   = True
            validation_attempts = 0
            threading.Thread(target=green_on, daemon=True).start()
            threading.Thread(target=beep, daemon=True).start()
        elif not results["is_valid"]:
            already_triggered = False
        validator_running = False

    while True:
        # 1. CAPTURE
        if config.CAMERA_SOURCE == "pi":
            frame_rgb = cap.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        fw, fh = config.FRAME_WIDTH, config.FRAME_HEIGHT

        # 2. ROI GUIDE BOX — crop centre region to eliminate background
        roi_x1 = fw // 8;  roi_y1 = fh // 6
        roi_x2 = fw - fw // 8;  roi_y2 = fh - fh // 6
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (180, 180, 180), 2)
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # 3. DETECTION (inside ROI only)
        result  = detect_card(roi, debug=debug_mode)
        card_img = result[0] if result else None
        contour  = result[1] if result else None
        edges    = result[2] if (result and len(result) == 3) else None

        # Shift contour back to full-frame coordinates
        if contour is not None:
            contour = contour + np.array([roi_x1, roi_y1])

        # 4. COASTING — keep last card for a few frames to avoid flicker
        if card_img is not None:
            last_card    = card_img
            last_contour = contour
            no_detect    = 0
        else:
            no_detect += 1
            if no_detect <= COAST_FRAMES and last_card is not None:
                card_img = last_card
                contour  = last_contour

        # 5. VALIDATE + DISPLAY
        if card_img is not None:
            ocr_frame += 1
            # Only run validator when not locked (already_triggered) and not busy
            if not validator_running and not already_triggered and \
               (last_results is None or ocr_frame % OCR_INTERVAL == 0):
                validator_running = True
                threading.Thread(target=run_async, args=(card_img, contour), daemon=True).start()

            if last_results is not None:
                display_contour = locked_contour if (already_triggered and locked_contour is not None) else contour
                draw_overlay(frame, display_contour, last_results)
        else:
            # Card fully gone — reset session
            if last_results is not None:
                if config.ENDPOINT_ENABLED:
                    post_result({"session_reset": True})
            last_results         = None
            locked_contour       = None
            already_triggered    = False
            ocr_frame            = 0
            validation_attempts  = 0
            cv2.putText(frame, "No card detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

        # 6. DISPLAY
        cv2.imshow("Card Validator", frame)
        if debug_mode and edges is not None:
            cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug_mode = not debug_mode

    if config.CAMERA_SOURCE != "pi":
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
