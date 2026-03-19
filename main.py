import cv2
import threading
import time

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
        # Configure for RGB to ensure color validation works correctly
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
        return {
            "card_type": None, "colour_conf": 0.0, "text_conf": 0.0,
            "layout_valid": False, "layout_conf": 0.0, "ml_conf": 0.0,
            "student_number": None, "name_found": False, "name": None,
            "score": 0.0, "is_valid": False,
        }

    # OCR and Text Analysis
    text = extract_text(card_img)
    text_conf = keyword_confidence(text, card_type)
    student_number = extract_student_number(text, card_img)
    name_found = has_name(text)
    name = extract_name(text, card_img)

    # Layout and ML Analysis
    layout_valid, layout_conf = validate_layout(card_img, card_type)
    _, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # Weighted Scoring
    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"] +
        layout_conf * w["layout"] +
        (ml_conf * w.get("ml", 0.2) if is_model_available() else 0),
        3
    )

    return {
        "card_type": card_type,
        "colour_conf": colour_conf,
        "text_conf": text_conf,
        "layout_valid": layout_valid,
        "layout_conf": layout_conf,
        "ml_conf": ml_conf,
        "student_number": student_number,
        "name_found": name_found,
        "name": name,
        "score": score,
        "is_valid": score >= config.VALIDATION_SCORE_THRESHOLD,
    }

# ---------------- UI OVERLAY ---------------- #

def draw_overlay(frame, contour, results):
    if results is None: return
    
    # Draw bounding box around card
    color = (0, 255, 0) if results["is_valid"] else (0, 0, 255) # RGB Logic
    cv2.drawContours(frame, [contour], -1, color, 3)

    # Info Labels
    x, y, w, h = cv2.boundingRect(contour)
    label = "VALID" if results["is_valid"] else "INVALID"
    cv2.rectangle(frame, (x, y - 35), (x + 120, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Debug Data at Bottom
    fh = frame.shape[0]
    debug_str = f"Type: {results['card_type']} | Score: {results['score']:.2f} | ML: {results['ml_conf']:.2f}"
    cv2.putText(frame, debug_str, (10, fh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ---------------- MAIN LOOP ---------------- #

# ... (Keep your imports the same) ...

def main():
    cap = get_camera()
    
    last_results = None
    no_detect, ocr_frame, frame_count = 0, 0, 0
    already_triggered = False
    debug_mode = False

    COAST_FRAMES = 15
    OCR_INTERVAL = 10
    FRAME_SKIP = 2 

    print("System Active. Press 'Q' to quit, 'D' for debug.")

    while True:
        # 1. CAPTURE & STANDARDIZE TO BGR
        if config.CAMERA_SOURCE == "pi":
            frame_rgb = cap.capture_array() 
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR immediately
        else:
            ret, frame = cap.read()
            if not ret: break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        # 2. DETECTION (Now consistently BGR)
        result = detect_card(frame, debug=debug_mode)
        card_img = result[0] if result else None
        contour = result[1] if result else None
        edges = result[2] if len(result) == 3 else None

        if card_img is not None:
            # 3. VALIDATION
            ocr_frame += 1
            if last_results is None or ocr_frame % OCR_INTERVAL == 0:
                last_results = run_validators(card_img)
                
                if config.SERIAL_ENABLED:
                    send_result(last_results["is_valid"])
                
                if config.ENDPOINT_ENABLED:
                    threading.Thread(target=post_result, args=(last_results,), daemon=True).start()

                if last_results["is_valid"] and not already_triggered:
                    threading.Thread(target=green_on, daemon=True).start()
                    threading.Thread(target=beep, daemon=True).start()
                    already_triggered = True
                elif not last_results["is_valid"]:
                    already_triggered = False

            draw_overlay(frame, contour, last_results)
        else:
            last_results, ocr_frame, already_triggered = None, 0, False
            cv2.putText(frame, "No card detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. DISPLAY (Already BGR, no conversion needed)
        cv2.imshow("Card Validator", frame)
        
        if debug_mode and edges is not None:
            cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        if key == ord("d"): debug_mode = not debug_mode

    if config.CAMERA_SOURCE != "pi": cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()