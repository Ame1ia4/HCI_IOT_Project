import cv2
import threading

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
from validation.supabase_validator import lookup_student
from validation.layout_validator import validate_layout
from validation.ml_validator import predict as ml_predict, is_model_available

from comms.arduino_serial import send_result
from comms.http_client import post_result
from comms.blink import green_on
from comms.buzzer import beep


# ---------------- CAMERA ---------------- #

def get_camera():
    if config.CAMERA_SOURCE == "pi":
        if Picamera2 is None:
            raise RuntimeError("Picamera2 not installed")

        cam = Picamera2()

        # ✅ FIX: use BGR directly
        config_ = cam.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        cam.configure(config_)
        cam.start()

        cam.set_controls({
            "AfMode": 2,
            "AfTrigger": 0
        })

        return cam

    source = config.SOURCES[config.CAMERA_SOURCE]
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {source}")

    return cap


# ---------------- VALIDATION ---------------- #

def run_validators(card_img):
    card_type, colour_conf = detect_card_type(card_img)

    if card_type is None:
        return {
            "card_type": None,
            "colour_conf": 0.0,
            "text_conf": 0.0,
            "layout_valid": False,
            "layout_conf": 0.0,
            "student_number": None,
            "name_found": False,
            "name": None,
            "db_found": False,
            "score": 0.0,
            "is_valid": False,
        }

    text = extract_text(card_img)
    text_conf = keyword_confidence(text, card_type)
    student_number = extract_student_number(text, card_img)
    name_found = has_name(text)

    db_found, db_name = lookup_student(student_number)
    name = db_name if db_found else extract_name(text, card_img)

    layout_valid, layout_conf = validate_layout(card_img, card_type)
    ml_valid, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"] +
        layout_conf * w["layout"] +
        ml_conf     * w["ml"],
        3,
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
        "db_found": db_found,
        "score": score,
        "is_valid": score >= config.VALIDATION_SCORE_THRESHOLD,
    }


# ---------------- MAIN ---------------- #

def main():
    cap = get_camera()

    debug = False
    last_card = None
    last_contour = None
    no_detect = 0
    last_results = None
    ocr_frame = 0
    already_triggered = False

    COAST_FRAMES = 20
    OCR_INTERVAL = 8

    buzzer_active = False

    def trigger_buzzer():
        nonlocal buzzer_active
        if not buzzer_active:
            buzzer_active = True
            beep()
            buzzer_active = False

    while True:
        # -------- CAPTURE -------- #
        if config.CAMERA_SOURCE == "pi":
            frame = cap.capture_array()
            # ❌ REMOVED: cv2.cvtColor (no longer needed)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # -------- DETECTION -------- #
        card_img, contour, edges = detect_card(frame, debug=True)

        if card_img is not None:
            last_card = card_img
            last_contour = contour
            no_detect = 0
        else:
            no_detect += 1
            if no_detect <= COAST_FRAMES and last_card is not None:
                card_img = last_card
                contour = last_contour

        # -------- VALIDATION -------- #
        if card_img is not None:
            ocr_frame += 1

            if last_results is None or ocr_frame % OCR_INTERVAL == 0:
                last_results = run_validators(card_img)

                send_result(last_results["is_valid"])
                post_result(last_results)

                if last_results["is_valid"] and not already_triggered:
                    threading.Thread(target=green_on).start()
                    threading.Thread(target=trigger_buzzer).start()
                    already_triggered = True

                if not last_results["is_valid"]:
                    already_triggered = False

            cv2.drawContours(frame, [contour], -1,
                             (0, 255, 0) if last_results["is_valid"] else (0, 0, 255), 3)

        else:
            last_results = None
            ocr_frame = 0
            cv2.putText(frame, "No card detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (100, 100, 100), 2)

        # -------- DISPLAY -------- #
        cv2.imshow("Validator", frame)

        if debug:
            cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug

    if config.CAMERA_SOURCE != "pi":
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()