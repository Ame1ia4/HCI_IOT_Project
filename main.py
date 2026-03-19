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


# =========================
# CAMERA SETUP
# =========================
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

        cam.set_controls({
            "AfMode": 2,
            "AwbEnable": True
        })

        return cam

    source = config.SOURCES[config.CAMERA_SOURCE]

    for attempt in range(5):
        cap = cv2.VideoCapture(source)
        time.sleep(1)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(2)

    raise RuntimeError(f"Could not open camera: {source}")


# =========================
# VALIDATION LOGIC (STRICT)
# =========================
def run_validators(card_img):
    card_type, colour_conf = detect_card_type(card_img)

    # HARD FAIL if no card type
    if card_type is None:
        return {
            "card_type": None,
            "colour_conf": 0.0,
            "text_conf": 0.0,
            "layout_valid": False,
            "layout_conf": 0.0,
            "ml_conf": 0.0,
            "student_number": None,
            "name_found": False,
            "name": None,
            "db_found": False,
            "score": 0.0,
            "is_valid": False,
        }

    card_img = cv2.resize(card_img, (224, 224))

    # OCR
    text = extract_text(card_img)
    text_conf = keyword_confidence(text, card_type)
    student_number = extract_student_number(text, card_img)
    name_found = has_name(text)
    name = extract_name(text, card_img)

    # Layout
    layout_valid, layout_conf = validate_layout(card_img, card_type)

    # ML
    ml_valid, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # Weighted score
    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"] +
        layout_conf * w["layout"] +
        ml_conf     * w["ml"],
        3,
    )

    # 🔴 STRICT VALIDATION RULES (THIS FIXES YOUR ISSUE)
    is_valid = (
        score >= config.VALIDATION_SCORE_THRESHOLD and
        student_number is not None and
        name_found and
        layout_valid and
        colour_conf > 0.5
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
        "db_found": False,
        "score": score,
        "is_valid": is_valid,
    }


# =========================
# MAIN LOOP
# =========================
def main():
    cap = get_camera()

    frame_count = 0
    FRAME_SKIP = 3

    last_card = None
    last_contour = None
    no_detect = 0
    last_results = None
    ocr_frame = 0
    already_triggered = False

    COAST_FRAMES = 15
    OCR_INTERVAL = 12

    def trigger_buzzer():
        beep()

    while True:
        # =========================
        # CAPTURE FRAME
        # =========================
        if config.CAMERA_SOURCE == "pi":
            frame = cap.capture_array()  # already RGB
        else:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = get_camera()
                continue

            # 🔵 FIX BLUE TINT → convert BGR → RGB for processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # =========================
        # CARD DETECTION
        # =========================
        result = detect_card(frame, debug=False)

        if len(result) == 3:
            card_img, contour, _ = result
        else:
            card_img, contour = result

        if card_img is not None:
            last_card = card_img
            last_contour = contour
            no_detect = 0
        else:
            no_detect += 1
            if no_detect <= COAST_FRAMES and last_card is not None:
                card_img = last_card
                contour = last_contour

        # =========================
        # VALIDATION
        # =========================
        if card_img is not None:
            ocr_frame += 1

            if last_results is None or ocr_frame % OCR_INTERVAL == 0:
                last_results = run_validators(card_img)

                send_result(last_results["is_valid"])

                threading.Thread(
                    target=post_result,
                    args=(last_results,),
                    daemon=True
                ).start()

                if last_results["is_valid"] and not already_triggered:
                    threading.Thread(target=green_on, daemon=True).start()
                    threading.Thread(target=trigger_buzzer, daemon=True).start()
                    already_triggered = True

                if not last_results["is_valid"]:
                    already_triggered = False

            # Draw contour
            cv2.drawContours(
                frame, [contour], -1,
                (0, 255, 0) if last_results and last_results["is_valid"] else (255, 0, 0),
                2
            )

        else:
            last_results = None
            ocr_frame = 0
            cv2.putText(
                frame, "No card detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (150, 150, 150),
                2
            )

        # =========================
        # DISPLAY (convert back to BGR)
        # =========================
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Validator", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if config.CAMERA_SOURCE != "pi":
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()