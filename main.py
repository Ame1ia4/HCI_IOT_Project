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


# =========================
# COLOR FIX (IMPORTANT)
# =========================
def fix_color(frame):
    # Convert RGB → BGR (OpenCV standard)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Reduce blue tint manually
    b, g, r = cv2.split(frame)

    b = cv2.multiply(b, 0.85)   # reduce blue
    r = cv2.multiply(r, 1.15)   # boost red

    frame = cv2.merge([b, g, r])

    return frame


def get_camera():
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


def run_validators(card_img):
    card_type, colour_conf = detect_card_type(card_img)

    card_img = cv2.resize(card_img, (224, 224))

    text = extract_text(card_img)
    text_conf = keyword_confidence(text, card_type if card_type else "ul_student")

    student_number = extract_student_number(text, card_img)
    name_found = has_name(text)
    name = extract_name(text, card_img)

    layout_valid, layout_conf = validate_layout(card_img, card_type or "ul_student")
    ml_valid, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # 🔥 KEY FIX: allow keyword OR colour
    keyword_hit = any(k.lower() in text.lower() for k in ["ul", "student", "university"])

    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"] +
        layout_conf * w["layout"] +
        ml_conf     * w["ml"],
        3,
    )

    is_valid = (
        score >= config.VALIDATION_SCORE_THRESHOLD and
        layout_valid and
        (
            colour_conf > 0.5 or
            keyword_hit
        )
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


def main():
    cap = get_camera()

    frame_count = 0
    FRAME_SKIP = 2  # faster detection

    last_card = None
    last_contour = None
    no_detect = 0
    last_results = None
    ocr_frame = 0
    already_triggered = False

    COAST_FRAMES = 20
    OCR_INTERVAL = 8

    def trigger_buzzer():
        beep()

    while True:
        frame = cap.capture_array()

        # ✅ FIX COLOR
        frame = fix_color(frame)

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # ✅ Improve far detection
        frame = cv2.convertScaleAbs(frame, alpha=1.4, beta=20)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

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

            cv2.drawContours(
                frame, [contour], -1,
                (0, 255, 0) if last_results and last_results["is_valid"] else (0, 0, 255),
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

        cv2.imshow("Validator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()