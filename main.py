import cv2
import threading

import config
from detection.card_detector import detect_card
from validation.colour_validator import detect_card_type
from validation.ocr_validator import (
    extract_text, keyword_confidence,
    extract_student_number, has_name
)
from validation.layout_validator import validate_layout
from validation.ml_validator import predict as ml_predict, is_model_available

from comms.arduino_serial import send_result
from comms.http_client import post_result
from comms.blink import green_on
from comms.buzzer import beep

from picamera2 import Picamera2


# ---------------- COLOR FIX ---------------- #

def fix_color(frame):
    # PiCamera2 gives RGB → convert to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Reduce blue tint
    b, g, r = cv2.split(frame)
    b = cv2.multiply(b, 0.85)
    r = cv2.multiply(r, 1.15)

    return cv2.merge([b, g, r])


# ---------------- CAMERA ---------------- #

def get_camera():
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


# ---------------- VALIDATION ---------------- #

def run_validators(card_img):

    card_type, colour_conf = detect_card_type(card_img)

    card_img = cv2.resize(card_img, (224, 224))

    text = extract_text(card_img)
    text_conf = keyword_confidence(text, card_type or "ul_student")

    student_number = extract_student_number(text, card_img)
    name_found = has_name(text)

    layout_valid, layout_conf = validate_layout(card_img, card_type or "ul_student")

    _, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # ✅ Keyword fallback (VERY IMPORTANT)
    keyword_hit = any(k in text.lower() for k in ["ul", "student", "university"])

    w = config.VALIDATION_WEIGHTS

    score = (
        colour_conf * w["colour"] +
        text_conf * w["text"] +
        layout_conf * w["layout"] +
        ml_conf * w["ml"]
    )

    score = round(score, 3)

    # 🔴 FINAL STRICT RULES (balanced)
    is_valid = (
        score >= config.VALIDATION_SCORE_THRESHOLD and
        layout_valid and
        (
            colour_conf > 0.5 or
            keyword_hit
        ) and
        (
            student_number is not None or
            name_found
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
        "score": score,
        "is_valid": is_valid,
    }


# ---------------- OVERLAY ---------------- #

def draw_overlay(frame, contour, results):

    colour = (0, 200, 0) if results["is_valid"] else (0, 0, 220)
    label = "VALID" if results["is_valid"] else "INVALID"

    cv2.drawContours(frame, [contour], -1, colour, 3)

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(frame, (x, y - 36), (x + w, y), colour, -1)

    cv2.putText(
        frame, label,
        (x + 6, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )

    ml_str = f"ML: {results['ml_conf']:.2f}" if is_model_available() else "ML: n/a"
    id_str = f"ID: {results['student_number']}" if results["student_number"] else "ID: not found"
    name_str = "Name: yes" if results["name_found"] else "Name: no"

    debug_lines = [
        f"Type: {results['card_type'] or 'unknown'}",
        f"Colour: {results['colour_conf']:.2f}  Text: {results['text_conf']:.2f}  Layout: {results['layout_conf']:.2f}  {ml_str}",
        f"Score: {results['score']:.2f}  {id_str}  {name_str}",
    ]

    fh = frame.shape[0]

    for i, line in enumerate(debug_lines):
        cv2.putText(
            frame,
            line,
            (10, fh - 20 - (len(debug_lines) - 1 - i) * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


# ---------------- MAIN ---------------- #

def main():

    cap = get_camera()

    print("Press Q to quit, D to toggle debug")

    debug = False
    last_card = None
    last_contour = None
    no_detect = 0
    last_results = None
    ocr_frame = 0
    already_triggered = False  # ✅ FIXED

    COAST_FRAMES = 20
    OCR_INTERVAL = 8

    while True:

        frame = cap.capture_array()

        # ✅ FIX COLOR
        frame = fix_color(frame)

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

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
                    threading.Thread(target=beep, daemon=True).start()
                    already_triggered = True

                if not last_results["is_valid"]:
                    already_triggered = False

            draw_overlay(frame, contour, last_results)

            if debug:
                cv2.imshow("Warped Card", card_img)

        else:
            last_results = None
            ocr_frame = 0

            cv2.putText(
                frame,
                "No card detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )

        cv2.imshow("Validator", frame)

        if debug:
            cv2.imshow("Canny Edges", edges)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("d"):
            debug = not debug

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()