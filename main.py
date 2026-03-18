import threading

import cv2
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

import config
from detection.card_detector import detect_card
from validation.colour_validator import detect_card_type
from validation.ocr_validator import extract_text, keyword_confidence, extract_student_number
from validation.supabase_validator import log_scan
from validation.layout_validator import validate_layout
from validation.ml_validator import predict as ml_predict, is_model_available
from comms.arduino_serial import send_result
from comms.http_client import post_result
from comms.blink import green_on
from comms.buzzer import beep

# ---------------- CAMERA ---------------- #

def get_camera():
    if config.CAMERA_SOURCE == "pi":
        cam = Picamera2()
        cam.start()
        cam.set_controls({"AfMode": 2})
        return cam

    source = config.SOURCES[config.CAMERA_SOURCE]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{config.CAMERA_SOURCE}': {source}"
        )
    return cap


# ---------------- VALIDATION ---------------- #

def run_validators(card_img):
    """
    Run all validators against the cropped card image.

    Returns a dict with keys:
      card_type, colour_conf, text_conf, layout_valid, layout_conf,
      student_number, ml_conf, score, is_valid
    """
    # 1. Colour — also determines which card type we're dealing with
    card_type, colour_conf = detect_card_type(card_img)

    if card_type is None:
        return {
            "card_type":      None,
            "colour_conf":    0.0,
            "text_conf":      0.0,
            "layout_valid":   False,
            "layout_conf":    0.0,
            "ml_conf":        0.0,
            "student_number": None,
            "score":          0.0,
            "is_valid":       False,
        }

    # 2. OCR
    text           = extract_text(card_img)
    text_conf      = keyword_confidence(text, card_type)
    student_number = extract_student_number(text, card_img)

    # 3. Layout
    layout_valid, layout_conf = validate_layout(card_img, card_type)

    # 4. ORB feature matching (only used if reference image is available)
    ml_valid, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # 5. Weighted score
    w = config.VALIDATION_WEIGHTS
    score = round(
        colour_conf * w["colour"] +
        text_conf   * w["text"]   +
        layout_conf * w["layout"] +
        ml_conf     * w["ml"],
        3,
    )

    is_valid = score >= config.VALIDATION_SCORE_THRESHOLD

    return {
        "card_type":      card_type,
        "colour_conf":    colour_conf,
        "text_conf":      text_conf,
        "layout_valid":   layout_valid,
        "layout_conf":    layout_conf,
        "ml_conf":        ml_conf,
        "student_number": student_number,
        "score":          score,
        "is_valid":       is_valid,
    }


# ---------------- OVERLAY ---------------- #

def draw_overlay(frame, contour, results):
    colour = (0, 200, 0) if results["is_valid"] else (0, 0, 220)
    label  = "VALID" if results["is_valid"] else "INVALID"

    cv2.drawContours(frame, [contour], -1, colour, 3)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y - 36), (x + w, y), colour, -1)
    cv2.putText(
        frame, label,
        (x + 6, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
    )

    id_str = f"ID: {results['student_number']}" if results["student_number"] else "ID: not found"
    debug_lines = [
        f"Type:   {results['card_type'] or 'unknown'}",
        f"Colour: {results['colour_conf']:.2f}  "
        f"Text: {results['text_conf']:.2f}  "
        f"Layout: {results['layout_conf']:.2f}  "
        f"ML: {results['ml_conf']:.2f}",
        f"Score:  {results['score']:.2f}  {id_str}",
    ]

    fh = frame.shape[0]
    for i, line in enumerate(debug_lines):
        cv2.putText(
            frame, line,
            (10, fh - 20 - (len(debug_lines) - 1 - i) * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )


# ---------------- MAIN ---------------- #

def main():
    cap = get_camera()
    print(f"Camera source: {config.CAMERA_SOURCE} — press Q to quit, D to toggle debug view")

    debug             = False
    last_card         = None
    last_contour      = None
    no_detect         = 0
    last_results      = None
    ocr_frame         = 0
    already_triggered = False

    COAST_FRAMES = 20
    OCR_INTERVAL = 8

    already_triggered = False
    buzzer_active = False

    def trigger_buzzer():
        nonlocal buzzer_active
        if not buzzer_active:
            buzzer_active = True
            beep()
            buzzer_active = False

    while True:
        if config.CAMERA_SOURCE == "pi":
            frame = cap.capture_array()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame — check camera connection")
                break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        card_img, contour, edges = detect_card(frame, debug=True)

        if card_img is not None:
            last_card    = card_img
            last_contour = contour
            no_detect    = 0
        else:
            no_detect += 1
            if no_detect <= COAST_FRAMES and last_card is not None:
                card_img = last_card
                contour  = last_contour

        if card_img is not None:
            ocr_frame += 1
            if last_results is None or ocr_frame % OCR_INTERVAL == 0:
                last_results = run_validators(card_img)
                send_result(last_results["is_valid"])
                post_result(last_results)
                log_scan(last_results["is_valid"])

                if last_results["is_valid"] and not already_triggered:
                    threading.Thread(target=green_on).start()
                    threading.Thread(target=trigger_buzzer).start()
                    already_triggered = True

                if not last_results["is_valid"]:
                    already_triggered = False

            draw_overlay(frame, contour, last_results)
            if debug:
                cv2.imshow("Warped Card", card_img)
        else:
            last_results = None
            ocr_frame    = 0
            cv2.putText(
                frame, "No card detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2,
            )

        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Disability Card Validator", display_frame)

        if debug:
            cv2.imshow("Canny Edges", edges)
        else:
            for win in ("Canny Edges", "Warped Card"):
                try:
                    cv2.destroyWindow(win)
                except cv2.error:
                    pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug
            print(f"Debug mode {'ON' if debug else 'OFF'}")

    if config.CAMERA_SOURCE != "pi":
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()