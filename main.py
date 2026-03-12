import cv2

import config
from detection.card_detector import detect_card
from validation.colour_validator import detect_card_type
from validation.ocr_validator import extract_text, keyword_confidence, extract_student_number, has_name
from validation.layout_validator import validate_layout
from comms.arduino_serial import send_result
from comms.http_client import post_result
from validation.ml_validator import predict as ml_predict, is_model_available


def get_camera():
    source = config.SOURCES[config.CAMERA_SOURCE]
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{config.CAMERA_SOURCE}': {source}"
        )
    return cap


def run_validators(card_img):
    """
    Run all three rule-based validators against the cropped card image.

    Returns a dict with keys:
      card_type, colour_conf, text_conf, layout_valid, layout_conf,
      ml_conf, student_number, name_found, score, is_valid
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
            "name_found":     False,
            "score":          0.0,
            "is_valid":       False,
        }

    # 2. OCR
    text           = extract_text(card_img)
    text_conf      = keyword_confidence(text, card_type)
    student_number = extract_student_number(text)
    name_found     = has_name(text)

    # 3. Layout
    layout_valid, layout_conf = validate_layout(card_img, card_type)

    # 4. ML (only when model is available)
    _, ml_conf = ml_predict(card_img) if is_model_available() else (False, 0.0)

    # 5. Weighted score
    # When ML model is available it contributes 20% and layout drops to 10%.
    w = config.VALIDATION_WEIGHTS
    if is_model_available():
        score = (
            colour_conf * w["colour"] +
            text_conf   * w["text"]   +
            layout_conf * 0.1         +
            ml_conf     * 0.2
        )
    else:
        score = (
            colour_conf * w["colour"] +
            text_conf   * w["text"]   +
            layout_conf * w["layout"]
        )
    score = round(score, 3)

    # Card is valid only when score threshold is met AND a student number is found
    is_valid = score >= config.VALIDATION_SCORE_THRESHOLD and student_number is not None

    return {
        "card_type":      card_type,
        "colour_conf":    colour_conf,
        "text_conf":      text_conf,
        "layout_valid":   layout_valid,
        "layout_conf":    layout_conf,
        "ml_conf":        ml_conf,
        "student_number": student_number,
        "name_found":     name_found,
        "score":          score,
        "is_valid":       is_valid,
    }


def draw_overlay(frame, contour, results):
    """Draw bounding box and VALID/INVALID label onto the frame in-place."""
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

    # Debug info in bottom-left corner
    ml_str  = f"ML: {results['ml_conf']:.2f}" if is_model_available() else "ML: n/a"
    id_str  = f"ID: {results['student_number']}" if results["student_number"] else "ID: not found"
    name_str = "Name: yes" if results["name_found"] else "Name: no"
    debug_lines = [
        f"Type:   {results['card_type'] or 'unknown'}",
        f"Colour: {results['colour_conf']:.2f}  "
        f"Text: {results['text_conf']:.2f}  "
        f"Layout: {results['layout_conf']:.2f}  {ml_str}",
        f"Score:  {results['score']:.2f}  {id_str}  {name_str}",
    ]
    fh = frame.shape[0]
    for i, line in enumerate(debug_lines):
        cv2.putText(
            frame, line,
            (10, fh - 20 - (len(debug_lines) - 1 - i) * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )


def main():
    cap = get_camera()
    print(f"Camera source: {config.CAMERA_SOURCE} — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame — check camera connection")
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        card_img, contour = detect_card(frame)

        if card_img is not None:
            results = run_validators(card_img)
            draw_overlay(frame, contour, results)
            send_result(results["is_valid"])
            post_result(results)
        else:
            cv2.putText(
                frame, "No card detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2,
            )

        cv2.imshow("Disability Card Validator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
