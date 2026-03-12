"""
Data collection helper — Phase 2

Opens the ESP32-CAM feed and lets you manually save card images for training.

Controls:
  V — save current frame to data/valid/
  I — save current frame to data/invalid/
  Q — quit

Aim for 200+ images in each folder, varying:
  - Distance from camera
  - Angle / slight tilt
  - Lighting conditions (indoor, outdoor, fluorescent)
  - Both IE and EU permit cards
"""

import os
import time

import cv2

import config
from detection.card_detector import detect_card

VALID_DIR   = os.path.join("data", "valid")
INVALID_DIR = os.path.join("data", "invalid")


def _next_filename(directory):
    """Return the next available numbered filename in the directory."""
    existing = [
        f for f in os.listdir(directory)
        if f.endswith(".jpg")
    ]
    return os.path.join(directory, f"{len(existing):04d}.jpg")


def main():
    os.makedirs(VALID_DIR,   exist_ok=True)
    os.makedirs(INVALID_DIR, exist_ok=True)

    source = config.SOURCES[config.CAMERA_SOURCE]
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {source}")

    valid_count   = len([f for f in os.listdir(VALID_DIR)   if f.endswith(".jpg")])
    invalid_count = len([f for f in os.listdir(INVALID_DIR) if f.endswith(".jpg")])

    print("Data collection started.")
    print(f"  Valid samples:   {valid_count}")
    print(f"  Invalid samples: {invalid_count}")
    print("Press V = save valid | I = save invalid | Q = quit")

    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # Show card detection box so user can frame the card correctly
        card_img, contour = detect_card(frame)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # HUD
        cv2.putText(frame, f"Valid: {valid_count}  Invalid: {invalid_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "V=Valid  I=Invalid  Q=Quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Collect Samples", frame)
        key = cv2.waitKey(1) & 0xFF

        # Debounce saves to avoid accidental duplicates
        now = time.time()
        if now - last_save_time < 0.5:
            continue

        if key == ord("v") and card_img is not None:
            path = _next_filename(VALID_DIR)
            cv2.imwrite(path, card_img)
            valid_count  += 1
            last_save_time = now
            print(f"Saved valid:   {path}  (total: {valid_count})")

        elif key == ord("i"):
            # Save the full frame for invalid — card may not be detected
            save_img = card_img if card_img is not None else frame
            path = _next_filename(INVALID_DIR)
            cv2.imwrite(path, save_img)
            invalid_count += 1
            last_save_time = now
            print(f"Saved invalid: {path}  (total: {invalid_count})")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Valid: {valid_count}  Invalid: {invalid_count}")


if __name__ == "__main__":
    main()
