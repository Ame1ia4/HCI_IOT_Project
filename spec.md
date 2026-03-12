# Ireland Disability Card Validation – Computer Vision Project

## Project Overview

Build a Python-based computer vision pipeline that uses a camera (phone, webcam, or ESP32-CAM) to capture an image of an Irish disability card and determine whether it is a **valid, authentic card** based on its visual appearance, layout, colours, text structure, and security features.

This is the **Python/OpenCV side only**. Arduino communication is handled separately.

---

## Goals

- Detect and locate a disability card within a camera frame
- Crop and perspective-correct the card to a standard orientation
- Validate the card against known visual characteristics of genuine Irish disability cards
- Output a clear `VALID` or `INVALID` result
- Send that result over serial to an Arduino (optional at this stage)
- Be camera-agnostic — switching between phone stream, webcam, and ESP32-CAM should only require changing one variable

---

## Target Cards

The system should be able to validate the following Irish disability cards:

- **Primary Medical Card** (Ireland HSE) — dark green
- **GP Visit Card** (Ireland HSE) — light green/teal
- **European Health Insurance Card (EHIC)** — blue, issued in Ireland
- **Disabled Parking Permit (Disabled Drivers Association of Ireland)** — orange/yellow
- **ILMI / Disability Federation membership cards** (if applicable)

> **Note:** Focus initially on the **HSE Medical Card** and **Disabled Parking Permit** as they are the most visually distinctive and common.

---

## Tech Stack

| Purpose | Library |
|---|---|
| Image capture & processing | `opencv-python` |
| Numerical operations | `numpy` |
| OCR (text reading) | `pytesseract` + Tesseract engine |
| ML classification (phase 2) | `tensorflow` + `tensorflow-lite` |
| Serial comms to Arduino | `pyserial` |
| Image augmentation (training) | `albumentations` |
| Utility / visualisation | `matplotlib` |

### Install all dependencies
```bash
pip install opencv-python numpy pytesseract pyserial tensorflow albumentations matplotlib
```
> Also install the Tesseract OCR engine: https://github.com/UB-Mannheim/tesseract/wiki

---

## Project Structure

```
disability_card_cv/
├── main.py                  # Entry point — runs the live camera loop
├── config.py                # Camera source, thresholds, serial port config
├── detection/
│   └── card_detector.py     # Finds and crops card from camera frame
├── validation/
│   ├── colour_validator.py  # Checks card colour profile
│   ├── ocr_validator.py     # Reads and checks text (HSE, name, expiry etc.)
│   ├── layout_validator.py  # Checks logo/text position and layout
│   └── ml_validator.py      # ML model inference (phase 2)
├── comms/
│   └── arduino_serial.py    # Sends VALID/INVALID over serial to Arduino
├── training/
│   ├── collect_samples.py   # Helper to save card images from camera
│   └── train_model.py       # Train a TFLite classifier
├── models/
│   └── card_model.tflite    # Trained model (generated, not committed)
├── data/
│   ├── valid/               # Sample images of valid cards (for training)
│   └── invalid/             # Sample images of invalid/fake cards
└── requirements.txt
```

---

## Phase 1 — Rule-Based Validation (No ML, start here)

Use OpenCV and OCR to validate cards based on known visual rules. No training data needed.

### Step 1: Camera Input

```python
# config.py
CAMERA_SOURCE = "phone"  # "phone" | "webcam" | "esp32" | "pi"

SOURCES = {
    "webcam": 0,
    "phone":  "http://192.168.1.100:8080/video",  # IP Webcam app
    "esp32":  "http://192.168.1.105/stream",
    "pi":     0
}
```

### Step 2: Card Detection

Use contour detection to find the card in the frame:

1. Convert frame to grayscale
2. Apply Gaussian blur
3. Use Canny edge detection
4. Find contours and filter for rectangular shapes with card-like aspect ratio (~85.6mm × 54mm → ratio ~1.585)
5. Apply perspective transform to get a flat, top-down crop of the card

```python
# detection/card_detector.py
# Expected functions:
# - detect_card(frame) → returns cropped, warped card image or None
# - order_points(pts) → helper for perspective transform
```

### Step 3: Colour Validation

Check the dominant colour of the card matches known Irish disability card colours:

| Card | Primary Colour (approx HSV) |
|---|---|
| HSE Medical Card | Dark green — H:90-150, S:40-255, V:30-180 |
| GP Visit Card | Teal/light green — H:80-100, S:60-200, V:100-220 |
| Disabled Parking Permit | Orange/yellow — H:15-35, S:150-255, V:150-255 |
| EHIC | Blue — H:100-130, S:80-255, V:80-220 |

```python
# validation/colour_validator.py
# Expected functions:
# - get_dominant_colour(card_img) → returns HSV values
# - validate_colour(card_img, card_type) → returns (bool, confidence_score)
```

### Step 4: OCR Text Validation

Use Tesseract to read text from the card and validate expected keywords:

| Card | Expected Text |
|---|---|
| HSE Medical Card | "HSE", "Health Service Executive", expiry date format DD/MM/YYYY |
| Disabled Parking Permit | "Disabled Parking", "DDAI" or "Ireland", permit number |
| EHIC | "European Health Insurance Card", country code "IE" |

```python
# validation/ocr_validator.py
# Expected functions:
# - extract_text(card_img) → returns raw string from Tesseract
# - validate_text(text, card_type) → returns (bool, matched_keywords)
# - extract_expiry(text) → returns datetime or None
# - is_expired(expiry_date) → returns bool
```

### Step 5: Layout Validation

Check that logos, text blocks, and design elements appear in expected positions:

- Divide the card into a grid (e.g. 3×2)
- Check that the HSE logo region (top-left) contains the correct colour blob
- Check that text regions are in expected zones

```python
# validation/layout_validator.py
# Expected functions:
# - validate_layout(card_img, card_type) → returns (bool, confidence)
```

### Step 6: Combine Results & Output

```python
# main.py logic
results = {
    "colour":  colour_validator.validate_colour(card),
    "text":    ocr_validator.validate_text(text),
    "layout":  layout_validator.validate_layout(card),
    "expired": ocr_validator.is_expired(expiry)
}

# Weighted decision
score = (
    results["colour"][1]  * 0.3 +
    results["text"][1]    * 0.5 +
    results["layout"][1]  * 0.2
)

is_valid = score > 0.75 and not results["expired"]
```

---

## Phase 2 — ML Classification (after phase 1 works)

Train a MobileNetV2 transfer learning model to classify card images as valid or invalid.

### Data Collection

Use `training/collect_samples.py` to capture card images from the live camera and save them into `data/valid/` or `data/invalid/`.

Aim for at least:
- **200+ images** of valid cards (different angles, lighting, distances)
- **200+ images** of invalid cards (wrong cards, fake cards, printed fakes, no card)

### Training

```python
# training/train_model.py
# - Load images from data/valid and data/invalid
# - Apply augmentation (rotation, brightness, blur) using albumentations
# - Fine-tune MobileNetV2 (pretrained on ImageNet)
# - Export as card_model.tflite for deployment
```

### Inference

```python
# validation/ml_validator.py
# - Load card_model.tflite
# - Preprocess card image to 224x224, normalise to [0,1]
# - Run inference
# - Return (is_valid: bool, confidence: float)
```

---

## Arduino Serial Communication (Optional)

```python
# comms/arduino_serial.py
import serial

def send_result(is_valid: bool, port="/dev/ttyUSB0", baud=9600):
    with serial.Serial(port, baud) as ser:
        msg = b'VALID\n' if is_valid else b'INVALID\n'
        ser.write(msg)
```

On the Arduino side, it reads this serial message and triggers LEDs, a buzzer, or a servo accordingly.

---

## Validation Logic Summary

```
Camera Frame
     ↓
Card Detection (contours + perspective warp)
     ↓
┌────────────────────────────────┐
│  Rule-Based Checks (Phase 1)  │
│  • Colour match                │
│  • OCR keyword check           │
│  • Expiry date check           │
│  • Layout check                │
└────────────────────────────────┘
     ↓ (Phase 2 — add ML on top)
┌────────────────────────────────┐
│  ML Classification             │
│  • MobileNetV2 TFLite model    │
│  • Confidence score            │
└────────────────────────────────┘
     ↓
Combined Score → VALID / INVALID
     ↓
Serial → Arduino (LED / buzzer / servo)
```

---

## Important Considerations

### Legal & Privacy
- Do **not** store or log card images or personal data captured during validation
- Only process frames in memory — never write card images to disk in production
- This system validates **visual appearance only** — it cannot verify against any government database

### Robustness
- Test under different lighting conditions (indoor, outdoor, fluorescent)
- Test at different angles before the perspective warp kicks in
- Cards may be worn, laminated, or slightly faded — build tolerance into colour thresholds

### Camera Resolution
- ESP32-CAM outputs 640×480 or lower — resize all test frames to this resolution during development so your thresholds generalise:
```python
frame = cv2.resize(frame, (640, 480))
```

---

## Deliverables / Definition of Done

- [ ] Camera feed displays live with card bounding box overlay
- [ ] Card is correctly cropped and perspective-corrected
- [ ] Colour validation works for at least HSE Medical Card
- [ ] OCR extracts and validates "HSE" keyword and expiry date
- [ ] Result displayed on screen as green (VALID) or red (INVALID) overlay
- [ ] Serial message sent to Arduino on each detection
- [ ] Switching camera source requires only changing `CAMERA_SOURCE` in `config.py`
- [ ] (Phase 2) ML model trained and integrated with >85% accuracy on test set