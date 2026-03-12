# Disability Card Validator

A Python/OpenCV computer vision pipeline that uses an ESP32-CAM to detect and validate Irish and EU disabled parking permit cards in real time.

---

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR engine
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Configure camera
Edit `config.py` and set `CAMERA_SOURCE`:
```python
CAMERA_SOURCE = "esp32"   # "esp32" | "phone"
```
Update the IP address in `SOURCES` to match your ESP32-CAM's address on the network.

### 4. Run
```bash
python main.py
```
Press `Q` to quit.

---

## Arduino Serial (optional)

To send `VALID`/`INVALID` to an Arduino, update `config.py`:
```python
SERIAL_ENABLED = True
SERIAL_PORT    = "COM3"   # update to your port
```

---

## Project Structure

```
├── main.py                      # Entry point — live camera loop
├── config.py                    # All configuration (camera, thresholds, serial)
├── detection/
│   └── card_detector.py         # Contour detection + perspective warp
├── validation/
│   ├── colour_validator.py      # HSV dominant colour check
│   ├── ocr_validator.py         # Tesseract OCR keyword + expiry check
│   └── layout_validator.py      # Grid-based zone layout check
├── comms/
│   └── arduino_serial.py        # Serial output to Arduino
├── training/                    # Phase 2 — ML training scripts (coming)
├── models/                      # Phase 2 — trained TFLite model (generated)
├── data/
│   ├── valid/                   # Phase 2 — sample valid card images
│   └── invalid/                 # Phase 2 — sample invalid card images
└── requirements.txt
```

---

## Supported Cards

| Card | Type key |
|---|---|
| Irish Disabled Parking Permit (DDAI) | `parking_permit_ie` |
| EU Disabled Parking Card | `parking_permit_eu` |

---

## Phase 1 Deliverables

- [x] Camera feed displays live with card bounding box overlay
- [x] Card is correctly cropped and perspective-corrected
- [x] Colour validation works for Irish and EU parking permits
- [x] OCR extracts and validates keywords and expiry date
- [x] Result displayed on screen as green (VALID) or red (INVALID) overlay
- [x] Serial message sent to Arduino on each detection
- [x] Switching camera source requires only changing `CAMERA_SOURCE` in `config.py`
- [ ] Phase 2 — ML model trained and integrated with >85% accuracy
