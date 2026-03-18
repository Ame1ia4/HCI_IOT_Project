import time

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.OUT)
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

BuzzerPin = 4

def beep(duration=5):
    if not _GPIO_AVAILABLE:
        return
    GPIO.output(BuzzerPin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BuzzerPin, GPIO.LOW)