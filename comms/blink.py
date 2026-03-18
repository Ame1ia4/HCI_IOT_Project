import time

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

LED_PIN = 18

def green_on(duration=20):
    if not _GPIO_AVAILABLE:
        return
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(LED_PIN, GPIO.LOW)