import time

BuzzerPin = 4

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BuzzerPin, GPIO.OUT)
    pwm = GPIO.PWM(BuzzerPin, 1000)
    _GPIO_AVAILABLE = True
except ImportError:
    pwm = None
    _GPIO_AVAILABLE = False


def beep(duration=0.3):
    if not _GPIO_AVAILABLE:
        return
    pwm.start(50)
    time.sleep(duration)
    pwm.stop()
