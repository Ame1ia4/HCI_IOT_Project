import time

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.OUT)
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

BuzzerPin = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(BuzzerPin, GPIO.OUT)

#nice tone
pwm = GPIO.PWM(BuzzerPin, 1000)

def beep(duration=0.3):
    pwm.start(50)  # 50% duty cycle
    time.sleep(duration)
    pwm.stop()