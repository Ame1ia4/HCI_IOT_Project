import RPi.GPIO as GPIO
import time

BuzzerPin = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(BuzzerPin, GPIO.OUT)

#nice tone
pwm = GPIO.PWM(BuzzerPin, 1000)

def beep(duration=0.3):
    pwm.start(50)  # 50% duty cycle
    time.sleep(duration)
    pwm.stop()