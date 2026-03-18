import RPi.GPIO as GPIO
import time

BuzzerPin = 4

GPIO.setmode(GPIO.BCM)
GPIO.setup(BuzzerPin, GPIO.OUT)

def beep(duration=5):
    GPIO.output(BuzzerPin, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BuzzerPin, GPIO.LOW)