"""
This Raspberry Pi code was developed by newbiely.com
This Raspberry Pi code is made available for public use w>
For comprehensive instructions and wiring diagrams, pleas>
https://newbiely.com/tutorials/raspberry-pi/raspberry-pi->
"""

import RPi.GPIO as GPIO
import time

# Set the GPIO mode and warning

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define the GPIO18 pin number
led_pin = 18

# Setup the GPIO pin as an output
GPIO.setup(led_pin, GPIO.OUT)

# Function to blink the LED
def blink_led():
    GPIO.output(led_pin, GPIO.HIGH)
    time.sleep(1)  # LED on for 1 second
    GPIO.output(led_pin, GPIO.LOW)
    time.sleep(1)  # LED off for 1 second

try:
    # Run the LED blinking forever until a keyboard interrupt (Ctrl + C)
    while True:
        blink_led()

except KeyboardInterrupt:
    # Cleanup GPIO settings on keyboard interrupt
    GPIO.cleanup()

