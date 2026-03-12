import serial

import config


def send_result(is_valid: bool):
    """
    Send VALID or INVALID over serial to the Arduino.
    Silently does nothing if SERIAL_ENABLED is False in config.

    Args:
      is_valid: True sends b'VALID\\n', False sends b'INVALID\\n'.
    """
    if not config.SERIAL_ENABLED:
        return

    msg = b"VALID\n" if is_valid else b"INVALID\n"
    try:
        with serial.Serial(config.SERIAL_PORT, config.SERIAL_BAUD, timeout=1) as ser:
            ser.write(msg)
    except serial.SerialException as e:
        print(f"[Serial] Could not send to {config.SERIAL_PORT}: {e}")
