import datetime

import requests

import config


def post_result(results: dict):
    """
    POST a card scan result to the configured IoT endpoint as JSON.
    Silently does nothing if ENDPOINT_ENABLED is False.

    Payload example:
    {
        "timestamp":   "2024-11-01T14:32:05",
        "is_valid":    true,
        "card_type":   "parking_permit_ie",
        "score":       0.83,
        "colour_conf": 0.91,
        "text_conf":   0.67,
        "layout_conf": 0.80,
        "ml_conf":     0.0,
        "expired":     false
    }
    """
    if not config.ENDPOINT_ENABLED:
        return

    # Session reset signal (card removed without a valid result)
    if results.get("session_reset"):
        try:
            requests.post(
                config.ENDPOINT_URL,
                json={"session_reset": True},
                timeout=config.ENDPOINT_TIMEOUT,
            )
        except requests.exceptions.RequestException:
            pass
        return

    payload = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "is_valid":       results.get("is_valid", False),
        "card_type":      results.get("card_type"),
        "score":          results.get("score", 0.0),
        "colour_conf":    results.get("colour_conf", 0.0),
        "text_conf":      results.get("text_conf", 0.0),
        "layout_conf":    results.get("layout_conf", 0.0),
        "ml_conf":        results.get("ml_conf", 0.0),
        "student_number": results.get("student_number"),
        "attempts":       results.get("attempts"),
    }

    try:
        requests.post(
            config.ENDPOINT_URL,
            json=payload,
            timeout=config.ENDPOINT_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        print(f"[HTTP] Could not reach endpoint: {e}")
