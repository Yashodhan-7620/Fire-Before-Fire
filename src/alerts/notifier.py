"""
Single place all alerts flow through, so the delivery channel can be
swapped (console -> webhook -> email/SMS) without touching the
detectors that call it.
"""
import os
import requests

WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")  # e.g. a Slack/Discord incoming webhook


def send_alert(message: str):
    print(f"[ALERT] {message}")
    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json={"text": message}, timeout=5)
        except requests.RequestException as e:
            print(f"[ALERT] webhook delivery failed: {e}")
