"""Manual, side-effecting smoke test against the real Twilio trial API.

Not part of the pytest suite (test_module_b.py mocks Twilio via
FakeTwilioClient throughout — see its docstring) — running this script
sends one real SMS. Not run automatically for that reason; run it yourself
when ready:

    uv run python -m module_b.manual_twilio_check

Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER (the
Twilio trial sender number) and TWILIO_TEST_TO_NUMBER (the student's own
verified destination number — trial accounts can only send to verified
numbers) in the environment or a .env file at the repo root.
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from module_b import twilio_client

TERMINAL_STATUSES = {"delivered", "failed", "undelivered"}


def main() -> None:
    load_dotenv()
    to = os.environ["TWILIO_TEST_TO_NUMBER"]

    print(f"Sending real SMS to {to}...")
    result = twilio_client.send_sms(to, "MobiSim Module B smoke test — real Twilio trial send.")
    print(f"Sent: sid={result.sid} status={result.status}")

    print("Polling for delivery status...")
    for _ in range(10):
        time.sleep(2)
        status = twilio_client.get_delivery_status(result.sid)
        print(f"  status={status.status} error_code={status.error_code}")
        if status.status in TERMINAL_STATUSES:
            break


if __name__ == "__main__":
    main()
