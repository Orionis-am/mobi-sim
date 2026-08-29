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

from dotenv import load_dotenv

from module_b import compare


def main() -> None:
    load_dotenv()
    to = os.environ["TWILIO_TEST_TO_NUMBER"]

    print(f"Sending real SMS to {to} and polling for delivery status...")
    sample = compare.measure_real_delivery(to, "MobiSim Module B smoke test — real Twilio trial send.")

    print(f"accept_latency_s={sample.accept_latency_s:.2f}")
    print(f"delivery_latency_s={sample.delivery_latency_s}")
    print(f"final_status={sample.final_status}")


if __name__ == "__main__":
    main()
