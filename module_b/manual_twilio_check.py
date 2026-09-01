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

The body below must be one of Twilio's ~10 predefined trial template names
(e.g. "sms_appointment_reminders") rather than free text — trial accounts
reject arbitrary content (error 60409) but accept a template name as the
body and substitute the real approved wording on delivery. See REPORT.md.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from module_b import compare

_TRIAL_TEMPLATE_BODY = "sms_appointment_reminders"


def main() -> None:
    load_dotenv()
    to = os.environ["TWILIO_TEST_TO_NUMBER"]

    print(f"Sending real SMS to {to} and polling for delivery status...")
    sample = compare.measure_real_delivery(to, _TRIAL_TEMPLATE_BODY)

    print(f"accept_latency_s={sample.accept_latency_s:.2f}")
    print(f"delivery_latency_s={sample.delivery_latency_s}")
    print(f"final_status={sample.final_status}")


if __name__ == "__main__":
    main()
