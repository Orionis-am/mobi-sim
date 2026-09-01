"""Manual, side-effecting smoke test against the real Twilio Verify API.

Not part of the pytest suite (test_module_b.py mocks Twilio via
FakeTwilioClient throughout — see its docstring). Uses Verify's own
system-generated OTP body instead of a customer-supplied SMS body, which is
what lets it succeed on a trial account: plain messages.create() with a
free-text body is rejected with error 60409 ("Custom message did not match
any template") on trial accounts, but Verify's body is never customer text
— see REPORT.md for the full story. Run it yourself when ready:

    uv run python -m module_b.manual_twilio_verify_check          # sends an OTP
    uv run python -m module_b.manual_twilio_verify_check <code>   # checks the code you received

Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_VERIFY_SERVICE_SID
(create one with client.verify.v2.services.create(...) — see REPORT.md) and
TWILIO_TEST_TO_NUMBER (the student's own verified destination number) in the
environment or a .env file at the repo root.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from module_b import twilio_client


def main() -> None:
    load_dotenv()
    to = os.environ["TWILIO_TEST_TO_NUMBER"]

    if len(sys.argv) > 1:
        code = sys.argv[1]
        print(f"Checking code for {to}...")
        result = twilio_client.check_verification(to, code)
        print(f"status={result.status}")
        print(f"valid={result.valid}")
        return

    print(f"Sending real OTP via Verify to {to}...")
    result = twilio_client.start_verification(to)
    print(f"sid={result.sid}")
    print(f"status={result.status}")
    print(f"channel={result.channel}")
    print("Once you receive the code, run:")
    print("    uv run python -m module_b.manual_twilio_verify_check <code>")


if __name__ == "__main__":
    main()
