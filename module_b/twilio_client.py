"""Real SMS send/status via the Twilio trial API, plus a webhook-payload parser.

Framework-agnostic on purpose: this module never imports FastAPI/Flask.
``handle_status_webhook`` takes and returns plain data — Module E (not built
yet) is expected to wire it into a future ``POST /twilio/webhook`` route,
the same way Module F consumes Module A's ``codec_fitness`` without either
side knowing about the other's framework.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client


def _client() -> Client:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        raise RuntimeError("TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set — required to call the Twilio API")
    return Client(account_sid, auth_token)


@dataclass
class SmsSendResult:
    sid: str
    status: str
    to: str
    from_: str
    date_created: datetime | None


@dataclass
class SmsStatus:
    sid: str
    status: str
    error_code: int | None
    date_updated: datetime | None


def send_sms(to: str, body: str, from_: str | None = None, client: Client | None = None) -> SmsSendResult:
    """Send one real SMS via the Twilio trial API. Costs nothing (trial credit) but is a real send."""
    sender = from_ or os.environ.get("TWILIO_PHONE_NUMBER")
    if not sender:
        raise RuntimeError("from_ not given and TWILIO_PHONE_NUMBER not set")
    client = client or _client()
    message = client.messages.create(to=to, from_=sender, body=body)
    return SmsSendResult(sid=message.sid, status=message.status, to=message.to, from_=message.from_, date_created=message.date_created)


def get_delivery_status(sid: str, client: Client | None = None) -> SmsStatus:
    """Poll Twilio for a message's current delivery status — the spec's "ou polling" alternative to the webhook.

    Falls back to listing recent messages and matching by sid when the
    single-resource fetch is forbidden — observed in practice on a trial
    account (``GET /Messages/{Sid}`` 403s even though the same message is
    fully visible via ``GET /Messages``; see REPORT.md).
    """
    client = client or _client()
    try:
        message = client.messages(sid).fetch()
    except TwilioRestException as exc:
        if exc.status != 403:
            raise
        message = next((m for m in client.messages.list(limit=50) if m.sid == sid), None)
        if message is None:
            raise
    return SmsStatus(sid=message.sid, status=message.status, error_code=message.error_code, date_updated=message.date_updated)


def _verify_service_sid() -> str:
    service_sid = os.environ.get("TWILIO_VERIFY_SERVICE_SID")
    if not service_sid:
        raise RuntimeError("TWILIO_VERIFY_SERVICE_SID not set — required to call the Twilio Verify API")
    return service_sid


@dataclass
class VerificationResult:
    sid: str
    status: str
    to: str
    channel: str


@dataclass
class VerificationCheckResult:
    status: str
    valid: bool


def start_verification(to: str, channel: str = "sms", client: Client | None = None) -> VerificationResult:
    """Send a real OTP via the Twilio Verify API.

    Trial accounts reject free-text ``messages.create(body=...)`` content
    (error 60409, "Custom message did not match any template" — see
    REPORT.md); Verify sidesteps the whole restriction because its SMS body
    is Twilio's own fixed OTP template, never customer-supplied text.
    """
    client = client or _client()
    verification = client.verify.v2.services(_verify_service_sid()).verifications.create(to=to, channel=channel)
    return VerificationResult(sid=verification.sid, status=verification.status, to=verification.to, channel=verification.channel)


def check_verification(to: str, code: str, client: Client | None = None) -> VerificationCheckResult:
    """Check an OTP code the recipient received via ``start_verification`` against the Verify API."""
    client = client or _client()
    check = client.verify.v2.services(_verify_service_sid()).verification_checks.create(to=to, code=code)
    return VerificationCheckResult(status=check.status, valid=check.valid)


def handle_status_webhook(payload: dict) -> SmsStatus:
    """Normalize a Twilio status-callback POST body (form fields as a dict) into an ``SmsStatus``.

    Pure function: no request parsing, no framework dependency. The webhook
    payload carries no pre-parsed timestamp, unlike the polling path, so
    ``date_updated`` is always ``None`` here.
    """
    error_code = payload.get("ErrorCode")
    return SmsStatus(
        sid=payload["MessageSid"],
        status=payload["MessageStatus"],
        error_code=int(error_code) if error_code else None,
        date_updated=None,
    )
