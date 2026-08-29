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
    """Poll Twilio for a message's current delivery status — the spec's "ou polling" alternative to the webhook."""
    client = client or _client()
    message = client.messages(sid).fetch()
    return SmsStatus(sid=message.sid, status=message.status, error_code=message.error_code, date_updated=message.date_updated)


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
