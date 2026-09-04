"""`/sms/*` — mocks every `module_b.twilio_client` call (real Twilio API, real trial send)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from module_b.twilio_client import SmsSendResult, SmsStatus
from module_e.routers import sms as sms_router


class TestSend:
    def test_sends_and_returns_result(self, client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sms_router,
            "send_sms",
            lambda to, body: SmsSendResult(sid="SM123", status="queued", to=to, from_="+15550000000", date_created=datetime.now(timezone.utc)),
        )
        response = client.post("/sms/send", headers=auth_headers, json={"to": "+15551234567", "body": "hello"})
        assert response.status_code == 200
        body = response.json()
        assert body["sid"] == "SM123"
        assert body["to"] == "+15551234567"

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/sms/send", json={"to": "+15551234567", "body": "hello"})
        assert response.status_code == 401


class TestStatus:
    def test_returns_status(self, client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sms_router, "get_delivery_status", lambda sid: SmsStatus(sid=sid, status="delivered", error_code=None, date_updated=None))
        response = client.get("/sms/SM123/status", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == {"sid": "SM123", "status": "delivered", "error_code": None, "date_updated": None}


class TestWebhook:
    def test_parses_form_payload_and_returns_204(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        received = {}
        monkeypatch.setattr(sms_router, "handle_status_webhook", lambda payload: received.update(payload))

        response = client.post("/sms/webhook", data={"MessageSid": "SM123", "MessageStatus": "delivered"})
        assert response.status_code == 204
        assert received == {"MessageSid": "SM123", "MessageStatus": "delivered"}

    def test_no_auth_required(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sms_router, "handle_status_webhook", lambda payload: None)
        response = client.post("/sms/webhook", data={"MessageSid": "SM1", "MessageStatus": "sent"})
        assert response.status_code == 204
