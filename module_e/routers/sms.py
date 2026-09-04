"""`POST /sms/send`, `GET /sms/{sid}/status`, `POST /twilio/webhook` (`docs/SUJET.md` MOD-E).

`module_b.twilio_client` is deliberately framework-agnostic (see its own docstring/REPORT.md) —
this router is the FastAPI-aware caller it was written for. No local DB table: Twilio itself is
the source of truth for a message's status, both via polling (`get_delivery_status`) and via the
webhook it POSTs back to `/twilio/webhook`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from module_b.twilio_client import get_delivery_status, handle_status_webhook, send_sms
from module_e.auth import get_current_user
from module_e.models import SmsSendRequest, SmsSendResponse, SmsStatusResponse
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(prefix="/sms", tags=["sms"])


@router.post("/send", response_model=SmsSendResponse, dependencies=[Depends(get_current_user)])
@limiter.limit(RATE_LIMIT)
def send(request: Request, payload: SmsSendRequest) -> SmsSendResponse:
    result = send_sms(payload.to, payload.body)
    return SmsSendResponse(sid=result.sid, status=result.status, to=result.to, from_=result.from_)


@router.get("/{sid}/status", response_model=SmsStatusResponse, dependencies=[Depends(get_current_user)])
@limiter.limit(RATE_LIMIT)
def status_lookup(request: Request, sid: str) -> SmsStatusResponse:
    result = get_delivery_status(sid)
    return SmsStatusResponse(sid=result.sid, status=result.status, error_code=result.error_code, date_updated=result.date_updated)


@router.post("/webhook", status_code=status.HTTP_204_NO_CONTENT, include_in_schema=True)
async def twilio_webhook(request: Request) -> None:
    """No auth (Twilio itself is the caller) and no rate limit — a real account can burst callbacks."""
    form = await request.form()
    handle_status_webhook(dict(form))
