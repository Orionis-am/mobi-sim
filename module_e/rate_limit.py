"""Shared SlowAPI `Limiter` instance (`docs/SUJET.md` MOD-E: "60 req/min par token").

A tiny standalone module rather than defining `limiter` in `main.py`, so every router can import
it for its own `@limiter.limit(...)` decorators without a circular import on `main`.
"""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from module_e import auth

RATE_LIMIT = "60/minute"


def _rate_limit_key(request: Request) -> str:
    """Per-token limiting once authenticated; falls back to the caller's IP before that."""
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        payload = auth.decode_token(authorization[7:])
        subject = payload.get("sub") if payload else None
        if subject:
            return f"user:{subject}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
