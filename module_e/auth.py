"""JWT OAuth2 password flow (`docs/SUJET.md` MOD-E `auth.py`).

Access tokens expire after 30 minutes (spec-mandated); refresh tokens (mentioned by the spec
without further detail) reuse the same encode/decode machinery with a longer expiry and a
``type: "refresh"`` claim. Scopes are ``admin``/``operator``/``user``, matching `database.py`'s
`User.scope`. `JWT_SECRET_KEY`/`JWT_ALGORITHM` are read lazily from the environment (like
`module_b.twilio_client._client()`'s lazy `os.environ.get`) rather than at import time, so tests
can monkeypatch them without needing a `.env` file.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from module_e.database import User, get_db

ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # one week — not spec-mandated, a reasonable default

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def _secret_key() -> str:
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError("JWT_SECRET_KEY not set — required to sign/verify JWTs")
    return key


def _algorithm() -> str:
    return os.environ.get("JWT_ALGORITHM", "HS256")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _pwd_context.verify(plain_password, hashed_password)


def _create_token(subject: str, scope: str, expires_delta: timedelta, token_type: str) -> str:
    expire = datetime.now(timezone.utc) + expires_delta
    payload = {"sub": subject, "scope": scope, "type": token_type, "exp": expire}
    return jwt.encode(payload, _secret_key(), algorithm=_algorithm())


def create_access_token(username: str, scope: str) -> str:
    return _create_token(username, scope, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES), "access")


def create_refresh_token(username: str, scope: str) -> str:
    return _create_token(username, scope, timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES), "refresh")


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if user is None or not verify_password(password, user.hashed_password):
        return None
    return user


_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, _secret_key(), algorithms=[_algorithm()])
    except JWTError as exc:
        raise _CREDENTIALS_ERROR from exc

    if payload.get("type") != "access":
        raise _CREDENTIALS_ERROR
    username = payload.get("sub")
    if username is None:
        raise _CREDENTIALS_ERROR

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise _CREDENTIALS_ERROR
    return user


def require_scope(*allowed_scopes: str) -> Callable[[User], User]:
    """Dependency factory: 403s unless `get_current_user`'s scope is one of `allowed_scopes`."""

    def _checker(user: User = Depends(get_current_user)) -> User:
        if user.scope not in allowed_scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=f"Requires scope in {allowed_scopes}, got '{user.scope}'")
        return user

    return _checker
