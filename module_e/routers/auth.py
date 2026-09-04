"""`POST /auth/token` (spec) + `POST /auth/register` (added — see REPORT.md for rationale)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from module_e import auth
from module_e.database import User, get_db
from module_e.models import Token, UserOut, UserRegister
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit(RATE_LIMIT)
def register(request: Request, payload: UserRegister, db: Session = Depends(get_db)) -> User:
    if db.query(User).filter(User.username == payload.username).first() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Username already registered")

    user = User(username=payload.username, hashed_password=auth.hash_password(payload.password), scope=payload.scope)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/token", response_model=Token)
@limiter.limit(RATE_LIMIT)
def issue_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"}
        )
    return Token(access_token=auth.create_access_token(user.username, user.scope), refresh_token=auth.create_refresh_token(user.username, user.scope))
