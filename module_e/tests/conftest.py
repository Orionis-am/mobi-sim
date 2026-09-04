"""Shared fixtures for module_e's tests: an isolated in-memory DB per test, plus auth helpers.

`database.engine`/`database.SessionLocal` are monkeypatched to a fresh `sqlite:///:memory:`
engine *before* the app's lifespan runs (`TestClient(app)` triggers it on `__enter__`), so
`init_db()`'s seeding and every request's `get_db()` session land on the same isolated database —
and so does `routers.optimize.run_job`'s background-task session, since it looks up
`database.SessionLocal` dynamically rather than importing it by value (see its docstring).
`StaticPool` keeps that one connection alive and shared across threads (the event loop vs. a
request's threadpool worker), which a bare `sqlite:///:memory:` engine would not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from module_e import database
from module_e.main import app

TEST_PASSWORD = "a-secret-password"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine))


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Independent of any real `.env` — tests must pass whether or not a developer has one locally."""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-not-for-production")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def register_and_login(client: TestClient, username: str = "alice", scope: str = "user", password: str = TEST_PASSWORD) -> dict[str, str]:
    client.post("/auth/register", json={"username": username, "password": password, "scope": scope})
    response = client.post("/auth/token", data={"username": username, "password": password})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def auth_headers(client: TestClient) -> dict[str, str]:
    return register_and_login(client)


@pytest.fixture()
def admin_headers(client: TestClient) -> dict[str, str]:
    return register_and_login(client, username="admin", scope="admin")
