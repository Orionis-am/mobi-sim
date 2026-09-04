from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from module_e import auth
from module_e.database import User
from module_e.tests.conftest import TEST_PASSWORD, register_and_login


class TestRegister:
    def test_creates_user(self, client: TestClient) -> None:
        response = client.post("/auth/register", json={"username": "carol", "password": TEST_PASSWORD})
        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "carol"
        assert body["scope"] == "user"
        assert "password" not in body

    def test_duplicate_username_conflicts(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "carol", "password": TEST_PASSWORD})
        response = client.post("/auth/register", json={"username": "carol", "password": TEST_PASSWORD})
        assert response.status_code == 409

    def test_short_password_rejected(self, client: TestClient) -> None:
        response = client.post("/auth/register", json={"username": "dave", "password": "short"})
        assert response.status_code == 422


class TestToken:
    def test_issues_access_and_refresh_tokens(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "carol", "password": TEST_PASSWORD})
        response = client.post("/auth/token", data={"username": "carol", "password": TEST_PASSWORD})
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["access_token"] and body["refresh_token"]
        assert body["access_token"] != body["refresh_token"]

    def test_wrong_password_rejected(self, client: TestClient) -> None:
        client.post("/auth/register", json={"username": "carol", "password": TEST_PASSWORD})
        response = client.post("/auth/token", data={"username": "carol", "password": "wrong-password"})
        assert response.status_code == 401

    def test_unknown_user_rejected(self, client: TestClient) -> None:
        response = client.post("/auth/token", data={"username": "ghost", "password": TEST_PASSWORD})
        assert response.status_code == 401


class TestProtectedRoute:
    def test_requires_bearer_token(self, client: TestClient) -> None:
        response = client.get("/services")
        assert response.status_code == 401

    def test_rejects_garbage_token(self, client: TestClient) -> None:
        response = client.get("/services", headers={"Authorization": "Bearer not-a-real-token"})
        assert response.status_code == 401

    def test_valid_token_is_accepted(self, client: TestClient) -> None:
        headers = register_and_login(client)
        response = client.get("/services", headers=headers)
        assert response.status_code == 200


class TestScopeGating:
    def test_register_endpoint_has_no_scope_requirement(self, client: TestClient) -> None:
        response = client.post("/auth/register", json={"username": "operator1", "password": TEST_PASSWORD, "scope": "operator"})
        assert response.status_code == 201
        assert response.json()["scope"] == "operator"

    def test_require_scope_allows_matching_scope(self) -> None:
        checker = auth.require_scope("admin", "operator")
        user = User(id=1, username="alice", hashed_password="x", scope="operator")
        assert checker(user) is user

    def test_require_scope_rejects_other_scope(self) -> None:
        checker = auth.require_scope("admin")
        user = User(id=1, username="alice", hashed_password="x", scope="user")
        with pytest.raises(HTTPException) as exc_info:
            checker(user)
        assert exc_info.value.status_code == 403


class TestDecodeToken:
    def test_decode_token_roundtrips(self) -> None:
        token = auth.create_access_token("alice", "user")
        payload = auth.decode_token(token)
        assert payload is not None
        assert payload["sub"] == "alice"
        assert payload["type"] == "access"

    def test_decode_token_returns_none_for_garbage(self) -> None:
        assert auth.decode_token("not-a-jwt") is None
