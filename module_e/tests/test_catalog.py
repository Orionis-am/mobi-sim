from __future__ import annotations

from fastapi.testclient import TestClient


class TestListServices:
    def test_returns_all_seeded_services(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services", headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 20
        assert {"id", "name", "category", "protocol", "codec", "min_mos", "min_bitrate_kbps", "max_delay_ms", "tariff"} <= body[0].keys()

    def test_filters_by_category(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services", headers=auth_headers, params={"category": "bearer"})
        assert response.status_code == 200
        body = response.json()
        assert len(body) > 0
        assert all(s["category"] == "bearer" for s in body)

    def test_filters_by_codec(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services", headers=auth_headers, params={"codec": "opus"})
        assert response.status_code == 200
        assert all(s["codec"] == "opus" for s in response.json())

    def test_filters_by_min_mos(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services", headers=auth_headers, params={"min_mos": 3.5})
        assert response.status_code == 200
        assert all(s["min_mos"] >= 3.5 for s in response.json())


class TestGetService:
    def test_returns_matching_service(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services/1", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == 1

    def test_404_for_unknown_id(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/services/9999", headers=auth_headers)
        assert response.status_code == 404
