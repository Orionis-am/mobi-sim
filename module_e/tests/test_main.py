from __future__ import annotations

from fastapi.testclient import TestClient


class TestRoot:
    def test_root_returns_docs_pointer(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["docs"] == "/docs"


class TestSwagger:
    def test_docs_page_renders(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200

    def test_openapi_schema_lists_all_spec_endpoints(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        expected = {
            "/auth/token",
            "/services",
            "/services/{service_id}",
            "/codecs/evaluate",
            "/sms/send",
            "/sms/{sid}/status",
            "/sms/webhook",
            "/location/estimate",
            "/location/poi",
            "/qos/predict",
            "/qos/session/live",
            "/optimize/run",
            "/optimize/{job_id}",
        }
        assert expected <= paths.keys()


class TestRateLimit:
    def test_exceeding_limit_returns_429(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        responses = [client.get("/services", headers=auth_headers) for _ in range(61)]
        assert any(r.status_code == 429 for r in responses)
