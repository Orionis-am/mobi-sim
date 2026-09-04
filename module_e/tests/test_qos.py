from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from module_e.routers import qos as qos_router


class _FakeRequest:
    """Reports disconnected after `disconnect_after` polls — drives `_live_qos_events` to a stop
    without depending on TestClient/ASGI disconnect signaling, which is what hung real end-to-end
    streaming through `TestClient` (see REPORT.md)."""

    def __init__(self, disconnect_after: int) -> None:
        self._polls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self._disconnect_after


class TestPredict:
    def test_returns_session_qos(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        payload = {"codec": "opus", "bandwidth_kbps": 64.0, "required_bandwidth_kbps": 64.0, "rtt_ms": 40.0, "jitter_ms": 5.0, "seed": 1}
        response = client.post("/qos/predict", headers=auth_headers, json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["codec"] == "opus"
        assert 1.0 <= body["mos"] <= 4.5

    def test_reproducible_with_same_seed(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        payload = {"codec": "gsm", "bandwidth_kbps": 20.0, "required_bandwidth_kbps": 32.0, "rtt_ms": 60.0, "jitter_ms": 10.0, "seed": 7}
        first = client.post("/qos/predict", headers=auth_headers, json=payload).json()
        second = client.post("/qos/predict", headers=auth_headers, json=payload).json()
        assert first == second

    def test_congested_link_scores_worse(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        base = {"codec": "opus", "rtt_ms": 40.0, "jitter_ms": 5.0, "seed": 1}
        healthy = client.post("/qos/predict", headers=auth_headers, json={**base, "bandwidth_kbps": 64.0, "required_bandwidth_kbps": 64.0}).json()
        starved = client.post("/qos/predict", headers=auth_headers, json={**base, "bandwidth_kbps": 5.0, "required_bandwidth_kbps": 64.0}).json()
        assert starved["mos"] < healthy["mos"]

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/qos/predict", json={"codec": "opus", "bandwidth_kbps": 64.0, "required_bandwidth_kbps": 64.0, "rtt_ms": 40.0, "jitter_ms": 5.0})
        assert response.status_code == 401


class TestSessionLive:
    def test_requires_auth(self, client: TestClient) -> None:
        """Fails fast on the auth dependency, before any streaming starts — deliberately not
        testing a real end-to-end stream through `TestClient` here (see REPORT.md: that hung)."""
        response = client.get("/qos/session/live")
        assert response.status_code == 401


class TestLiveQosEventsGenerator:
    def test_yields_one_sse_event_per_tick_until_disconnected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(qos_router, "LIVE_TICK_SECONDS", 0.0)
        request = _FakeRequest(disconnect_after=2)

        async def collect() -> list[str]:
            return [chunk async for chunk in qos_router._live_qos_events(request, "opus", 64.0, 64.0, 40.0, 5.0, 1)]

        events = asyncio.run(collect())
        assert len(events) == 2
        for chunk in events:
            assert chunk.startswith("data: ") and chunk.endswith("\n\n")
            event = json.loads(chunk[len("data: "):].strip())
            assert event["codec"] == "opus"
            assert "mos" in event and "r_factor" in event
