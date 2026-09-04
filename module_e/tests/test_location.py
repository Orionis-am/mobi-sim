"""`/location/*` — `cell_id`/`toa`/`wifi` run for real against the local OpenCelliD terrain (no
network, no API key). `ip` (ipinfo.io, real network + token) and `poi` (Nominatim/Overpass, real
network) are mocked, mirroring `module_c/test_module_c.py`'s own mocking boundary.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from module_c.fitness import _get_default_terrain
from module_c.ipinfo_client import IpLocation
from module_c.lbs_poi import Poi
from module_c.terrain_sim import xy_to_lonlat
from module_e.routers import location as location_router


@pytest.fixture()
def known_point() -> tuple[float, float]:
    terrain = _get_default_terrain()
    x, y = terrain.positions_xy[0]
    lon, lat = xy_to_lonlat(x, y, terrain.lon0, terrain.lat0)
    return float(lat), float(lon)


class TestEstimate:
    @pytest.mark.parametrize("method", ["cell_id", "toa", "wifi"])
    def test_estimates_against_real_terrain(self, client: TestClient, auth_headers: dict[str, str], known_point: tuple[float, float], method: str) -> None:
        lat, lon = known_point
        response = client.post("/location/estimate", headers=auth_headers, json={"method": method, "lat": lat, "lon": lon, "seed": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == method
        assert body["lat"] is not None and body["lon"] is not None
        assert body["error_m"] >= 0.0

    def test_missing_lat_lon_is_rejected(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post("/location/estimate", headers=auth_headers, json={"method": "cell_id"})
        assert response.status_code == 422

    def test_ip_method_mocked(self, client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            location_router, "locate_ip", lambda ip: IpLocation(ip="203.0.113.5", city="Toulouse", region="Occitanie", country="FR", lat=43.6, lon=1.44)
        )
        response = client.post("/location/estimate", headers=auth_headers, json={"method": "ip", "ip": "203.0.113.5"})
        assert response.status_code == 200
        body = response.json()
        assert body["city"] == "Toulouse"
        assert body["lat"] == 43.6

    def test_requires_auth(self, client: TestClient, known_point: tuple[float, float]) -> None:
        lat, lon = known_point
        response = client.post("/location/estimate", json={"method": "cell_id", "lat": lat, "lon": lon})
        assert response.status_code == 401


class TestPoi:
    def test_returns_pois(self, client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            location_router, "nearby_pois", lambda lat, lon, radius_m=500.0, limit=10: [Poi(name="Café Test", category="cafe", lat=lat, lon=lon, distance_m=42.0)]
        )
        response = client.get("/location/poi", headers=auth_headers, params={"lat": 44.35, "lon": 2.57})
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["name"] == "Café Test"
