"""Real POIs near an estimated position, via the Overpass API (`docs/SUJET.md` line 200-202).

Nominatim's free-text search isn't built for "N nearest POIs of any category around a point";
Overpass QL's `around:radius,lat,lon` filter is the natural fit for that query shape, and the spec
names "Nominatim/Overpass" together as the acceptable real-data source — both are the same
OpenStreetMap data, free, no registration. Mandatory `User-Agent` header and a rate-limit note per
line 202: a single call per invocation here has nothing to throttle against; a caller looping over
many positions is responsible for spacing calls at least 1s apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "MobiSim/0.1 (student LBS demo project, docs/SUJET.md)"
DEFAULT_RADIUS_M = 500.0
DEFAULT_LIMIT = 10
DEFAULT_TIMEOUT_S = 25.0
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class Poi:
    name: str
    category: str
    lat: float
    lon: float
    distance_m: float


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)))


def _build_query(lat: float, lon: float, radius_m: float, fetch_limit: int) -> str:
    return f"[out:json][timeout:25];\nnode(around:{radius_m},{lat},{lon})[amenity];\nout body {fetch_limit};"


def nearby_pois(
    lat: float, lon: float, radius_m: float = DEFAULT_RADIUS_M, limit: int = DEFAULT_LIMIT, session: requests.Session | None = None, timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[Poi]:
    """The ``limit`` real OSM POIs (amenity-tagged nodes) nearest to (lat, lon), within ``radius_m``.

    Over-fetches (3x ``limit``) since Overpass's own result order isn't guaranteed distance-sorted,
    then sorts and trims client-side using the real haversine distance to each result.
    """
    session = session if session is not None else requests
    query = _build_query(lat, lon, radius_m, limit * 3)

    response = session.post(OVERPASS_URL, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout_s)
    response.raise_for_status()
    elements = response.json().get("elements", [])

    pois = []
    for element in elements:
        el_lat, el_lon = element.get("lat"), element.get("lon")
        if el_lat is None or el_lon is None:
            continue
        tags = element.get("tags", {})
        distance = _haversine_distance_m(lat, lon, el_lat, el_lon)
        pois.append(Poi(name=tags.get("name", "?"), category=tags.get("amenity", "?"), lat=el_lat, lon=el_lon, distance_m=distance))

    pois.sort(key=lambda p: p.distance_m)
    return pois[:limit]
