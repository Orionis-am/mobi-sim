"""IP-based geolocation via ipinfo.io (`docs/SUJET.md` line 196-197): approximate position from a
public IP address, free tier (50k req/month, no credit card). City-level granularity at best —
legitimate uses (e.g. content localization, fraud-signal, coarse analytics) and misuses (tracking
someone precisely, which IP geolocation cannot actually do) belong in the final report's
discussion, not hardcoded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

IPINFO_BASE_URL = "https://ipinfo.io"
DEFAULT_TIMEOUT_S = 5.0


@dataclass
class IpLocation:
    ip: str
    city: str | None
    region: str | None
    country: str | None
    lat: float | None
    lon: float | None


def _token() -> str:
    token = os.environ.get("IPINFO_TOKEN")
    if not token:
        raise RuntimeError("IPINFO_TOKEN not set — required to call the ipinfo.io API")
    return token


def _parse_loc(loc: str | None) -> tuple[float | None, float | None]:
    """ipinfo.io returns "lat,lon" as one comma-joined string, or omits the field entirely."""
    if not loc:
        return None, None
    lat_str, lon_str = loc.split(",")
    return float(lat_str), float(lon_str)


def locate_ip(ip: str | None = None, token: str | None = None, session: requests.Session | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> IpLocation:
    """Look up the approximate location of ``ip`` — the caller's own public IP if omitted."""
    token = token or _token()
    session = session if session is not None else requests
    url = f"{IPINFO_BASE_URL}/{ip}/json" if ip else f"{IPINFO_BASE_URL}/json"

    response = session.get(url, params={"token": token}, timeout=timeout_s)
    response.raise_for_status()
    data = response.json()

    lat, lon = _parse_loc(data.get("loc"))
    return IpLocation(ip=data.get("ip", ip or ""), city=data.get("city"), region=data.get("region"), country=data.get("country"), lat=lat, lon=lon)
