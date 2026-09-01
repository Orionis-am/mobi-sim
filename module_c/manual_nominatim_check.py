"""Manual, side-effecting smoke test against the real Overpass (OpenStreetMap) API.

Not part of the pytest suite (test_module_c.py mocks the Overpass API via FakeOverpassSession
throughout — see its docstring) — running this script makes one real HTTP call. Not run
automatically for that reason; run it yourself when ready:

    uv run python -m module_c.manual_nominatim_check

No API key needed (unlike ipinfo.io) — Overpass/OpenStreetMap data is free and unauthenticated,
just a mandatory User-Agent header and a 1 req/s rate limit (docs/SUJET.md line 202), both already
handled by lbs_poi.py.
"""

from __future__ import annotations

from module_c import lbs_poi

# Rodez city center (Aveyron), the module's target region — a real, densely-tagged OSM point.
RODEZ_LAT = 44.3506
RODEZ_LON = 2.5731


def main() -> None:
    print(f"Looking up real POIs near Rodez ({RODEZ_LAT}, {RODEZ_LON}) via Overpass...")
    pois = lbs_poi.nearby_pois(RODEZ_LAT, RODEZ_LON, radius_m=500.0, limit=10)

    if not pois:
        print("No POIs returned within 500m.")
        return

    for poi in pois:
        print(f"{poi.distance_m:6.0f}m  {poi.category:15s} {poi.name}")


if __name__ == "__main__":
    main()
