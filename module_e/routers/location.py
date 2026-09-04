"""`POST /location/estimate` (cell_id/toa/wifi/ip) + `GET /location/poi` (`docs/SUJET.md` MOD-E).

`cell_id`/`toa`/`wifi` all simulate a positioning technique against the same real-BTS `Terrain`
Module F already uses (`module_c.fitness._get_default_terrain`, reused directly — same precedent
as `module_f/pb2_bts.py`). `ip` is the odd one out: it geolocates a real (or the caller's own)
public IP via `module_c.ipinfo_client`, no terrain/true-position involved at all.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from module_c import cell_id as cell_id_module
from module_c import toa as toa_module
from module_c import wifi_fp as wifi_fp_module
from module_c.fitness import _get_default_terrain
from module_c.ipinfo_client import locate_ip
from module_c.lbs_poi import nearby_pois
from module_c.terrain_sim import Terrain, lonlat_to_xy, xy_to_lonlat
from module_e.auth import get_current_user
from module_e.models import LocationEstimateRequest, LocationEstimateResponse, PoiOut
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(tags=["location"], dependencies=[Depends(get_current_user)])

_cell_id_centroids_cache: dict[int, np.ndarray] = {}
_wifi_grid_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _cell_id_centroids(terrain: Terrain) -> np.ndarray:
    key = id(terrain)
    if key not in _cell_id_centroids_cache:
        _cell_id_centroids_cache[key] = cell_id_module.voronoi_cell_centroids(terrain)
    return _cell_id_centroids_cache[key]


def _wifi_grid(terrain: Terrain) -> tuple[np.ndarray, np.ndarray]:
    key = id(terrain)
    if key not in _wifi_grid_cache:
        _wifi_grid_cache[key] = wifi_fp_module.build_fingerprint_grid(terrain)
    return _wifi_grid_cache[key]


@router.post("/location/estimate", response_model=LocationEstimateResponse)
@limiter.limit(RATE_LIMIT)
def estimate(request: Request, payload: LocationEstimateRequest) -> LocationEstimateResponse:
    if payload.method == "ip":
        location = locate_ip(payload.ip)
        return LocationEstimateResponse(method="ip", lat=location.lat, lon=location.lon, ip=location.ip, city=location.city, region=location.region, country=location.country)

    if payload.lat is None or payload.lon is None:
        raise HTTPException(422, detail=f"method={payload.method!r} requires lat/lon (the true position to simulate positioning from)")

    terrain = _get_default_terrain()
    x, y = lonlat_to_xy(payload.lon, payload.lat, terrain.lon0, terrain.lat0)
    true_xy = np.array([float(x), float(y)])
    rng = np.random.default_rng(payload.seed)

    if payload.method == "cell_id":
        estimated_xy = cell_id_module.estimate_position(true_xy, terrain, centroids=_cell_id_centroids(terrain))
    elif payload.method == "toa":
        estimated_xy = toa_module.estimate_position(true_xy, terrain, k=payload.k or toa_module.DEFAULT_K_ANCHORS, rng=rng)
    else:  # wifi
        centroids, fingerprints = _wifi_grid(terrain)
        model = wifi_fp_module.fit_knn(fingerprints)
        noise_std_db = payload.noise_std_db if payload.noise_std_db is not None else wifi_fp_module.DEFAULT_RSSI_NOISE_STD_DB
        query_fingerprint = wifi_fp_module.simulate_rssi(true_xy, terrain.positions_xy, noise_std_db, rng)
        estimated_xy = wifi_fp_module.estimate_position(query_fingerprint, centroids, model)

    est_lon, est_lat = xy_to_lonlat(estimated_xy[0], estimated_xy[1], terrain.lon0, terrain.lat0)
    error_m = float(np.linalg.norm(estimated_xy - true_xy))
    return LocationEstimateResponse(method=payload.method, lat=float(est_lat), lon=float(est_lon), error_m=error_m)


@router.get("/location/poi", response_model=list[PoiOut])
@limiter.limit(RATE_LIMIT)
def poi(request: Request, lat: float, lon: float, radius_m: float = 500.0, limit: int = 10) -> list[PoiOut]:
    pois = nearby_pois(lat, lon, radius_m=radius_m, limit=limit)
    return [PoiOut(name=p.name, category=p.category, lat=p.lat, lon=p.lon, distance_m=p.distance_m) for p in pois]
