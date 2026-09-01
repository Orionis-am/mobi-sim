"""Local planar terrain built from real OpenCelliD BTS coordinates.

Everything downstream (Voronoi, TOA trilateration, Wi-Fi fingerprinting grids) works far more
naturally in Euclidean meters than in (lon, lat) degrees — distances, grids and `scipy.optimize`
all assume a flat metric space. The study region (Aveyron, ~100 km across) is small enough that a
full geodesic projection (e.g. UTM via `pyproj`) is unnecessary precision `docs/SUJET.md` §6.1
doesn't ask for; an equirectangular approximation centered on the region's own centroid keeps
error well under 0.1% at this scale with no new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def lonlat_to_xy(lon: float | np.ndarray, lat: float | np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection onto a local (x, y) meter frame centered at (lon0, lat0)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS_M
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def xy_to_lonlat(x: float | np.ndarray, y: float | np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``lonlat_to_xy`` — needed since Folium (map_viz.py) renders in (lon, lat)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    lat = lat0 + np.degrees(y / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(x / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lon, lat


@dataclass(frozen=True)
class Terrain:
    bts: pd.DataFrame  # original columns + local "x"/"y" (meters)
    lon0: float
    lat0: float

    @property
    def positions_xy(self) -> np.ndarray:
        """(N, 2) array of real BTS positions in local meters — the shape scipy/sklearn expect."""
        return self.bts[["x", "y"]].to_numpy()

    @property
    def bbox_xy(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the real BTS positions, in local meters."""
        x, y = self.bts["x"], self.bts["y"]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def build_terrain(bts_df: pd.DataFrame) -> Terrain:
    """Build a local planar Terrain from a (lon, lat)-indexed BTS DataFrame (e.g. from ``sample_bts``).

    The projection is centered on the sampled BTS's own centroid, not a fixed global reference
    point — keeps the local frame's origin near (0, 0) regardless of which region was sampled.
    """
    lon0 = float(bts_df["lon"].mean())
    lat0 = float(bts_df["lat"].mean())
    x, y = lonlat_to_xy(bts_df["lon"].to_numpy(), bts_df["lat"].to_numpy(), lon0, lat0)
    bts = bts_df.copy()
    bts["x"] = x
    bts["y"] = y
    return Terrain(bts=bts, lon0=lon0, lat0=lat0)
