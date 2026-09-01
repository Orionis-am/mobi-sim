"""Folium HTML maps for Module C (`docs/SUJET.md` lines 193, 203-204).

Mirrors `module_a/visualize.py`'s discipline: builder functions return a `folium.Map` already
populated from data computed elsewhere (cell_id.py, toa.py, lbs_poi.py) — no I/O, no
recomputation. `save_map_html` is the one explicit place a file gets written, into the gitignored
`results/` directory the rest of the project uses for generated output. POIs are accepted
duck-typed (``.lat``/``.lon``/``.name``/``.category``/``.distance_m``) rather than importing
`lbs_poi.Poi`, so this file has no hard dependency on where the data came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import folium

DEFAULT_ZOOM_START = 15


def build_position_map(
    true_lat: float | None,
    true_lon: float | None,
    estimated_lat: float,
    estimated_lon: float,
    uncertainty_radius_m: float | None = None,
    pois: Sequence | None = None,
    zoom_start: int = DEFAULT_ZOOM_START,
) -> folium.Map:
    """Base map: estimated-position marker, optional true-position marker, uncertainty circle, POI markers."""
    m = folium.Map(location=[estimated_lat, estimated_lon], zoom_start=zoom_start)

    folium.Marker(
        location=[estimated_lat, estimated_lon],
        popup="Position estimee",
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    ).add_to(m)

    if true_lat is not None and true_lon is not None:
        folium.Marker(
            location=[true_lat, true_lon],
            popup="Position reelle",
            icon=folium.Icon(color="green", icon="check", prefix="fa"),
        ).add_to(m)

    if uncertainty_radius_m is not None:
        folium.Circle(
            location=[estimated_lat, estimated_lon],
            radius=uncertainty_radius_m,
            color="red",
            fill=True,
            fill_opacity=0.1,
            popup=f"Incertitude ~{uncertainty_radius_m:.0f} m",
        ).add_to(m)

    for poi in pois or []:
        folium.Marker(
            location=[poi.lat, poi.lon],
            popup=f"{poi.category}: {poi.name} ({poi.distance_m:.0f} m)",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

    return m


def add_toa_range_circles(m: folium.Map, anchors_latlon: Sequence[tuple[float, float]], radii_m: Sequence[float]) -> folium.Map:
    """Overlay one TOA range circle per anchor BTS on an existing map (`docs/SUJET.md` line 193)."""
    for (lat, lon), radius in zip(anchors_latlon, radii_m):
        folium.Circle(location=[lat, lon], radius=radius, color="orange", fill=False, weight=1).add_to(m)
    return m


def save_map_html(m: folium.Map, filename: str, output_dir: str | Path = "results") -> Path:
    """Save ``m`` as a standalone HTML file under ``output_dir`` (created if missing); returns the file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    m.save(str(path))
    return path
