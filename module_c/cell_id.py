"""Cell-ID positioning: nearest-BTS Voronoi cell centroid (`docs/SUJET.md` line 190-191).

``scipy.spatial.Voronoi`` leaves edge cells unbounded (some vertices "at infinity"), which has no
well-defined centroid. The standard trick avoids that entirely: add four dummy points far outside
the terrain's bounding box before building the diagram, so every *real* BTS's region comes out
finite, then clip that region to the terrain's actual bounding box (Sutherland-Hodgman, plain
numpy — no `shapely` needed) before computing its centroid.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from module_c.terrain_sim import Terrain


def _add_bounding_points(points: np.ndarray, bbox: tuple[float, float, float, float], margin_factor: float = 10.0) -> np.ndarray:
    x_min, x_max, y_min, y_max = bbox
    dx, dy = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    far = margin_factor * max(dx, dy, 1.0)
    dummies = np.array([[cx - far, cy - far], [cx - far, cy + far], [cx + far, cy - far], [cx + far, cy + far]])
    return np.vstack([points, dummies])


def _interp(a: np.ndarray, b: np.ndarray, axis: int, value: float) -> np.ndarray:
    t = (value - a[axis]) / (b[axis] - a[axis])
    return a + t * (b - a)


def _clip_polygon_to_bbox(polygon: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Sutherland-Hodgman polygon clip against an axis-aligned box, one edge of the box at a time."""
    x_min, x_max, y_min, y_max = bbox
    edges = [(0, x_min, True), (0, x_max, False), (1, y_min, True), (1, y_max, False)]

    for axis, value, keep_greater in edges:
        if len(polygon) == 0:
            return polygon

        def inside(p: np.ndarray) -> bool:
            return p[axis] >= value if keep_greater else p[axis] <= value

        output = []
        n = len(polygon)
        for i in range(n):
            curr, prev = polygon[i], polygon[i - 1]
            curr_in, prev_in = inside(curr), inside(prev)
            if curr_in:
                if not prev_in:
                    output.append(_interp(prev, curr, axis, value))
                output.append(curr)
            elif prev_in:
                output.append(_interp(prev, curr, axis, value))
        polygon = np.array(output) if output else np.empty((0, 2))

    return polygon


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    """Area-weighted polygon centroid (shoelace formula); falls back to the vertex mean if degenerate."""
    if len(polygon) < 3:
        return polygon.mean(axis=0) if len(polygon) else np.array([np.nan, np.nan])
    x, y = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return polygon.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6 * area)
    cy = ((y + y1) * cross).sum() / (6 * area)
    return np.array([cx, cy])


def voronoi_cell_centroids(terrain: Terrain) -> np.ndarray:
    """One bbox-clipped Voronoi centroid per real BTS, in the same order as ``terrain.positions_xy``.

    Computed once (not per query) since it only depends on the terrain, not on any simulated true
    position — callers evaluating many positions should compute this once and reuse it.
    """
    positions = terrain.positions_xy
    bbox = terrain.bbox_xy
    augmented = _add_bounding_points(positions, bbox)
    vor = Voronoi(augmented)

    centroids = np.empty((len(positions), 2))
    for i in range(len(positions)):
        region = vor.regions[vor.point_region[i]]
        polygon = vor.vertices[region]  # finite by construction: dummy points absorb all unbounded regions
        clipped = _clip_polygon_to_bbox(polygon, bbox)
        centroids[i] = _polygon_centroid(clipped) if len(clipped) else positions[i]
    return centroids


def estimate_position(true_xy: np.ndarray, terrain: Terrain, centroids: np.ndarray | None = None) -> np.ndarray:
    """Cell-ID estimate: the Voronoi centroid of whichever real BTS is nearest ``true_xy``.

    Pass a precomputed ``centroids`` (from ``voronoi_cell_centroids``) when calling this
    repeatedly — recomputing the full Voronoi diagram per query would be wasteful.
    """
    if centroids is None:
        centroids = voronoi_cell_centroids(terrain)
    positions = terrain.positions_xy
    nearest_idx = int(np.argmin(np.linalg.norm(positions - true_xy, axis=1)))
    return centroids[nearest_idx]


def evaluate_accuracy(terrain: Terrain, n_positions: int = 100, seed: int | None = None) -> dict:
    """Median Cell-ID positioning error over ``n_positions`` random true positions in the terrain bbox."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = terrain.bbox_xy
    true_positions = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_positions, 2))

    centroids = voronoi_cell_centroids(terrain)
    errors = np.array([np.linalg.norm(estimate_position(p, terrain, centroids) - p) for p in true_positions])

    return {"median_error_m": float(np.median(errors)), "errors_m": errors}
