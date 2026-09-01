"""Wi-Fi fingerprinting via k-NN on a simulated RSSI grid (`docs/SUJET.md` line 194-195).

Reuses the terrain's real BTS positions as the fingerprint's signal sources — a separate, smaller
"Wi-Fi AP" dataset isn't specified anywhere in the spec, and every other Module C algorithm
(Cell-ID, TOA) is already built the same way: real infrastructure positions, simulated radio
behavior on top. RSSI is a standard log-distance path-loss model, not a physically calibrated one
(same spirit as `module_a/codecs.py`'s SNR-based degradation — realistic *shape*, not a certified
propagation model).

Wi-Fi fingerprinting is inherently a local-area technique (an office, a campus, a neighborhood —
hundreds of meters to a few km), unlike Cell-ID/TOA which operate over the whole macro network. A
100m grid spanning the *entire* terrain (the Aveyron sample is ~160km x 122km) would be ~1.9
million cells — the grid is scoped to a local zone around a center point instead, not the full
terrain bounding box.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from module_c.terrain_sim import Terrain

GRID_CELL_SIZE_M = 100.0
DEFAULT_ZONE_SIZE_M = 2_000.0
DEFAULT_K = 3
RSSI_AT_1M_DBM = -30.0
PATH_LOSS_EXPONENT = 3.0
DEFAULT_RSSI_NOISE_STD_DB = 4.0


def _zone_bbox(terrain: Terrain, zone_center_xy: np.ndarray | None, zone_size_m: float) -> tuple[float, float, float, float]:
    if zone_center_xy is None:
        zone_center_xy = terrain.positions_xy.mean(axis=0)
    cx, cy = zone_center_xy
    half = zone_size_m / 2
    return cx - half, cx + half, cy - half, cy + half


def simulate_rssi(position_xy: np.ndarray, bts_positions_xy: np.ndarray, noise_std_db: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """One simulated RSSI reading per BTS, via a log-distance path-loss model."""
    distances = np.linalg.norm(bts_positions_xy - position_xy, axis=1)
    rssi = RSSI_AT_1M_DBM - 10.0 * PATH_LOSS_EXPONENT * np.log10(np.clip(distances, 1.0, None))
    if noise_std_db > 0:
        rng = rng if rng is not None else np.random.default_rng()
        rssi = rssi + rng.normal(0.0, noise_std_db, size=rssi.shape)
    return rssi


def build_fingerprint_grid(
    terrain: Terrain, zone_center_xy: np.ndarray | None = None, zone_size_m: float = DEFAULT_ZONE_SIZE_M, cell_size_m: float = GRID_CELL_SIZE_M
) -> tuple[np.ndarray, np.ndarray]:
    """Noiseless reference fingerprints on a regular grid: (centroids (M, 2), fingerprints (M, n_bts)).

    The grid spans a ``zone_size_m`` x ``zone_size_m`` square centered on ``zone_center_xy``
    (defaults to the terrain's own BTS centroid), not the whole terrain bounding box.
    """
    x_min, x_max, y_min, y_max = _zone_bbox(terrain, zone_center_xy, zone_size_m)
    xs = np.arange(x_min + cell_size_m / 2, x_max, cell_size_m)
    ys = np.arange(y_min + cell_size_m / 2, y_max, cell_size_m)
    if len(xs) == 0:
        xs = np.array([(x_min + x_max) / 2])
    if len(ys) == 0:
        ys = np.array([(y_min + y_max) / 2])

    grid_x, grid_y = np.meshgrid(xs, ys)
    centroids = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    positions = terrain.positions_xy
    fingerprints = np.array([simulate_rssi(c, positions) for c in centroids])
    return centroids, fingerprints


def fit_knn(fingerprints: np.ndarray, k: int = DEFAULT_K) -> NearestNeighbors:
    model = NearestNeighbors(n_neighbors=min(k, len(fingerprints)))
    model.fit(fingerprints)
    return model


def estimate_position(query_fingerprint: np.ndarray, centroids: np.ndarray, model: NearestNeighbors) -> np.ndarray:
    """k-NN position estimate: mean of the k grid centroids whose fingerprint is closest to the query."""
    _, indices = model.kneighbors(query_fingerprint.reshape(1, -1))
    return centroids[indices[0]].mean(axis=0)


def evaluate_accuracy(
    terrain: Terrain,
    n_positions: int = 100,
    zone_center_xy: np.ndarray | None = None,
    zone_size_m: float = DEFAULT_ZONE_SIZE_M,
    cell_size_m: float = GRID_CELL_SIZE_M,
    k: int = DEFAULT_K,
    noise_std_db: float = DEFAULT_RSSI_NOISE_STD_DB,
    seed: int | None = None,
) -> dict:
    """Median Wi-Fi fingerprinting error over ``n_positions`` random true positions within the same local zone the grid covers."""
    rng = np.random.default_rng(seed)
    centroids, fingerprints = build_fingerprint_grid(terrain, zone_center_xy, zone_size_m, cell_size_m)
    model = fit_knn(fingerprints, k)

    x_min, x_max, y_min, y_max = _zone_bbox(terrain, zone_center_xy, zone_size_m)
    true_positions = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_positions, 2))
    positions = terrain.positions_xy

    errors = np.array(
        [np.linalg.norm(estimate_position(simulate_rssi(p, positions, noise_std_db, rng), centroids, model) - p) for p in true_positions]
    )

    return {"median_error_m": float(np.median(errors)), "errors_m": errors}


def sweep_noise(terrain: Terrain, noise_levels_db: list[float], n_positions: int = 100, seed: int | None = None) -> list[dict]:
    """``evaluate_accuracy`` repeated across noise levels — the spec's "précision en fonction du bruit" curve.

    Derives one seed per noise level (rather than sharing a single RNG across the whole sweep) so
    points stay reproducible without one level's sampling noise leaking into the next — same idea
    as ``module_b/network_sim.py``'s ``sweep_overload``.
    """
    results = []
    for i, noise_db in enumerate(noise_levels_db):
        point_seed = None if seed is None else seed + i
        result = evaluate_accuracy(terrain, n_positions=n_positions, noise_std_db=noise_db, seed=point_seed)
        results.append({"noise_std_db": noise_db, **result})
    return results
