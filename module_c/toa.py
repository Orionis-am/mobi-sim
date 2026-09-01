"""TOA (Time of Arrival) trilateration (`docs/SUJET.md` line 192-193).

Propagation delay is simulated as ``distance / c`` plus Gaussian timing noise, then a position is
solved for via `scipy.optimize.minimize` (SLSQP) minimizing the sum of squared residuals between
measured and hypothesized distances to the anchor BTS — the spec names the delay model and the
solver but not the objective function itself, so this is the natural least-squares choice.
Visualizing the range circles on a Folium map is ``map_viz.py``'s job, not this file's.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from module_c.terrain_sim import Terrain

SPEED_OF_LIGHT_M_S = 299_792_458.0
DEFAULT_TIMING_NOISE_STD_S = 50e-9  # ~15 m of 1-sigma range noise at c — plausible TOA/GPS-like ranging
DEFAULT_K_ANCHORS = 4


def _nearest_anchors(true_xy: np.ndarray, terrain: Terrain, k: int) -> np.ndarray:
    positions = terrain.positions_xy
    k = min(k, len(positions))
    distances = np.linalg.norm(positions - true_xy, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    return positions[nearest_idx]


def simulate_pseudoranges(
    true_xy: np.ndarray, anchors_xy: np.ndarray, timing_noise_std_s: float = DEFAULT_TIMING_NOISE_STD_S, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Simulate one noisy TOA-derived distance measurement per anchor."""
    rng = rng if rng is not None else np.random.default_rng()
    true_distances = np.linalg.norm(anchors_xy - true_xy, axis=1)
    delays_s = true_distances / SPEED_OF_LIGHT_M_S
    noisy_delays_s = np.clip(delays_s + rng.normal(0.0, timing_noise_std_s, size=len(anchors_xy)), 0.0, None)
    return noisy_delays_s * SPEED_OF_LIGHT_M_S


def trilaterate(anchors_xy: np.ndarray, measured_distances: np.ndarray, initial_guess: np.ndarray | None = None) -> np.ndarray:
    """Solve for the (x, y) minimizing squared residuals between measured and hypothesized anchor distances."""
    if initial_guess is None:
        initial_guess = anchors_xy.mean(axis=0)

    def objective(pos: np.ndarray) -> float:
        hypothesized = np.linalg.norm(anchors_xy - pos, axis=1)
        return float(np.sum((hypothesized - measured_distances) ** 2))

    result = minimize(objective, initial_guess, method="SLSQP")
    return result.x


def estimate_position(
    true_xy: np.ndarray, terrain: Terrain, k: int = DEFAULT_K_ANCHORS, timing_noise_std_s: float = DEFAULT_TIMING_NOISE_STD_S, rng: np.random.Generator | None = None
) -> np.ndarray:
    """End-to-end: pick the k nearest real BTS as anchors, simulate ranges, trilaterate."""
    anchors = _nearest_anchors(true_xy, terrain, k)
    measured = simulate_pseudoranges(true_xy, anchors, timing_noise_std_s, rng)
    return trilaterate(anchors, measured)


def evaluate_accuracy(
    terrain: Terrain, n_positions: int = 100, k: int = DEFAULT_K_ANCHORS, timing_noise_std_s: float = DEFAULT_TIMING_NOISE_STD_S, seed: int | None = None
) -> dict:
    """Median TOA positioning error over ``n_positions`` random true positions in the terrain bbox."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = terrain.bbox_xy
    true_positions = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_positions, 2))

    errors = np.array([np.linalg.norm(estimate_position(p, terrain, k, timing_noise_std_s, rng) - p) for p in true_positions])

    return {"median_error_m": float(np.median(errors)), "errors_m": errors}
