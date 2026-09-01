"""BTS placement fitness for Module F's Pb2 (`docs/SUJET.md` lines 198, 330-340).

``bts_coverage_fitness(chromosome) -> (f1, f2, f3)`` is a deliberate departure from
module_a/module_b's single-scalar ``*_fitness()`` contract: Pb2 is explicitly 3-objective Pareto
(NSGA-II via pymoo, which expects an objectives array per individual, not one number), so this
returns the tuple directly rather than a scalarized score.

f1 = -coverage%, f2 = mean inter-cell interference, f3 = deployment cost — the spec gives f3 as
"proportional to distance to roads" but names no road dataset anywhere; per the design decision
recorded in REPORT.md, distance to the nearest *existing* real BTS is used as the road/
infrastructure proxy (real BTS cluster along roads/infrastructure in practice), keeping this
function cheap and fully offline — it will be called thousands of times by NSGA-II, the same
reasoning `module_a/fitness.py` documents for why `codec_fitness` avoids a real Whisper call per
evaluation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from module_c import opencellid_loader, terrain_sim
from module_c.terrain_sim import Terrain

COVERAGE_RADIUS_M = 3_000.0  # plausible rural macro-cell footprint, consistent with cell_id.py's ~3.4km median error on the real Aveyron sample
INTERFERENCE_RADIUS_M = 1_000.0
DEFAULT_N_TEST_POINTS = 300

_terrain_cache: Terrain | None = None


def _get_default_terrain() -> Terrain:
    """Lazily load and cache the real Aveyron terrain — module_c's equivalent of module_a/fitness.py's
    ``_reference_cache``. Tests always pass an explicit synthetic terrain instead, so this (and its
    disk read of docs/208.csv) is only exercised by real Module F runs, not the test suite."""
    global _terrain_cache
    if _terrain_cache is None:
        df = opencellid_loader.load_opencellid_csv()
        sample = opencellid_loader.sample_bts(df, n=500, seed=42)
        _terrain_cache = terrain_sim.build_terrain(sample)
    return _terrain_cache


def decode_chromosome(chromosome: Sequence[float], terrain: Terrain) -> np.ndarray:
    """Decode ``[x1, y1, ..., xN, yN]`` (normalized to [0, 1], clipped) into (N, 2) local-meter positions."""
    genes = np.asarray(chromosome, dtype=float)
    if len(genes) % 2 != 0:
        raise ValueError(f"chromosome length must be even ([x1,y1,...,xN,yN]), got {len(genes)}")

    pairs = np.clip(genes, 0.0, 1.0).reshape(-1, 2)
    x_min, x_max, y_min, y_max = terrain.bbox_xy
    x = x_min + pairs[:, 0] * (x_max - x_min)
    y = y_min + pairs[:, 1] * (y_max - y_min)
    return np.column_stack([x, y])


def coverage_fraction(terrain: Terrain, new_positions_xy: np.ndarray, n_test_points: int = DEFAULT_N_TEST_POINTS, coverage_radius_m: float = COVERAGE_RADIUS_M, seed: int | None = None) -> float:
    """Fraction of random test points in the terrain bbox within ``coverage_radius_m`` of any BTS (existing + new)."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = terrain.bbox_xy
    test_points = rng.uniform([x_min, y_min], [x_max, y_max], size=(n_test_points, 2))

    all_bts = np.vstack([terrain.positions_xy, new_positions_xy])
    diffs = test_points[:, None, :] - all_bts[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float((distances.min(axis=1) <= coverage_radius_m).mean())


def mean_interference(new_positions_xy: np.ndarray, existing_positions_xy: np.ndarray, interference_radius_m: float = INTERFERENCE_RADIUS_M) -> float:
    """Mean normalized interference each new BTS receives from nearby cells (existing + other new ones).

    Each neighbor within ``interference_radius_m`` contributes ``(radius - distance) / radius`` —
    0 at the radius's edge, 1 at zero distance — averaged over neighbors, then over new BTS.
    """
    all_bts = np.vstack([existing_positions_xy, new_positions_xy])
    contributions = []
    for position in new_positions_xy:
        distances = np.linalg.norm(all_bts - position, axis=1)
        distances = distances[distances > 1e-6]  # exclude the BTS's own (zero) distance to itself
        within_range = distances[distances < interference_radius_m]
        contributions.append(float(np.mean((interference_radius_m - within_range) / interference_radius_m)) if len(within_range) else 0.0)
    return float(np.mean(contributions)) if contributions else 0.0


def mean_cost_to_infrastructure(new_positions_xy: np.ndarray, existing_positions_xy: np.ndarray) -> float:
    """Mean distance from each new BTS to its nearest existing real BTS — the road/infrastructure cost proxy."""
    costs = [float(np.linalg.norm(existing_positions_xy - position, axis=1).min()) for position in new_positions_xy]
    return float(np.mean(costs)) if costs else 0.0


def bts_coverage_fitness_components(
    chromosome: Sequence[float],
    terrain: Terrain | None = None,
    n_test_points: int = DEFAULT_N_TEST_POINTS,
    coverage_radius_m: float = COVERAGE_RADIUS_M,
    interference_radius_m: float = INTERFERENCE_RADIUS_M,
    seed: int | None = None,
) -> dict[str, float]:
    """Full breakdown behind ``bts_coverage_fitness``'s (f1, f2, f3), for debugging/plotting."""
    terrain = terrain if terrain is not None else _get_default_terrain()
    new_positions = decode_chromosome(chromosome, terrain)

    coverage_pct = 100.0 * coverage_fraction(terrain, new_positions, n_test_points, coverage_radius_m, seed)
    interference = mean_interference(new_positions, terrain.positions_xy, interference_radius_m)
    cost_m = mean_cost_to_infrastructure(new_positions, terrain.positions_xy)

    return {
        "coverage_pct": coverage_pct,
        "interference": interference,
        "cost_m": cost_m,
        "f1": -coverage_pct,
        "f2": interference,
        "f3": cost_m,
    }


def bts_coverage_fitness(chromosome: Sequence[float], **kwargs) -> tuple[float, float, float]:
    """Module F's Pb2 fitness contract: (f1, f2, f3) = (-coverage%, interference, cost) for NSGA-II."""
    components = bts_coverage_fitness_components(chromosome, **kwargs)
    return (components["f1"], components["f2"], components["f3"])
