"""Pb2 — optimal BTS placement: NSGA-II vs. MOEA/D via pymoo, maximizing coverage while
minimizing interference and deployment cost, scored by `module_c.fitness.bts_coverage_fitness`
(`docs/SUJET.md` lines 330-340) — "le plus riche" problem in Module F.

`bts_coverage_fitness` already returns the exact `(f1, f2, f3)` 3-objective shape pymoo's
`Problem._evaluate` needs, so `BtsPlacementProblem` is a thin wrapper, no additional reformulation.

**Hypervolume reference point resolved**: the spec gives no value. Computed dynamically per run as
the componentwise nadir (worst observed value per objective across the run's own `F_history`) plus
a 10% margin — *added*, not multiplied, because `f1 = -coverage%` is negative and multiplying a
negative nadir by 1.1 would make it 10% *better* (less negative), not worse, which would violate
HV's requirement that the reference point be dominated by (worse than) every observed point. Adding
`0.1 * abs(nadir)` stays correct regardless of sign, and keeps NSGA-II vs. MOEA/D comparable for a
given N without a hand-picked global constant that breaks when N changes the objective ranges.

**"Best solution" for the map resolved**: the spec says "meilleure solution sur carte Folium"
without defining "best" under Pareto-optimality. `select_best_compromise` normalizes each objective
to [0, 1] across the final front and picks the point closest (Euclidean) to the ideal point
(0, 0, 0) — a standard, defensible knee-point heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from module_c.fitness import _get_default_terrain, bts_coverage_fitness
from module_c.terrain_sim import Terrain


class BtsPlacementProblem(Problem):
    """pymoo `Problem` wrapping `bts_coverage_fitness`: `n_var = 2*n_new_bts` normalized-coordinate
    genes, 3 objectives, bounds `[0, 1]` (chromosome genes are normalized, decoded against `terrain`
    inside `bts_coverage_fitness` itself)."""

    def __init__(self, n_new_bts: int, terrain: Terrain | None = None, **fitness_kwargs):
        self.terrain = terrain if terrain is not None else _get_default_terrain()
        self.fitness_kwargs = fitness_kwargs
        super().__init__(n_var=2 * n_new_bts, n_obj=3, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([bts_coverage_fitness(row, terrain=self.terrain, **self.fitness_kwargs) for row in X])


@dataclass(frozen=True)
class Pb2Result:
    F_history: list[np.ndarray]  # one (pop_size, 3) array per generation -- whole-population progress
    final_F: np.ndarray  # final non-dominated front, shape (n_front, 3), n_front <= pop_size
    final_X: np.ndarray
    n_evaluations: int


def run_nsga2(n_new_bts: int, terrain: Terrain | None = None, pop_size: int = 100, n_generations: int = 100, seed: int | None = None, **fitness_kwargs) -> Pb2Result:
    """NSGA-II via pymoo (`docs/SUJET.md` line 338).

    `seed` seeds both pymoo's own search (population init, selection) and, forwarded into
    `fitness_kwargs`, `bts_coverage_fitness`'s internal random test-point sampling — without the
    latter, `coverage_fraction` would redraw fresh random points on every single evaluation,
    making fitness noisy from one call to the next even for an identical chromosome and defeating
    reproducibility (and NSGA-II's own selection pressure, which assumes a stable fitness).
    """
    problem = BtsPlacementProblem(n_new_bts, terrain, seed=seed, **fitness_kwargs)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, save_history=True, verbose=False)
    return Pb2Result(
        F_history=[gen.pop.get("F") for gen in res.history],
        final_F=res.F,
        final_X=res.X,
        n_evaluations=res.algorithm.evaluator.n_eval,
    )


def run_moead(n_new_bts: int, terrain: Terrain | None = None, n_partitions: int = 12, n_generations: int = 100, seed: int | None = None, **fitness_kwargs) -> Pb2Result:
    """MOEA/D via pymoo, compared against NSGA-II via hypervolume (`docs/SUJET.md` line 340).

    `n_partitions=12` with 3 objectives gives ~91 reference directions (documented choice, a
    reasonably dense front) via `das-dennis`; MOEA/D's own population size is fixed by the number
    of reference directions, not independently choosable like NSGA-II's `pop_size`.
    """
    problem = BtsPlacementProblem(n_new_bts, terrain, seed=seed, **fitness_kwargs)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
    algorithm = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    res = minimize(problem, algorithm, ("n_gen", n_generations), seed=seed, save_history=True, verbose=False)
    return Pb2Result(
        F_history=[gen.pop.get("F") for gen in res.history],
        final_F=res.F,
        final_X=res.X,
        n_evaluations=res.algorithm.evaluator.n_eval,
    )


def hypervolume_history(result: Pb2Result, ref_point: np.ndarray | None = None) -> list[float]:
    """Hypervolume per generation, using `result.F_history`'s whole-population progress."""
    if ref_point is None:
        nadir = np.vstack(result.F_history).max(axis=0)
        margin = np.where(np.abs(nadir) > 0, 0.1 * np.abs(nadir), 1.0)
        ref_point = nadir + margin
    hv = HV(ref_point=ref_point)
    return [float(hv(gen_F)) for gen_F in result.F_history]


def select_best_compromise(result: Pb2Result) -> np.ndarray:
    """Knee-point pick on the final Pareto front closest to the normalized ideal point."""
    F = result.final_F
    f_min, f_max = F.min(axis=0), F.max(axis=0)
    span = np.where(f_max > f_min, f_max - f_min, 1.0)
    normalized = (F - f_min) / span
    distances = np.linalg.norm(normalized, axis=1)
    return result.final_X[int(np.argmin(distances))]
