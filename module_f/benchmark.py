"""Rastrigin/Rosenbrock validation of `ag_scratch`'s from-scratch GA against random search and
hill climbing, plus the spec's population/crossover/mutation hyperparameter sweep (`docs/SUJET.md`
lines 316-319).

**Hill climbing lives here, not in `ag_scratch.py`**: it's a comparison baseline for this file's
validation, outside the spec's "AG from scratch" scope (`docs/SUJET.md` line 313 names tournament
selection/SBX/mutation/elitism as the AG's own operators — hill climbing and random search are
both external baselines, not GA components).

**Heatmap ambiguity resolved**: the spec asks for one heatmap of final scores averaged over 10 runs,
swept over 3 hyperparameters (population, crossover rate, mutation rate) — that can't collapse into
one 2D grid. `hyperparameter_sweep` returns the full 3D array (population x crossover x mutation);
rendering slices it into **three** 2D heatmaps (crossover x mutation), one per population size, via
`visualize.plot_hyperparam_heatmap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from module_f import ag_scratch
from module_f.ag_scratch import GAConfig

RASTRIGIN_BOUNDS = (-5.12, 5.12)
ROSENBROCK_BOUNDS = (-5.0, 10.0)


def rastrigin(x: np.ndarray) -> float:
    """Multimodal benchmark, global minimum 0 at the origin."""
    x = np.asarray(x, dtype=float)
    a = 10.0
    return float(a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Narrow-valley benchmark, global minimum 0 at all-ones."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


@dataclass(frozen=True)
class BaselineResult:
    best_x: np.ndarray
    best_fitness: float
    history_best_fitness: list[float]
    n_evaluations: int


def _bounds_arrays(bounds: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def random_search(fitness_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], n_evaluations: int, seed: int | None = None) -> BaselineResult:
    """Uniform-random sampling baseline, one evaluation per draw."""
    rng = np.random.default_rng(seed)
    low, high = _bounds_arrays(bounds)
    best_x, best_fitness = None, np.inf
    history: list[float] = []
    for _ in range(n_evaluations):
        x = rng.uniform(low, high)
        fitness = fitness_fn(x)
        if fitness < best_fitness:
            best_x, best_fitness = x, fitness
        history.append(best_fitness)
    return BaselineResult(best_x=best_x, best_fitness=best_fitness, history_best_fitness=history, n_evaluations=n_evaluations)


def hill_climbing(fitness_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], n_iterations: int, step_size: float = 0.1, seed: int | None = None) -> BaselineResult:
    """Stochastic hill climbing: one Gaussian perturbation per iteration, accepted only if it improves fitness.

    One evaluation per iteration, directly comparable to `random_search`/`run_ga` on evaluation count.
    """
    rng = np.random.default_rng(seed)
    low, high = _bounds_arrays(bounds)
    current_x = rng.uniform(low, high)
    current_fitness = fitness_fn(current_x)
    history = [current_fitness]
    for _ in range(n_iterations - 1):
        candidate = np.clip(current_x + rng.normal(0.0, step_size * (high - low)), low, high)
        candidate_fitness = fitness_fn(candidate)
        if candidate_fitness < current_fitness:
            current_x, current_fitness = candidate, candidate_fitness
        history.append(current_fitness)
    return BaselineResult(best_x=current_x, best_fitness=current_fitness, history_best_fitness=history, n_evaluations=n_iterations)


def run_benchmark_suite(dim: int, seed: int | None = None, ga_config: GAConfig | None = None) -> dict:
    """Run GA vs. random search vs. hill climbing on both benchmarks at `dim` dimensions, matched
    to the same evaluation budget (`docs/SUJET.md` line 316: "2D et 10D")."""
    ga_config = ga_config if ga_config is not None else GAConfig(seed=seed)
    n_evaluations = ga_config.pop_size * ga_config.n_generations

    results: dict[str, dict[str, object]] = {}
    for name, fn, bound_range in (("rastrigin", rastrigin, RASTRIGIN_BOUNDS), ("rosenbrock", rosenbrock, ROSENBROCK_BOUNDS)):
        bounds = [bound_range] * dim
        results[name] = {
            "ga": ag_scratch.run_ga(fn, bounds, ga_config),
            "random_search": random_search(fn, bounds, n_evaluations, seed=seed),
            "hill_climbing": hill_climbing(fn, bounds, n_evaluations, seed=seed),
        }
    return results


def hyperparameter_sweep(
    fitness_fn: Callable[[np.ndarray], float],
    bounds: tuple[float, float],
    dim: int,
    pop_sizes: tuple[int, ...] = (10, 50, 200),
    crossover_rates: tuple[float, ...] = (0.6, 0.8, 0.95),
    mutation_rates: tuple[float, ...] = (0.01, 0.05, 0.2),
    n_runs: int = 10,
    n_generations: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Mean final best-fitness over `n_runs`, for every (pop_size, crossover_rate, mutation_rate)
    combination (`docs/SUJET.md` lines 318-319) — shape `(len(pop_sizes), len(crossover_rates),
    len(mutation_rates))`. Per-run seeding is `seed + run_index`, matching the idiom already used
    by `module_d/fitness.py::qos_fitness_components`.
    """
    problem_bounds = [bounds] * dim
    grid = np.zeros((len(pop_sizes), len(crossover_rates), len(mutation_rates)))

    for i, pop_size in enumerate(pop_sizes):
        for j, crossover_rate in enumerate(crossover_rates):
            for k, mutation_rate in enumerate(mutation_rates):
                scores = []
                for run in range(n_runs):
                    run_seed = None if seed is None else seed + run
                    config = GAConfig(pop_size=pop_size, n_generations=n_generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate, seed=run_seed)
                    scores.append(ag_scratch.run_ga(fitness_fn, problem_bounds, config).best_fitness)
                grid[i, j, k] = float(np.mean(scores))
    return grid
