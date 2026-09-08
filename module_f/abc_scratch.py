"""Artificial Bee Colony (ABC) from scratch, Module F: Karaboga (2005)'s classic formulation.

Deliberately chosen as a contrast to `ag_scratch.py`'s GA: ABC has **no crossover operator and no
per-gene mutation-rate operator at all**. Its exploration move comes from a neighbor-difference
perturbation (employed/onlooker bees) and its diversity/escape mechanism comes from scout bees
abandoning stale food sources -- demonstrating what a population-based metaheuristic can do without
either GA operator.

Validated against Rastrigin/Rosenbrock in `benchmark.py::run_extended_benchmark_suite`, then wired
as a bonus (non-mandated) solver for Pb1 (`pb1_codec.py::abc_codec`) alongside the existing
DEAP-GA/random/grid search -- unlike `ag_scratch.py`'s GA (which would have duplicated Pb1's
mandated "AG via DEAP" role), ABC is a genuinely different algorithm and Pb1 has no
one-extra-solver-only restriction.

`run_abc` MINIMIZES `fitness_fn`, matching `ag_scratch.run_ga`'s convention.

**`history_best_fitness`/`n_evaluations` convention -- per-evaluation, not per-iteration.** Unlike
GA/CMA-ES (whose per-generation evaluation cost is a known constant), ABC's scout-bee phase costs a
*variable*, data-dependent number of evaluations per iteration (0 to `n_food_sources`, depending on
how many `trial` counters exceed `limit` that iteration) -- not knowable in advance, not constant
across runs. Logging one history point after every single `fitness_fn` call sidesteps this: `len(
history_best_fitness) == n_evaluations` always, directly comparable to `random_search`/
`hill_climbing`'s per-evaluation histories with no `x_values` rescaling needed in
`visualize.plot_convergence` -- avoids repeating the generation-vs-evaluation axis mismatch already
fixed once (see REPORT.md's "Complement -- `plot_convergence` melangeait deux unites d'axe X
differentes").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ABCConfig:
    """Hyperparameters for `run_abc`. `limit` (scout-bee abandonment threshold) defaults to
    Karaboga's classic `n_food_sources * dim` heuristic -- resolved inside `run_abc`, not here,
    since `dim` is only known once `bounds` is seen (unlike `ag_scratch.GAConfig`, whose fields are
    all resolvable at construction time)."""

    n_food_sources: int = 25  # == number of employed bees == number of onlooker bees (Karaboga's 1:1:1 ratio)
    n_iterations: int = 100
    limit: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class ABCResult:
    best_x: np.ndarray
    best_fitness: float
    history_best_fitness: list[float]  # one entry per evaluation -- see module docstring
    n_evaluations: int


def _bounds_arrays(bounds: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _neighbor_search_move(
    x_i: np.ndarray, source_population: np.ndarray, i: int, bounds: Sequence[tuple[float, float]], rng: np.random.Generator
) -> np.ndarray:
    """Karaboga's neighbor-search move: perturb ONE random dimension of `x_i` toward/away from one
    other random food source, `v[j] = x_i[j] + phi * (x_i[j] - x_k[j])`, `phi ~ U(-1, 1)`, clipped
    back into bounds. Not a mutation in the GA sense (no per-gene probability, no fixed
    distribution) and not a crossover (only one other individual contributes, and only to a single
    dimension) -- this move IS the entirety of ABC's exploration operator."""
    n = len(source_population)
    j = int(rng.integers(0, len(x_i)))
    k = i
    while k == i:
        k = int(rng.integers(0, n))
    phi = rng.uniform(-1.0, 1.0)
    v = x_i.copy()
    v[j] = x_i[j] + phi * (x_i[j] - source_population[k, j])
    low, high = _bounds_arrays(bounds)
    return np.clip(v, low, high)


def _abc_fitness(f: np.ndarray) -> np.ndarray:
    """Karaboga's minimization-to-positive-fitness transform, needed because onlooker-bee selection
    weights must be positive: `1/(1+f)` for `f >= 0`, `1+|f|` for `f < 0` (both equal `1` at `f=0`,
    both decrease monotonically as `f` gets worse)."""
    f = np.asarray(f, dtype=float)
    return np.where(f >= 0, 1.0 / (1.0 + f), 1.0 + np.abs(f))


def run_abc(fitness_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], config: ABCConfig = ABCConfig()) -> ABCResult:
    """Minimize `fitness_fn` over `bounds` via Artificial Bee Colony (Karaboga 2005): employed-bee
    phase (every food source tries one neighbor-search move, greedily replaced if it improves,
    otherwise its `trial` counter increments), onlooker-bee phase (roulette-wheel selection of
    sources weighted by `_abc_fitness`, same neighbor-move/greedy-replace), scout-bee phase (any
    source whose `trial` counter exceeds `limit` is abandoned and replaced by a fresh random draw)."""
    rng = np.random.default_rng(config.seed)
    dim = len(bounds)
    limit = config.limit if config.limit is not None else config.n_food_sources * dim
    low, high = _bounds_arrays(bounds)

    sources = rng.uniform(low, high, size=(config.n_food_sources, dim))
    fitnesses = np.empty(config.n_food_sources)
    trial = np.zeros(config.n_food_sources, dtype=int)

    history: list[float] = []
    best_fitness = np.inf
    best_x = sources[0].copy()
    for i in range(config.n_food_sources):
        f = float(fitness_fn(sources[i]))
        fitnesses[i] = f
        if f < best_fitness:
            best_fitness, best_x = f, sources[i].copy()
        history.append(best_fitness)
    n_evaluations = config.n_food_sources

    for _ in range(config.n_iterations):
        for i in range(config.n_food_sources):
            candidate = _neighbor_search_move(sources[i], sources, i, bounds, rng)
            candidate_fitness = float(fitness_fn(candidate))
            n_evaluations += 1
            if candidate_fitness < fitnesses[i]:
                sources[i], fitnesses[i], trial[i] = candidate, candidate_fitness, 0
            else:
                trial[i] += 1
            if candidate_fitness < best_fitness:
                best_fitness, best_x = candidate_fitness, candidate.copy()
            history.append(best_fitness)

        probabilities = _abc_fitness(fitnesses)
        probabilities = probabilities / probabilities.sum()
        for _ in range(config.n_food_sources):
            i = int(rng.choice(config.n_food_sources, p=probabilities))
            candidate = _neighbor_search_move(sources[i], sources, i, bounds, rng)
            candidate_fitness = float(fitness_fn(candidate))
            n_evaluations += 1
            if candidate_fitness < fitnesses[i]:
                sources[i], fitnesses[i], trial[i] = candidate, candidate_fitness, 0
            else:
                trial[i] += 1
            if candidate_fitness < best_fitness:
                best_fitness, best_x = candidate_fitness, candidate.copy()
            history.append(best_fitness)

        for i in range(config.n_food_sources):
            if trial[i] > limit:
                sources[i] = rng.uniform(low, high)
                fitnesses[i] = float(fitness_fn(sources[i]))
                n_evaluations += 1
                trial[i] = 0
                if fitnesses[i] < best_fitness:
                    best_fitness, best_x = float(fitnesses[i]), sources[i].copy()
                history.append(best_fitness)

    return ABCResult(best_x=best_x, best_fitness=best_fitness, history_best_fitness=history, n_evaluations=n_evaluations)
