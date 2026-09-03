"""From-scratch genetic algorithm foundation for Module F: real-valued encoding, tournament
selection, SBX crossover, adaptive Gaussian mutation, elitist replacement (`docs/SUJET.md`
lines 312-319).

Validated against Rastrigin/Rosenbrock in `benchmark.py` before Module F's three real telecom
problems are tackled. Scope decision: this engine is used *only* for that Rastrigin/Rosenbrock
validation. Pb1 (codec config optimization) explicitly mandates "AG via DEAP" (`docs/SUJET.md`
line 321), so pointing this engine at Pb1 too would duplicate a GA already required there rather
than add required scope. CLAUDE.md's "validated against Rastrigin/Rosenbrock before being pointed
at the real problems" is read as describing methodology (prove the technique works on known
benchmarks first), not as requiring this exact code object to be reused as Pb1's solver.

`run_ga` MINIMIZES `fitness_fn` — the natural convention for Rastrigin/Rosenbrock, whose global
minimum is 0. Pb1/Pb2/Pb3's own fitness functions are maximized (or, for Pb2, multi-objective);
their solvers (DEAP/pymoo/DE/PSO) negate as needed, so this module's convention doesn't need to
match theirs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class GAConfig:
    """Hyperparameters for `run_ga`, matching `docs/SUJET.md` line 313-315 (tournament k=3, SBX
    eta=20, elitism keeps the top 10%) except for the adaptive-mutation schedule, which the spec
    only names ("mutation gaussienne adaptative (sigma decroissant)") without giving a formula —
    see `_adaptive_sigma`'s docstring for the resolved schedule."""

    pop_size: int = 50
    n_generations: int = 100
    tournament_size: int = 3
    sbx_eta: float = 20.0
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    mutation_sigma_init: float = 0.5  # fraction of each gene's bound range
    mutation_sigma_final: float = 0.01
    elite_fraction: float = 0.10
    seed: int | None = None


@dataclass(frozen=True)
class GAResult:
    best_x: np.ndarray
    best_fitness: float
    history_best_fitness: list[float]
    history_mean_fitness: list[float]
    n_evaluations: int


def _bounds_arrays(bounds: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _init_population(bounds: Sequence[tuple[float, float]], pop_size: int, rng: np.random.Generator) -> np.ndarray:
    low, high = _bounds_arrays(bounds)
    return rng.uniform(low, high, size=(pop_size, len(bounds)))


def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick one individual via k-way tournament (lowest fitness wins — `run_ga` minimizes)."""
    contenders = rng.integers(0, len(population), size=k)
    winner = contenders[np.argmin(fitnesses[contenders])]
    return population[winner].copy()


def sbx_crossover(
    parent1: np.ndarray, parent2: np.ndarray, eta: float, bounds: Sequence[tuple[float, float]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (Deb & Agrawal): higher `eta` keeps children closer to parents."""
    u = rng.random(len(parent1))
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1.0)),
        (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0)),
    )
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    low, high = _bounds_arrays(bounds)
    return np.clip(child1, low, high), np.clip(child2, low, high)


def gaussian_mutation(individual: np.ndarray, sigma: float, bounds: Sequence[tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    """Perturb each gene by `N(0, sigma * gene_range)`, clipped back into bounds."""
    low, high = _bounds_arrays(bounds)
    noise = rng.normal(0.0, sigma * (high - low))
    return np.clip(individual + noise, low, high)


def _adaptive_sigma(generation: int, n_generations: int, sigma_init: float, sigma_final: float) -> float:
    """Exponential decay from `sigma_init` to `sigma_final` over the run — the spec names an
    adaptive/decreasing schedule without specifying its shape; exponential decay is the standard
    choice (linear decay overshoots early, geometric/exponential tapers exploration smoothly)."""
    if n_generations <= 1:
        return sigma_final
    progress = generation / (n_generations - 1)
    return sigma_init * (sigma_final / sigma_init) ** progress


def run_ga(fitness_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], config: GAConfig = GAConfig()) -> GAResult:
    """Minimize `fitness_fn` over `bounds` via tournament selection + SBX crossover + adaptive
    Gaussian mutation + elitist replacement (`docs/SUJET.md` lines 313-315)."""
    rng = np.random.default_rng(config.seed)
    population = _init_population(bounds, config.pop_size, rng)
    fitnesses = np.array([fitness_fn(ind) for ind in population])
    n_evaluations = config.pop_size

    history_best = [float(fitnesses.min())]
    history_mean = [float(fitnesses.mean())]
    n_elite = max(1, ceil(config.elite_fraction * config.pop_size))

    for gen in range(1, config.n_generations):
        sigma = _adaptive_sigma(gen, config.n_generations, config.mutation_sigma_init, config.mutation_sigma_final)
        elite_idx = np.argsort(fitnesses)[:n_elite]
        next_population = [population[i].copy() for i in elite_idx]

        while len(next_population) < config.pop_size:
            parent1 = tournament_selection(population, fitnesses, config.tournament_size, rng)
            parent2 = tournament_selection(population, fitnesses, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                child1, child2 = sbx_crossover(parent1, parent2, config.sbx_eta, bounds, rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            for child in (child1, child2):
                if len(next_population) >= config.pop_size:
                    break
                if rng.random() < config.mutation_rate:
                    child = gaussian_mutation(child, sigma, bounds, rng)
                next_population.append(child)

        population = np.array(next_population[: config.pop_size])
        fitnesses = np.array([fitness_fn(ind) for ind in population])
        n_evaluations += config.pop_size

        history_best.append(min(history_best[-1], float(fitnesses.min())))
        history_mean.append(float(fitnesses.mean()))

    best_idx = int(np.argmin(fitnesses))
    return GAResult(
        best_x=population[best_idx],
        best_fitness=history_best[-1],
        history_best_fitness=history_best,
        history_mean_fitness=history_mean,
        n_evaluations=n_evaluations,
    )
