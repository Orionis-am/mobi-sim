"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) from scratch, Module F: Hansen's
standard formulation ("The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772).

The elite-driven pick: each generation's mean/covariance update is a WEIGHTED RECOMBINATION of
only the top-mu (elite) sampled individuals -- elitism isn't a side feature here, it IS the
mechanism driving both movement and the self-adapting mutation distribution (the covariance matrix
itself). Directly contrasts with `ag_scratch.py`'s hand-picked exponential sigma decay: instead of
a fixed schedule, CMA-ES *learns* its mutation step size and shape from which of its own elite
offspring actually succeeded.

Named explicitly in the project's own bonus list (`docs/SUJET.md`: "+5 pts max ... implementation
d'un AE supplementaire (CMA-ES, SPEA2, MOEA/D)" -- MOEA/D is already implemented for Pb2).

Validated against Rastrigin/Rosenbrock in `benchmark.py::run_extended_benchmark_suite`, then wired
as a bonus (non-mandated) solver for Pb1 (`pb1_codec.py::cma_es_codec`) -- see `abc_scratch.py`'s
module docstring for why this doesn't duplicate Pb1's mandated "AG via DEAP" role.

`run_cma_es` MINIMIZES `fitness_fn`, matching `ag_scratch.run_ga`'s convention.

**Boundary handling**: CMA-ES is naturally unconstrained (its Gaussian sampling can land outside
`bounds`). Sampled points are clipped back into bounds; if clipping changes a point, the internal
`y`/`z` vectors are RECOMPUTED from the clipped point (see `_sample_and_clip`) so the mean update,
evolution paths, and covariance update all stay consistent with the point that was actually
evaluated -- never with a point the algorithm merely intended to try. This matters most on narrow
genes (e.g. Pb1's `plc_level in [0,1]`), where clipping is frequent: keeping the algorithm's
internal model synced to reality avoids silently miscalibrating the covariance estimate.

**Stopping**: fixed `n_generations` only, no early-stopping/tolerance criteria -- matches every
other Module F algorithm's convention (`ag_scratch.run_ga`, DE, PSO), keeping `n_evaluations =
lambda * n_generations` exact and directly comparable.

**`history_best_fitness`/`n_evaluations` convention -- per-generation**, mirroring
`ag_scratch.run_ga` exactly: each generation evaluates exactly `lambda` (population size)
individuals, a known constant, so `n_evaluations = lambda * n_generations` is closed-form (unlike
`abc_scratch.py`'s data-dependent budget). Each history point therefore costs `lambda` evaluations
-- callers building `x_values` for `visualize.plot_convergence` should use
`[i * lambda for i in range(n_generations)]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, log, sqrt
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CMAESConfig:
    """Hyperparameters for `run_cma_es`. `pop_size` (lambda) and `initial_sigma` default to
    Hansen's standard recommendations, resolved inside `run_cma_es` once `bounds` (and therefore
    the dimension N) is known -- unlike `ag_scratch.GAConfig`'s fields, which are all resolvable at
    construction time. The remaining CMA-ES hyperparameters (mu, weights, mu_eff, cc, cs, c1, cmu,
    damps) are NOT exposed here at all: they are closed-form functions of N and lambda per Hansen's
    tutorial, not free knobs -- exposing them would only invite accidental deviation from "the
    standard formulas."""

    n_generations: int = 100
    pop_size: int | None = None  # None -> Hansen default: 4 + floor(3*ln(N))
    initial_sigma: float | None = None  # None -> 0.3 * mean(bound_range)
    seed: int | None = None


@dataclass(frozen=True)
class CMAESResult:
    best_x: np.ndarray
    best_fitness: float
    history_best_fitness: list[float]  # one entry per generation, lambda evaluations/point -- see module docstring
    n_evaluations: int


@dataclass(frozen=True)
class _CMAHyperparams:
    mu: int
    weights: np.ndarray  # length mu, positive, sums to 1
    mu_eff: float
    cc: float
    cs: float
    c1: float
    cmu: float
    damps: float
    chi_n: float  # E||N(0,I)||


def _default_pop_size(n: int) -> int:
    """Hansen's standard population-size default: `lambda = 4 + floor(3*ln(N))`."""
    return 4 + floor(3 * log(n))


def _hansen_defaults(n: int, lam: int) -> _CMAHyperparams:
    """Hansen's standard closed-form default hyperparameters (positive-weights-only "pure" CMA-ES,
    not active CMA-ES), purely a function of the dimension `n` and population size `lam`."""
    mu = lam // 2
    raw_weights = np.array([log((lam + 1) / 2.0) - log(i + 1) for i in range(mu)])
    weights = raw_weights / raw_weights.sum()
    mu_eff = 1.0 / np.sum(weights**2)

    cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    cs = (mu_eff + 2) / (n + mu_eff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0.0, sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
    chi_n = sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    return _CMAHyperparams(mu=mu, weights=weights, mu_eff=mu_eff, cc=cc, cs=cs, c1=c1, cmu=cmu, damps=damps, chi_n=chi_n)


def _bounds_arrays(bounds: Sequence[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _sample_and_clip(
    mean: np.ndarray, sigma: float, B: np.ndarray, D: np.ndarray, bounds: Sequence[tuple[float, float]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one offspring `x = mean + sigma * B @ (D * z)`, `z ~ N(0, I)`, then clip `x` into
    `bounds`. If clipping changed `x`, recompute `y`/`z` from the CLIPPED point so every downstream
    consumer (mean update, evolution paths, covariance update) stays consistent with the point that
    was actually evaluated -- see module docstring."""
    low, high = _bounds_arrays(bounds)
    z = rng.standard_normal(len(mean))
    y = B @ (D * z)
    x = mean + sigma * y
    x_clipped = np.clip(x, low, high)
    if not np.array_equal(x_clipped, x):
        y = (x_clipped - mean) / sigma
        z = B.T @ (y / D)
        x = x_clipped
    return x, y, z


def _weighted_recombine(weights: np.ndarray, vectors: list[np.ndarray]) -> np.ndarray:
    """Weighted sum of `vectors` (elite `y`s or `z`s) by `weights` -- the "elitism IS the mechanism"
    step: only the top-mu individuals ever contribute, weighted by rank."""
    return sum(w * v for w, v in zip(weights, vectors))


def _hsig_indicator(ps_norm: float, cs: float, gen: int, n: int, chi_n: float) -> float:
    """Heuristic stall-guard on the evolution path `pc`'s rank-one update: `1.0` (path is behaving
    as expected under random selection) unless `ps` has grown implausibly large for this point in
    the run, in which case `0.0` temporarily stops `pc` from accumulating further and compensates
    `C`'s rank-one term (see `_update_covariance`) so a spurious growth spurt doesn't get baked into
    the covariance estimate."""
    threshold = (1.4 + 2 / (n + 1)) * chi_n
    return 1.0 if (ps_norm / sqrt(1 - (1 - cs) ** (2 * gen))) < threshold else 0.0


def _update_covariance(C: np.ndarray, pc: np.ndarray, hp: _CMAHyperparams, ys: list[np.ndarray], order: np.ndarray, hsig: float) -> np.ndarray:
    """Rank-one (from the evolution path `pc`) + rank-mu (from the elite `ys`) covariance update."""
    rank_mu = sum(hp.weights[rank] * np.outer(ys[idx], ys[idx]) for rank, idx in enumerate(order))
    return (1 - hp.c1 - hp.cmu) * C + hp.c1 * (np.outer(pc, pc) + (1 - hsig) * hp.cc * (2 - hp.cc) * C) + hp.cmu * rank_mu


def run_cma_es(fitness_fn: Callable[[np.ndarray], float], bounds: Sequence[tuple[float, float]], config: CMAESConfig = CMAESConfig()) -> CMAESResult:
    """Minimize `fitness_fn` over `bounds` via CMA-ES (Hansen's standard formulation): each
    generation samples `lambda` offspring from `N(mean, sigma^2 * C)`, weighted-recombines the top
    `mu` (elite) by fitness into a new mean, and updates `sigma`/`C` from evolution paths tracking
    how those elites moved relative to the previous mean."""
    n = len(bounds)
    low, high = _bounds_arrays(bounds)
    lam = config.pop_size if config.pop_size is not None else _default_pop_size(n)
    sigma = config.initial_sigma if config.initial_sigma is not None else 0.3 * float(np.mean(high - low))
    hp = _hansen_defaults(n, lam)

    rng = np.random.default_rng(config.seed)
    mean = (low + high) / 2.0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)

    best_x = mean.copy()
    best_fitness = np.inf
    history: list[float] = []

    for gen in range(1, config.n_generations + 1):
        eigenvalues, B = np.linalg.eigh((C + C.T) / 2.0)
        eigenvalues = np.clip(eigenvalues, 1e-20, None)
        D = np.sqrt(eigenvalues)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        fs: list[float] = []
        for _ in range(lam):
            x, y, z = _sample_and_clip(mean, sigma, B, D, bounds, rng)
            f = float(fitness_fn(x))
            xs.append(x)
            ys.append(y)
            zs.append(z)
            fs.append(f)
            if f < best_fitness:
                best_fitness, best_x = f, x.copy()

        order = np.argsort(fs)[: hp.mu]
        y_w = _weighted_recombine(hp.weights, [ys[idx] for idx in order])
        z_w = _weighted_recombine(hp.weights, [zs[idx] for idx in order])

        mean = mean + sigma * y_w

        ps = (1 - hp.cs) * ps + sqrt(hp.cs * (2 - hp.cs) * hp.mu_eff) * (B @ z_w)
        hsig = _hsig_indicator(float(np.linalg.norm(ps)), hp.cs, gen, n, hp.chi_n)

        pc = (1 - hp.cc) * pc + hsig * sqrt(hp.cc * (2 - hp.cc) * hp.mu_eff) * y_w

        C = _update_covariance(C, pc, hp, ys, order, hsig)

        sigma = sigma * float(np.exp((hp.cs / hp.damps) * (np.linalg.norm(ps) / hp.chi_n - 1)))

        history.append(best_fitness)

    n_evaluations = lam * config.n_generations
    return CMAESResult(best_x=best_x, best_fitness=best_fitness, history_best_fitness=history, n_evaluations=n_evaluations)
