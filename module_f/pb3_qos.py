"""Pb3 — QoS resource allocation: DE (scipy) vs. PSO (pyswarm) on `module_d.fitness.qos_fitness`
(`docs/SUJET.md` lines 342-350).

Both wrap `-qos_fitness(...)` since `qos_fitness` is maximized but `differential_evolution`/`pso`
minimize.

**pyswarm viability and reproducibility — verified live, not assumed**: the `pyswarm` PyPI
package (distinct from the actively-maintained `pyswarms`) was revived starting v0.7.0 through the
current v1.0.1 (maintainer `eggzec`), requires Python `>=3.10`, ships `win_amd64` wheels matching
this environment. Its `pso(..., seed=seed)` seeds a local RNG used for every random draw — no
global `np.random.seed()` workaround needed. Confirmed with a throwaway run: identical `seed`
gives an identical `x`/`fun`, and `nfev == swarmsize * (maxiter + 1)` exactly (verified against
`swarmsize=10, maxiter=5 -> nfev=60`). Pinned exactly (`pyswarm==1.0.1`, not a loose range) since
it's a small, single-maintainer revival.

**PSO convergence-curve reconstruction**: pyswarm exposes no per-iteration callback, only a final
`OptimizeResult`. The objective wrapper records every raw evaluation in order; since
`nfev = swarmsize * (maxiter + 1)` (first `swarmsize` evaluations are the initial swarm, each
subsequent chunk of `swarmsize` is one iteration), `run_pso` chunks the flat evaluation history
into `swarmsize`-sized groups and takes the running best-so-far of each chunk's best.

**DE convergence curve**: via `differential_evolution`'s `callback`, one extra `qos_fitness`
evaluation per generation to record the real (sign-corrected) value — negligible next to
`popsize * maxiter` evaluations already performed.

**`polish=False` required**: the chromosome mixes a discrete codec gene with a continuous
bandwidth gene per user; `polish=True`'s L-BFGS-B step assumes a gradient that doesn't exist
across the discrete gene.

**Equal-budget matching between DE and PSO is `compare.py`'s job, not this file's**: DE's total
evaluations are approximately `popsize * len(bounds) * (maxiter + 1)` (scipy multiplies `popsize`
by dimensionality internally) while PSO's are `swarmsize * (maxiter + 1)` — the two don't match
under equal default parameters, so matching them to one shared target budget is left to the
comparison layer, which can back-solve parameter combinations and document the resulting rounding
gap.

**`seed` is forwarded into every `qos_fitness` call, not just DE's/PSO's own optimizer seed**:
`qos_fitness` draws unseeded Gaussian loss noise internally (`module_d/session_sim.py`) unless a
`seed` is passed, same bug class caught and fixed in `module_d/test_module_d.py`'s
`test_fitness_matches_manual_weighted_formula`. Without forwarding it here, DE/PSO would each be
searching a fitness surface that changes randomly between evaluations of the *same* chromosome,
breaking both reproducibility and the callback-based convergence-history reconstruction (found via
a failing `test_reproducible_with_same_seed` before this fix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyswarm import pso
from scipy.optimize import differential_evolution

from module_d.fitness import DEFAULT_TOTAL_CAPACITY_KBPS, DEFAULT_USER_PROFILES, MAX_BANDWIDTH_KBPS, UserProfile, qos_fitness


def _bounds(profiles: tuple[UserProfile, ...] = DEFAULT_USER_PROFILES) -> list[tuple[float, float]]:
    """`[codec, bandwidth] * len(profiles)` bounds — codec gene in `[0, 2]` (snapped by
    `module_d.fitness.decode_chromosome`'s modulo), bandwidth in `[0, MAX_BANDWIDTH_KBPS]`."""
    return [(0.0, 2.0), (0.0, MAX_BANDWIDTH_KBPS)] * len(profiles)


@dataclass(frozen=True)
class OptResult:
    best_chromosome: np.ndarray
    best_fitness: float  # sign-corrected back to "higher is better"
    history_best_fitness: list[float]
    n_evaluations: int


def run_de(
    profiles: tuple[UserProfile, ...] = DEFAULT_USER_PROFILES,
    total_capacity_kbps: float = DEFAULT_TOTAL_CAPACITY_KBPS,
    maxiter: int = 100,
    popsize: int = 15,
    F: float = 0.8,
    CR: float = 0.9,
    seed: int | None = None,
) -> OptResult:
    """DE via `scipy.optimize.differential_evolution` (`docs/SUJET.md` line 348: F=0.8, CR=0.9)."""
    bounds = _bounds(profiles)
    history: list[float] = []

    def objective(x: np.ndarray) -> float:
        return -qos_fitness(x, profiles=profiles, total_capacity_kbps=total_capacity_kbps, seed=seed)

    def record(xk: np.ndarray, convergence: float) -> None:
        history.append(qos_fitness(xk, profiles=profiles, total_capacity_kbps=total_capacity_kbps, seed=seed))

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        mutation=F,
        recombination=CR,
        seed=seed,
        polish=False,
        callback=record,
    )
    return OptResult(best_chromosome=result.x, best_fitness=-result.fun, history_best_fitness=history, n_evaluations=result.nfev)


def run_pso(
    profiles: tuple[UserProfile, ...] = DEFAULT_USER_PROFILES,
    total_capacity_kbps: float = DEFAULT_TOTAL_CAPACITY_KBPS,
    swarmsize: int = 50,
    maxiter: int = 100,
    seed: int | None = None,
) -> OptResult:
    """PSO via `pyswarm` (`docs/SUJET.md` line 348)."""
    bounds = _bounds(profiles)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    raw_evaluations: list[float] = []

    def objective(x: np.ndarray) -> float:
        fitness = qos_fitness(x, profiles=profiles, total_capacity_kbps=total_capacity_kbps, seed=seed)
        raw_evaluations.append(fitness)
        return -fitness

    result = pso(objective, lb=lb, ub=ub, swarmsize=swarmsize, maxiter=maxiter, seed=seed)

    history: list[float] = []
    best_so_far = -np.inf
    for start in range(0, len(raw_evaluations), swarmsize):
        chunk = raw_evaluations[start : start + swarmsize]
        best_so_far = max(best_so_far, max(chunk))
        history.append(best_so_far)

    return OptResult(best_chromosome=result.x, best_fitness=-result.fun, history_best_fitness=history, n_evaluations=result.nfev)
