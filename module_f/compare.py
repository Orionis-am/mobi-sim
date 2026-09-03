"""Comparative table and fixed-budget cross-algorithm curves across Module F's three problems
(`docs/SUJET.md` lines 352-359): best value, convergence time, variance over 10 runs, recommended
parameters, per algorithm per problem.

**Convergence-time metric ordering resolved**: equal-`n_evaluations` is the *primary* fairness axis
for every cross-algorithm curve (`run_fixed_budget_comparison`), matching the spec's explicit
"budget d'evaluations fixe et identique entre algorithmes" (line 356). `convergence_time_s` (wall
clock) is recorded in `AlgoRunSummary` and shown as one extra table column, but never used to gate
or normalize the primary comparison — the spec lists both without saying which is primary.

**Style**: plain dataclass/dict, not `pandas.DataFrame` — none of A/B/C/D's `compare.py`-equivalents
(`module_b/compare.py` is the closest precedent) use pandas for their tables; staying consistent
rather than introducing a new style here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from module_f import pb3_qos


@dataclass(frozen=True)
class AlgoRunSummary:
    problem: str
    algorithm: str
    best_value: float
    convergence_time_s: float  # secondary metric only -- see module docstring
    variance_over_runs: float
    recommended_params: dict
    n_evaluations: int


def summarize_runs(
    problem: str,
    algorithm: str,
    run_fn: Callable[..., object],
    n_runs: int = 10,
    seed: int | None = None,
    maximize: bool = True,
    recommended_params: dict | None = None,
    **run_fn_kwargs,
) -> AlgoRunSummary:
    """Run `run_fn(seed=..., **run_fn_kwargs)` `n_runs` times (seed `None` or `seed + run_index`),
    summarizing best value, wall-clock time, and variance. `run_fn` must return an object exposing
    `.best_fitness`/`.n_evaluations` — `GAResult`/`BaselineResult`/`DeapGaResult`/`OptResult` all do.

    `maximize` picks whether "best" means the run with the highest or lowest `best_fitness` — needed
    because Module F's own conventions differ (`ag_scratch`/`benchmark` minimize raw fitness,
    `pb1_codec`/`pb3_qos` maximize a sign-corrected score).
    """
    values: list[float] = []
    n_evaluations = 0
    start = time.perf_counter()
    for run in range(n_runs):
        run_seed = None if seed is None else seed + run
        result = run_fn(seed=run_seed, **run_fn_kwargs)
        values.append(float(result.best_fitness))
        n_evaluations = result.n_evaluations
    elapsed = time.perf_counter() - start

    return AlgoRunSummary(
        problem=problem,
        algorithm=algorithm,
        best_value=max(values) if maximize else min(values),
        convergence_time_s=elapsed,
        variance_over_runs=float(np.var(values)),
        recommended_params=recommended_params if recommended_params is not None else dict(run_fn_kwargs),
        n_evaluations=n_evaluations,
    )


def build_comparison_table(summaries: list[AlgoRunSummary]) -> list[dict]:
    """One row per `(problem, algorithm)`, ready for a report table."""
    return [
        {
            "problem": s.problem,
            "algorithm": s.algorithm,
            "best_value": s.best_value,
            "convergence_time_s": s.convergence_time_s,
            "variance_over_runs": s.variance_over_runs,
            "recommended_params": s.recommended_params,
            "n_evaluations": s.n_evaluations,
        }
        for s in summaries
    ]


def _fixed_budget_pb3(n_evaluations: int, n_runs: int, seed: int | None) -> dict[str, list[float]]:
    n_dims = 2 * len(pb3_qos.DEFAULT_USER_PROFILES)
    maxiter = 20
    # DE's internal eval count ~= popsize * n_dims * (maxiter+1) (scipy multiplies popsize by
    # dimensionality); PSO's = swarmsize * (maxiter+1) -- back-solved to hit the same target
    # budget, accepting the small rounding gap from each formula's own integer population size.
    de_popsize = max(1, round(n_evaluations / (n_dims * (maxiter + 1))))
    pso_swarmsize = max(1, round(n_evaluations / (maxiter + 1)))

    de_histories, pso_histories = [], []
    for run in range(n_runs):
        run_seed = None if seed is None else seed + run
        de_histories.append(pb3_qos.run_de(maxiter=maxiter, popsize=de_popsize, seed=run_seed).history_best_fitness)
        pso_histories.append(pb3_qos.run_pso(maxiter=maxiter, swarmsize=pso_swarmsize, seed=run_seed).history_best_fitness)

    min_len = min(min(len(h) for h in de_histories), min(len(h) for h in pso_histories))
    return {
        "de": list(np.mean([h[:min_len] for h in de_histories], axis=0)),
        "pso": list(np.mean([h[:min_len] for h in pso_histories], axis=0)),
    }


def run_fixed_budget_comparison(problem: str, n_evaluations: int, n_runs: int = 10, seed: int | None = None) -> dict[str, list[float]]:
    """Re-run every algorithm applicable to `problem` at the same `n_evaluations` budget, `n_runs`
    times each, averaging the per-run convergence histories -- `{"algorithm": mean_history}`,
    ready for `visualize.plot_convergence`. Currently wired for `"pb3_qos"` (DE vs. PSO), the
    simplest two-algorithm case.
    """
    if problem == "pb3_qos":
        return _fixed_budget_pb3(n_evaluations, n_runs, seed)
    raise ValueError(f"no fixed-budget comparison wired for problem={problem!r}")
