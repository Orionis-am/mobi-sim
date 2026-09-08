"""Pb1 — codec configuration optimization: AG via DEAP vs. random search vs. exhaustive grid
search, all scored by `module_a.fitness.codec_fitness` (`docs/SUJET.md` lines 321-328).

**Grid discretization resolved**: `plc_level` is continuous (`[0, 1]`) but an exhaustive grid
search needs a finite step count the spec doesn't give. `GRID_PLC_STEPS = 5` (`np.linspace(0, 1,
5)`) is a documented, plausible choice — same pattern as `module_d`'s `DEFAULT_USER_PROFILES`.
Combined with the 5 bitrates x 4 frame sizes x 3 codecs already fixed by the chromosome's discrete
genes, the full grid is `5*4*5*3 = 300` evaluations.

**DEAP reproducibility resolved**: DEAP's `tools.sel*`/`tools.mut*`/`tools.cx*` draw from Python's
stdlib `random` module, not numpy — `deap_ga_codec` seeds via `random.seed(seed)`, not
`np.random.seed()`. `codec_fitness`'s own internal packet-loss RNG is seeded too (via `seed` in
`codec_fitness_kwargs`), so every individual evaluates deterministically regardless of which point
in chromosome-space DEAP visits.

No repair operator is needed for DEAP's real-valued individuals: `module_a.fitness.
decode_chromosome`'s `_nearest`/modulo snapping already tolerates any float, by design (it exists
precisely so a GA can mutate genes freely without producing invalid configs).

**`cma_es_codec`/`abc_codec` — bonus, non-mandated solvers, added alongside the three above**:
unlike `module_f/ag_scratch.py`'s from-scratch GA (deliberately never reused for Pb1, since Pb1
already mandates "AG via DEAP" and reusing it there would just duplicate that role — see
`ag_scratch.py`'s own module docstring), CMA-ES (`module_f/cmaes_scratch.py`) and ABC
(`module_f/abc_scratch.py`) are genuinely different algorithms, and Pb1 has no
one-extra-solver-only restriction. Both wrappers below follow `random_search_codec`'s exact shape:
bounds fixed to `CHROMOSOME_BOUNDS`, `seed` forwarded into every single `codec_fitness` call (not
just the optimizer's own seed), negated internally since both minimize by convention while
`codec_fitness` is maximized.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

import numpy as np
from deap import algorithms, base, creator, tools

from module_a.fitness import BITRATE_CHOICES_KBPS, CODEC_BY_INDEX, FRAME_SIZE_CHOICES_MS, codec_fitness
from module_f.abc_scratch import ABCConfig, run_abc
from module_f.cmaes_scratch import CMAESConfig, run_cma_es

CHROMOSOME_BOUNDS = [
    (float(min(BITRATE_CHOICES_KBPS)), float(max(BITRATE_CHOICES_KBPS))),
    (float(min(FRAME_SIZE_CHOICES_MS)), float(max(FRAME_SIZE_CHOICES_MS))),
    (0.0, 1.0),
    (0.0, float(len(CODEC_BY_INDEX) - 1)),
]
GRID_PLC_STEPS = 5  # np.linspace(0, 1, 5) -- documented assumption, see REPORT.md

if not hasattr(creator, "FitnessMaxCodec"):
    creator.create("FitnessMaxCodec", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualCodec"):
    creator.create("IndividualCodec", list, fitness=creator.FitnessMaxCodec)


@dataclass(frozen=True)
class DeapGaResult:
    best_chromosome: list[float]
    best_fitness: float
    history_best_fitness: list[float]
    n_evaluations: int


def _make_toolbox(seed: int | None, **codec_fitness_kwargs) -> base.Toolbox:
    toolbox = base.Toolbox()
    low = [b[0] for b in CHROMOSOME_BOUNDS]
    up = [b[1] for b in CHROMOSOME_BOUNDS]

    def init_individual():
        genes = [random.uniform(lo, hi) for lo, hi in CHROMOSOME_BOUNDS]
        return creator.IndividualCodec(genes)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (codec_fitness(ind, seed=seed, **codec_fitness_kwargs),))
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=low, up=up)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0 / len(CHROMOSOME_BOUNDS))
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def deap_ga_codec(pop_size: int = 50, n_generations: int = 100, cxpb: float = 0.8, mutpb: float = 0.2, seed: int | None = None, **codec_fitness_kwargs) -> DeapGaResult:
    """AG via DEAP maximizing `codec_fitness` (`docs/SUJET.md` line 321-328): tournament selection
    (k=3), SBX crossover (eta=20, same as `ag_scratch.py` for methodological consistency, though
    not spec-required here), polynomial-bounded mutation (eta=20), a `HallOfFame` tracking the best
    individual ever seen and merged back into the selection pool each generation."""
    random.seed(seed)
    toolbox = _make_toolbox(seed, **codec_fitness_kwargs)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    for ind, fit in zip(population, map(toolbox.evaluate, population)):
        ind.fitness.values = fit
    hof.update(population)
    history = [hof[0].fitness.values[0]]
    n_evaluations = pop_size

    for _ in range(1, n_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        for ind, fit in zip(offspring, map(toolbox.evaluate, offspring)):
            ind.fitness.values = fit
        n_evaluations += len(offspring)

        hof.update(offspring)
        population = toolbox.select(offspring + [toolbox.clone(ind) for ind in hof], k=pop_size)
        history.append(hof[0].fitness.values[0])

    best = hof[0]
    return DeapGaResult(best_chromosome=list(best), best_fitness=best.fitness.values[0], history_best_fitness=history, n_evaluations=n_evaluations)


def random_search_codec(n_evaluations: int, seed: int | None = None, **codec_fitness_kwargs) -> DeapGaResult:
    """Uniform-random sampling baseline, one evaluation per draw."""
    rng = np.random.default_rng(seed)
    best_chromosome: list[float] | None = None
    best_fitness = -np.inf
    history: list[float] = []
    for _ in range(n_evaluations):
        chromosome = [rng.uniform(lo, hi) for lo, hi in CHROMOSOME_BOUNDS]
        fitness = codec_fitness(chromosome, seed=seed, **codec_fitness_kwargs)
        if fitness > best_fitness:
            best_chromosome, best_fitness = chromosome, fitness
        history.append(best_fitness)
    return DeapGaResult(best_chromosome=best_chromosome, best_fitness=best_fitness, history_best_fitness=history, n_evaluations=n_evaluations)


def grid_search_codec(plc_steps: int = GRID_PLC_STEPS, **codec_fitness_kwargs) -> DeapGaResult:
    """Exhaustive grid over the 5 bitrates x 4 frame sizes x `plc_steps` plc levels x 3 codecs."""
    plc_values = np.linspace(0.0, 1.0, plc_steps)
    best_chromosome: list[float] | None = None
    best_fitness = -np.inf
    n_evaluations = 0

    for bitrate, frame_size, plc, codec_index in product(BITRATE_CHOICES_KBPS, FRAME_SIZE_CHOICES_MS, plc_values, range(len(CODEC_BY_INDEX))):
        chromosome = [float(bitrate), float(frame_size), float(plc), float(codec_index)]
        fitness = codec_fitness(chromosome, **codec_fitness_kwargs)
        n_evaluations += 1
        if fitness > best_fitness:
            best_chromosome, best_fitness = chromosome, fitness

    return DeapGaResult(best_chromosome=best_chromosome, best_fitness=best_fitness, history_best_fitness=[], n_evaluations=n_evaluations)


def cma_es_codec(n_generations: int = 100, pop_size: int | None = None, seed: int | None = None, **codec_fitness_kwargs) -> DeapGaResult:
    """CMA-ES (`cmaes_scratch.run_cma_es`) as a bonus, non-mandated Pb1 solver -- see this module's
    docstring for scope rationale."""

    def objective(chromosome: np.ndarray) -> float:
        return -codec_fitness(chromosome, seed=seed, **codec_fitness_kwargs)

    config = CMAESConfig(n_generations=n_generations, pop_size=pop_size, seed=seed)
    result = run_cma_es(objective, CHROMOSOME_BOUNDS, config)
    return DeapGaResult(
        best_chromosome=list(result.best_x),
        best_fitness=-result.best_fitness,
        history_best_fitness=[-v for v in result.history_best_fitness],
        n_evaluations=result.n_evaluations,
    )


def abc_codec(n_food_sources: int = 25, n_iterations: int = 100, limit: int | None = None, seed: int | None = None, **codec_fitness_kwargs) -> DeapGaResult:
    """Artificial Bee Colony (`abc_scratch.run_abc`) as a bonus, non-mandated Pb1 solver -- see this
    module's docstring for scope rationale. `history_best_fitness` is already per-evaluation
    (`abc_scratch.py`'s convention), so `len(history_best_fitness) == n_evaluations` here too,
    unlike `deap_ga_codec`'s per-generation history."""

    def objective(chromosome: np.ndarray) -> float:
        return -codec_fitness(chromosome, seed=seed, **codec_fitness_kwargs)

    config = ABCConfig(n_food_sources=n_food_sources, n_iterations=n_iterations, limit=limit, seed=seed)
    result = run_abc(objective, CHROMOSOME_BOUNDS, config)
    return DeapGaResult(
        best_chromosome=list(result.best_x),
        best_fitness=-result.best_fitness,
        history_best_fitness=[-v for v in result.history_best_fitness],
        n_evaluations=result.n_evaluations,
    )
