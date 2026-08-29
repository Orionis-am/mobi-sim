"""SMS routing/retry-policy fitness for Module F (`routing_fitness` contract).

docs/SUJET.md §3 MOD-B asks for ``routing_fitness(chromosome) → latence
simulée, utilisée par le Module F`` — but unlike Module A's Pb1 (codec
config, wired to DEAP explicitly in §3 MOD-F) there is no defined Pb for SMS
routing among Module F's actual three problems (§3 MOD-F: Pb1 codec/Module
A, Pb2 BTS placement/Module C, Pb3 QoS/Module D). This is still built to the
same contract-stability bar as ``codec_fitness`` per CLAUDE.md ("keep the
signature stable once Module F starts depending on it"), but — unlike
``codec_fitness`` — nothing in Module F concretely calls it for a graded
deliverable; see REPORT.md.

Chromosome (this module's own design, since the spec doesn't define one):
``[retry_backoff_s, backoff_factor, max_retry_window_hours]`` — the
retry/backoff policy ``network_sim.simulate_delivery`` uses when retrying a
failed delivery. Tradeoff a GA should be able to discover: a short backoff
lowers delay for messages that eventually succeed but means more attempts
(more SMSC load) per message; a short retry window frees up resources
faster but abandons reachable-but-slow-to-reattach destinations sooner,
lowering the delivery rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from module_b import network_sim

RETRY_BACKOFF_CHOICES_S = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
MAX_RETRY_WINDOW_CHOICES_HOURS = (1.0, 6.0, 24.0, 48.0, 72.0)
BACKOFF_FACTOR_BOUNDS = (1.0, 3.0)

DEFAULT_WEIGHTS = (1.0, 0.3, 0.1)  # (w1: delivery rate, w2: mean delay, w3: mean attempts) — needs calibration, see REPORT.md
DEFAULT_N_MESSAGES = 200
DEFAULT_REACHABLE_PROBABILITY = 0.9  # assumed nominal fraction of destinations attached at a time; not a chromosome gene
DEFAULT_LOSS_PROBABILITY = 0.03  # assumed nominal transient-loss rate; matches network_sim.simulate_delivery's own default
DEFAULT_MEAN_DELAY_S = 2.0
DEFAULT_DELAY_NORMALIZATION_S = 3600.0  # 1h: keeps a normal few-second delay near 0, penalizes retry pile-ups
DEFAULT_ATTEMPTS_NORMALIZATION = 10.0


@dataclass(frozen=True)
class RoutingConfig:
    retry_backoff_s: float
    backoff_factor: float
    max_retry_window_hours: float


def _nearest(value: float, choices: Sequence[float]) -> float:
    return float(min(choices, key=lambda c: abs(c - value)))


def decode_chromosome(chromosome: Sequence[float]) -> RoutingConfig:
    """Decode a raw ``[retry_backoff_s, backoff_factor, max_retry_window_hours]`` chromosome.

    Snaps the two discrete genes to their nearest valid choice and clips the
    continuous ``backoff_factor`` gene to its bounds — same tolerant
    decoding as Module A's ``fitness.decode_chromosome``, since a GA's
    mutation/crossover can't be trusted to only ever emit exactly-valid
    values.
    """
    if len(chromosome) != 3:
        raise ValueError(f"expected a 3-gene chromosome [retry_backoff_s, backoff_factor, max_retry_window_hours], got {len(chromosome)} genes")
    retry_backoff_raw, backoff_factor_raw, max_retry_window_raw = chromosome
    retry_backoff_s = _nearest(retry_backoff_raw, RETRY_BACKOFF_CHOICES_S)
    backoff_factor = float(np.clip(backoff_factor_raw, *BACKOFF_FACTOR_BOUNDS))
    max_retry_window_hours = _nearest(max_retry_window_raw, MAX_RETRY_WINDOW_CHOICES_HOURS)
    return RoutingConfig(retry_backoff_s, backoff_factor, max_retry_window_hours)


def routing_fitness_components(
    chromosome: Sequence[float],
    n_messages: int = DEFAULT_N_MESSAGES,
    reachable_probability: float = DEFAULT_REACHABLE_PROBABILITY,
    loss_probability: float = DEFAULT_LOSS_PROBABILITY,
    mean_delay_s: float = DEFAULT_MEAN_DELAY_S,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    delay_normalization_s: float = DEFAULT_DELAY_NORMALIZATION_S,
    attempts_normalization: float = DEFAULT_ATTEMPTS_NORMALIZATION,
    min_delivery_rate: float | None = None,
    constraint_penalty_weight: float = 1.0,
    seed: int | None = None,
) -> dict[str, float]:
    """Full breakdown behind ``routing_fitness``'s scalar score, for debugging/plotting."""
    config = decode_chromosome(chromosome)
    stats = network_sim.simulate_batch_delivery(
        n_messages,
        reachable_probability=reachable_probability,
        seed=seed,
        loss_probability=loss_probability,
        mean_delay_s=mean_delay_s,
        retry_backoff_s=config.retry_backoff_s,
        backoff_factor=config.backoff_factor,
        max_retry_window_s=config.max_retry_window_hours * 3600.0,
    )

    w1, w2, w3 = weights
    fitness = (
        w1 * stats.delivery_rate
        - w2 * (stats.mean_delay_s / delay_normalization_s)
        - w3 * (stats.mean_attempts / attempts_normalization)
    )

    constraint_violation = max(0.0, min_delivery_rate - stats.delivery_rate) if min_delivery_rate is not None else 0.0
    fitness -= constraint_penalty_weight * constraint_violation

    return {
        "retry_backoff_s": config.retry_backoff_s,
        "backoff_factor": config.backoff_factor,
        "max_retry_window_hours": config.max_retry_window_hours,
        "delivery_rate": stats.delivery_rate,
        "mean_delay_s": stats.mean_delay_s,
        "mean_attempts": stats.mean_attempts,
        "constraint_violation": constraint_violation,
        "fitness": float(fitness),
    }


def routing_fitness(chromosome: Sequence[float], **kwargs) -> float:
    """Module F's contract: score a ``[retry_backoff_s, backoff_factor, max_retry_window_hours]`` chromosome.

    Single-argument by design (``routing_fitness(chromosome)``), same shape
    as ``module_a.fitness.codec_fitness``. Unlike ``codec_fitness``, there's
    no expensive resource to cache across calls — ``network_sim``'s
    simulation is pure, cheap Python/NumPy, not a network call — so this is
    just a thin wrapper over ``routing_fitness_components``. Same default
    (``seed=None``) as ``codec_fitness``: reproducibility across calls is
    the caller's (Module F's) responsibility, not fixed in here.
    """
    return routing_fitness_components(chromosome, **kwargs)["fitness"]
