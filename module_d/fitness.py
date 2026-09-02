"""QoS resource-allocation fitness for Module F's Pb3 (`qos_fitness` contract).

Module F's Pb3 (DE via scipy.optimize.differential_evolution vs. PSO via pyswarm) evolves a
20-gene chromosome ``[codec_u1, bw_u1, ..., codec_u10, bw_u10]`` allocating a codec and a
bandwidth share to each of 10 simultaneous users (VoIP/SMS/streaming profiles, each with a
required minimum QoS) drawn from a fixed shared network capacity (`docs/SUJET.md` line 342-350).

``docs/SUJET.md`` line 241-243 says ``qos_fitness(chromosome) -> MOS moyen`` (a bare scalar) while
Pb3's objectives are phrased as a pair ``(-MOS moyen, bande totale)`` — resolved the same way
module_a resolved an analogous mismatch (Pb1's WER term): a single weighted-sum scalar with a
hard-constraint penalty for exceeding total capacity, configurable weights. The contract stays a
plain ``float`` (not a tuple), since DE/PSO are single-objective solvers, unlike Pb2's NSGA-II
(``module_c.fitness.bts_coverage_fitness``, which does return a tuple for that reason).

No real STUN call happens inside the fitness loop: DE/PSO can evaluate this thousands of times,
and opening a UDP socket on every evaluation is both slow and non-reproducible — same reasoning as
module_a's WER proxy. ``rtt_ms``/``jitter_ms`` default to fixed, documented baseline constants
instead of a cached real measurement (there is nothing expensive to cache here, unlike module_a's
TTS reference signal or module_c's BTS terrain — simpler than sketched in the initial plan).
Callers who want the measured value (e.g. one validation pass on the DE/PSO's best individual) can
just pass real ``rtt_ms``/``jitter_ms`` from ``stun_probe.measure_rtt_jitter()`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from module_d import session_sim

CODEC_BY_INDEX = {0: "aac", 1: "gsm", 2: "opus"}
MAX_BANDWIDTH_KBPS = 128.0  # generous per-user cap, above any single profile's requirement

DEFAULT_WEIGHTS = (0.8, 0.2)  # (w_mos, w_bandwidth) — needs calibration, see REPORT.md
DEFAULT_BASELINE_RTT_MS = 40.0  # plausible mobile-network baseline; documented assumption, see REPORT.md
DEFAULT_BASELINE_JITTER_MS = 5.0  # plausible baseline jitter; documented assumption, see REPORT.md


@dataclass(frozen=True)
class UserProfile:
    name: str
    required_bandwidth_kbps: float
    min_mos: float


# 3 VoIP + 3 SMS + 4 streaming = 10 users; bandwidth/MOS-min values are documented assumptions,
# the spec names the profile types but gives no numeric requirements — see REPORT.md.
DEFAULT_USER_PROFILES: tuple[UserProfile, ...] = (
    UserProfile("voip", 32.0, 3.5),
    UserProfile("voip", 32.0, 3.5),
    UserProfile("voip", 32.0, 3.5),
    UserProfile("sms", 1.0, 3.0),
    UserProfile("sms", 1.0, 3.0),
    UserProfile("sms", 1.0, 3.0),
    UserProfile("streaming", 64.0, 3.8),
    UserProfile("streaming", 64.0, 3.8),
    UserProfile("streaming", 64.0, 3.8),
    UserProfile("streaming", 64.0, 3.8),
)
# Capacity fixed below full demand (70%) so the allocation problem has a genuine tradeoff to solve.
DEFAULT_TOTAL_CAPACITY_KBPS = 0.7 * sum(p.required_bandwidth_kbps for p in DEFAULT_USER_PROFILES)


@dataclass(frozen=True)
class UserAllocation:
    codec: str
    bandwidth_kbps: float
    profile: UserProfile


def decode_chromosome(chromosome: Sequence[float], profiles: tuple[UserProfile, ...] = DEFAULT_USER_PROFILES) -> list[UserAllocation]:
    """Decode a flat ``[codec, bandwidth] * len(profiles)`` chromosome, one pair per user profile.

    The codec gene snaps to the nearest of {0,1,2} via modulo (a DE/PSO individual isn't
    guaranteed to land exactly on an integer); bandwidth clips to ``[0, MAX_BANDWIDTH_KBPS]``.
    """
    expected_len = 2 * len(profiles)
    if len(chromosome) != expected_len:
        raise ValueError(f"Expected a {expected_len}-gene chromosome (codec, bandwidth per user), got {len(chromosome)}")

    allocations = []
    for i, profile in enumerate(profiles):
        codec_raw, bandwidth_raw = chromosome[2 * i], chromosome[2 * i + 1]
        codec_index = int(round(codec_raw)) % len(CODEC_BY_INDEX)
        bandwidth = float(np.clip(bandwidth_raw, 0.0, MAX_BANDWIDTH_KBPS))
        allocations.append(UserAllocation(CODEC_BY_INDEX[codec_index], bandwidth, profile))
    return allocations


def qos_fitness_components(
    chromosome: Sequence[float],
    profiles: tuple[UserProfile, ...] = DEFAULT_USER_PROFILES,
    total_capacity_kbps: float = DEFAULT_TOTAL_CAPACITY_KBPS,
    rtt_ms: float = DEFAULT_BASELINE_RTT_MS,
    jitter_ms: float = DEFAULT_BASELINE_JITTER_MS,
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    constraint_penalty_weight: float = 1.0,
    min_mos_penalty_weight: float = 1.0,
    seed: int | None = None,
) -> dict[str, float | list[float]]:
    """Full breakdown behind ``qos_fitness``'s scalar score, for debugging/plotting."""
    allocations = decode_chromosome(chromosome, profiles)

    sessions = [
        session_sim.simulate_session(
            alloc.codec, alloc.bandwidth_kbps, alloc.profile.required_bandwidth_kbps, rtt_ms, jitter_ms, seed=None if seed is None else seed + i
        )
        for i, alloc in enumerate(allocations)
    ]

    mean_mos = float(np.mean([s.mos for s in sessions]))
    total_bandwidth_kbps = float(sum(a.bandwidth_kbps for a in allocations))
    capacity_violation_kbps = max(0.0, total_bandwidth_kbps - total_capacity_kbps)
    min_mos_violation = float(sum(max(0.0, alloc.profile.min_mos - s.mos) for alloc, s in zip(allocations, sessions)))

    w_mos, w_bandwidth = weights
    fitness = w_mos * mean_mos - w_bandwidth * (total_bandwidth_kbps / total_capacity_kbps)
    fitness -= constraint_penalty_weight * capacity_violation_kbps
    fitness -= min_mos_penalty_weight * min_mos_violation

    return {
        "mean_mos": mean_mos,
        "total_bandwidth_kbps": total_bandwidth_kbps,
        "capacity_violation_kbps": capacity_violation_kbps,
        "min_mos_violation": min_mos_violation,
        "per_user_mos": [s.mos for s in sessions],
        "fitness": float(fitness),
    }


def qos_fitness(chromosome: Sequence[float], **kwargs) -> float:
    """Module F's Pb3 fitness contract: score a 20-gene ``[codec, bandwidth] * 10`` chromosome."""
    return qos_fitness_components(chromosome, **kwargs)["fitness"]
