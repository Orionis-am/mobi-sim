"""Inject measured/simulated network metrics into the E-model (`docs/SUJET.md` line 233).

`simulate_session` turns (codec, available bandwidth, required bandwidth, measured RTT/jitter)
into a full QoS prediction. Both `dashboard.py` and `fitness.py` build on this.

Bandwidth congestion is modeled as: loss rises with the bandwidth deficit (how far `bandwidth_kbps`
falls short of `required_bandwidth_kbps`), plus queueing delay stacked on top of the measured RTT
— buffers grow before they overflow, so a congested link degrades delay before it starts dropping
packets, which is the real qualitative behavior this is modeling. The two ceiling constants below
(how much loss/delay a *fully* starved link adds) have no numeric basis in the spec — documented
assumptions, same spirit as module_a's DEFAULT_PACKET_LOSS_RATE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module_d import model_e

MAX_CONGESTION_LOSS_PCT = 30.0  # loss added at a 100% bandwidth deficit (fully starved link)
MAX_CONGESTION_DELAY_MS = 150.0  # queueing delay added at a 100% bandwidth deficit
LOSS_NOISE_STD_PCT = 1.0  # seeded Gaussian noise on top of the congestion-driven loss


@dataclass(frozen=True)
class SessionQoS:
    r_factor: float
    mos: float
    delay_ms: float
    loss_pct: float
    codec: str


def _bandwidth_deficit_ratio(bandwidth_kbps: float, required_bandwidth_kbps: float) -> float:
    if required_bandwidth_kbps <= 0:
        return 0.0
    return max(0.0, (required_bandwidth_kbps - bandwidth_kbps) / required_bandwidth_kbps)


def simulate_session(
    codec: str,
    bandwidth_kbps: float,
    required_bandwidth_kbps: float,
    rtt_ms: float,
    jitter_ms: float,
    seed: int | None = None,
) -> SessionQoS:
    """One simulated session's QoS given a codec choice, a bandwidth allocation, and network conditions."""
    rng = np.random.default_rng(seed)
    deficit_ratio = _bandwidth_deficit_ratio(bandwidth_kbps, required_bandwidth_kbps)

    loss_pct = deficit_ratio * MAX_CONGESTION_LOSS_PCT + rng.normal(0.0, LOSS_NOISE_STD_PCT)
    loss_pct = float(np.clip(loss_pct, 0.0, 100.0))

    congestion_delay_ms = deficit_ratio * MAX_CONGESTION_DELAY_MS
    delay_ms = rtt_ms / 2.0 + jitter_ms + congestion_delay_ms

    r = model_e.r_factor(codec, delay_ms, loss_pct)
    mos = model_e.r_to_mos(r)
    return SessionQoS(r_factor=r, mos=mos, delay_ms=delay_ms, loss_pct=loss_pct, codec=codec)
