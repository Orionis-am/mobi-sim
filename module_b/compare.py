"""Sim-vs-real comparison for Module B: simulated delivery stats (network_sim.py)
against real Twilio measurements (twilio_client.py), the running "sim vs
real gap" thread CLAUDE.md asks every module to maintain.

Textual causes for the discrepancy (SS7/carrier routing, GSMA inter-operator
hubs, etc. — docs/SUJET.md §8 Q2) belong in the final report, not hardcoded
here: this module only produces the numbers to discuss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from module_b import network_sim, twilio_client

_TERMINAL_STATUSES = {"delivered", "failed", "undelivered"}


@dataclass
class RealDeliverySample:
    accept_latency_s: float
    delivery_latency_s: float | None
    delivered: bool
    final_status: str


def measure_real_delivery(
    to: str,
    body: str,
    *,
    from_: str | None = None,
    client=None,
    poll_interval_s: float = 2.0,
    poll_timeout_s: float = 60.0,
    sleep=time.sleep,
    now=time.monotonic,
) -> RealDeliverySample:
    """Send one real SMS, timing API-accept latency and delivery-callback latency via polling.

    ``sleep``/``now`` are injectable purely for testability (no real waiting
    in unit tests, same idea as ``client`` being injectable) — real callers
    just use the defaults.
    """
    t0 = now()
    result = twilio_client.send_sms(to, body, from_=from_, client=client)
    accept_latency_s = now() - t0

    delivery_latency_s = None
    final_status = result.status
    deadline = t0 + poll_timeout_s
    while now() < deadline:
        status = twilio_client.get_delivery_status(result.sid, client=client)
        final_status = status.status
        if final_status in _TERMINAL_STATUSES:
            delivery_latency_s = now() - t0
            break
        sleep(poll_interval_s)

    return RealDeliverySample(
        accept_latency_s=accept_latency_s,
        delivery_latency_s=delivery_latency_s,
        delivered=(final_status == "delivered"),
        final_status=final_status,
    )


@dataclass
class RealStats:
    n_samples: int
    delivery_rate: float
    mean_accept_latency_s: float
    mean_delivery_latency_s: float | None


def summarize_real_samples(samples: list[RealDeliverySample]) -> RealStats:
    delivered = [s for s in samples if s.delivered]
    known_latencies = [s.delivery_latency_s for s in samples if s.delivery_latency_s is not None]
    return RealStats(
        n_samples=len(samples),
        delivery_rate=(len(delivered) / len(samples)) if samples else 0.0,
        mean_accept_latency_s=(sum(s.accept_latency_s for s in samples) / len(samples)) if samples else 0.0,
        mean_delivery_latency_s=(sum(known_latencies) / len(known_latencies)) if known_latencies else None,
    )


def build_comparison_table(simulated: network_sim.BatchDeliveryStats, real: RealStats) -> dict:
    """Sim-vs-real table ready for the report: {"simulated": {...}, "real": {...}}."""
    return {
        "simulated": {
            "mean_delay_s": simulated.mean_delay_s,
            "delivery_rate": simulated.delivery_rate,
        },
        "real": {
            "mean_accept_latency_s": real.mean_accept_latency_s,
            "mean_delivery_latency_s": real.mean_delivery_latency_s,
            "delivery_rate": real.delivery_rate,
            "n_samples": real.n_samples,
        },
    }
