"""Delivery simulation layered on top of entities.py: delay, loss, deferred
retransmission (up to 72h), and an overload mode measuring QoS degradation
(loss rate, mean delay) as offered load grows.

``entities.Smsc.deliver`` is a single, synchronous, all-or-nothing
reachability check (destination known to HLR *and* attached in VLR, no
notion of time). Everything stochastic and time-based lives here instead —
the same separation Module A kept between ``codecs.py`` (pure transforms)
and ``fitness.py`` (seeded randomness, retry policy).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from module_b.entities import Hlr, MessageStatus, Msc, Smsc, Vlr

DEFAULT_MAX_RETRY_WINDOW_S = 72 * 3600.0  # spec: retransmission deferred up to 72h


# --- single-message delivery with delay, transient loss, and retry ----------


@dataclass
class DeliveryOutcome:
    message_id: int
    delivered: bool
    delay_s: float
    attempts: int


def simulate_delivery(
    smsc: Smsc,
    message,
    rng: np.random.Generator,
    *,
    mean_delay_s: float = 2.0,
    loss_probability: float = 0.03,
    retry_backoff_s: float = 30.0,
    backoff_factor: float = 2.0,
    max_retry_window_s: float = DEFAULT_MAX_RETRY_WINDOW_S,
) -> DeliveryOutcome:
    """Attempt delivery of ``message``, retrying with exponential backoff on failure.

    Two independent failure modes, matching real SMS delivery: the
    destination may simply be unreachable (``Msc.route`` false — e.g. phone
    switched off, checked fresh on every attempt since VLR state could
    change between retries in a fuller simulation) and even a reachable
    destination can drop a given attempt with ``loss_probability``
    (transient network loss). Backoff grows geometrically
    (``backoff_factor``) after each failed attempt; the message is declared
    permanently undeliverable once cumulative elapsed time exceeds
    ``max_retry_window_s``.
    """
    elapsed = 0.0
    backoff = retry_backoff_s
    attempts = 0

    while True:
        attempts += 1
        elapsed += rng.exponential(mean_delay_s)
        reachable = smsc.deliver(message)
        transient_loss = rng.random() < loss_probability

        if reachable and not transient_loss:
            message.status = MessageStatus.DELIVERED
            return DeliveryOutcome(message.id, True, elapsed, attempts)

        elapsed += backoff
        if elapsed > max_retry_window_s:
            message.status = MessageStatus.UNDELIVERABLE
            return DeliveryOutcome(message.id, False, elapsed, attempts)

        message.status = MessageStatus.QUEUED
        backoff *= backoff_factor


@dataclass
class BatchDeliveryStats:
    n_messages: int
    delivery_rate: float
    mean_delay_s: float
    mean_attempts: float


def simulate_batch_delivery(
    n_messages: int,
    *,
    reachable_probability: float = 0.95,
    seed: int | None = None,
    **delivery_kwargs,
) -> BatchDeliveryStats:
    """Simulate ``n_messages`` independent deliveries to a mix of reachable/unreachable destinations.

    Each message gets its own destination MSISDN, attached to the VLR with
    probability ``reachable_probability`` — models a population of phones,
    some switched off — sharing one HLR/VLR/MSC/SMSC (fresh per call, no
    cross-message state).
    """
    rng = np.random.default_rng(seed)
    hlr, vlr = Hlr(), Vlr()
    msc = Msc(hlr, vlr)
    smsc = Smsc(msc)

    outcomes: list[DeliveryOutcome] = []
    for i in range(n_messages):
        destination = f"+1555000{i:04d}"
        hlr.register_ms(destination, imsi=f"00101{i:010d}")
        if rng.random() < reachable_probability:
            vlr.attach(destination, "LAC-1")
        message = smsc.send("+15550000000", destination, "hello")
        outcomes.append(simulate_delivery(smsc, message, rng, **delivery_kwargs))

    delivered = [o for o in outcomes if o.delivered]
    return BatchDeliveryStats(
        n_messages=n_messages,
        delivery_rate=len(delivered) / n_messages,
        mean_delay_s=float(np.mean([o.delay_s for o in outcomes])),
        mean_attempts=float(np.mean([o.attempts for o in outcomes])),
    )


# --- overload / QoS-under-load simulation ------------------------------------


@dataclass
class OverloadResult:
    arrival_rate_msgs_per_s: float
    n_arrived: int
    n_delivered: int
    n_dropped: int
    n_still_queued: int
    loss_rate: float
    mean_delay_s: float


def simulate_overload(
    arrival_rate_msgs_per_s: float,
    duration_s: float,
    *,
    throughput_msgs_per_s: float = 100.0,
    queue_capacity: int = 500,
    seed: int | None = None,
) -> OverloadResult:
    """Discrete-time single-server queue: Poisson arrivals, fixed service rate.

    The SMSC is modeled as processing exactly one message per
    ``1/throughput_msgs_per_s`` second time step; arrivals in excess of
    ``queue_capacity`` are dropped immediately (measuring loss under
    sustained overload), matching the spec's "surcharge (centaines de
    SMS/s), dégradation QoS (taux de perte, délai moyen)".
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / throughput_msgs_per_s
    n_steps = int(duration_s / dt)
    expected_arrivals_per_step = arrival_rate_msgs_per_s * dt

    queue: deque[float] = deque()
    n_arrived = 0
    n_dropped = 0
    delays: list[float] = []

    for step in range(n_steps):
        t = step * dt
        for _ in range(int(rng.poisson(expected_arrivals_per_step))):
            n_arrived += 1
            if len(queue) >= queue_capacity:
                n_dropped += 1
            else:
                queue.append(t)
        if queue:
            arrival_t = queue.popleft()
            delays.append(t + dt - arrival_t)

    return OverloadResult(
        arrival_rate_msgs_per_s=arrival_rate_msgs_per_s,
        n_arrived=n_arrived,
        n_delivered=len(delays),
        n_dropped=n_dropped,
        n_still_queued=len(queue),
        loss_rate=(n_dropped / n_arrived) if n_arrived else 0.0,
        mean_delay_s=float(np.mean(delays)) if delays else 0.0,
    )


def sweep_overload(
    arrival_rates_msgs_per_s: list[float],
    duration_s: float,
    *,
    throughput_msgs_per_s: float = 100.0,
    queue_capacity: int = 500,
    seed: int | None = None,
) -> list[OverloadResult]:
    """``simulate_overload`` at each offered load — the QoS-vs-load curve data for the report."""
    return [
        simulate_overload(
            rate,
            duration_s,
            throughput_msgs_per_s=throughput_msgs_per_s,
            queue_capacity=queue_capacity,
            seed=None if seed is None else seed + i,
        )
        for i, rate in enumerate(arrival_rates_msgs_per_s)
    ]
