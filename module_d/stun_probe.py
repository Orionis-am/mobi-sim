"""Real STUN RTT/jitter measurement over UDP via stdlib `socket` (`docs/SUJET.md` line 234-239).

Sends RFC 5389 STUN Binding Request packets to a public STUN server (`stun.l.google.com:19302` by
default) and times the round trip with `time.perf_counter()`. No third-party STUN client library
is used — the spec explicitly names stdlib `socket`.

"Loss rate" is the fraction of the `n_measurements` real requests that timed out or failed to
parse as a valid response — not an injected synthetic value. A genuine UDP round trip to a real
server already produces real loss under real conditions, so there was no need to invent an extra
probability parameter on top of it; a fully controllable, swept loss rate for the required
MOS=f(loss) plot is instead handled directly by `correlation.py`/`session_sim.py`, which call
`model_e` with an arbitrary chosen loss_pct, independent of what this module happens to measure.
"""

from __future__ import annotations

import os
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np

MAGIC_COOKIE = 0x2112A442
BINDING_REQUEST = 0x0001
BINDING_SUCCESS_RESPONSE = 0x0101
DEFAULT_HOST = "stun.l.google.com"
DEFAULT_PORT = 19302
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_N_MEASUREMENTS = 20


def build_binding_request() -> bytes:
    """RFC 5389 STUN Binding Request: type=0x0001, length=0, magic cookie, random 96-bit txn id."""
    txn_id = os.urandom(12)
    return struct.pack(">HHI12s", BINDING_REQUEST, 0, MAGIC_COOKIE, txn_id)


def transaction_id(packet: bytes) -> bytes:
    return packet[8:20]


def parse_binding_response(data: bytes, expected_txn_id: bytes) -> bool:
    """True iff `data` is a well-formed Binding Success Response matching `expected_txn_id`."""
    if len(data) < 20:
        return False
    msg_type, _, cookie, txn_id = struct.unpack(">HHI12s", data[:20])
    return msg_type == BINDING_SUCCESS_RESPONSE and cookie == MAGIC_COOKIE and txn_id == expected_txn_id


def measure_one_rtt(sock, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout_s: float = DEFAULT_TIMEOUT_S) -> float | None:
    """One real STUN round trip; RTT in ms, or None on timeout/parse failure."""
    request = build_binding_request()
    sock.settimeout(timeout_s)
    start = time.perf_counter()
    try:
        sock.sendto(request, (host, port))
        data, _ = sock.recvfrom(2048)
    except OSError:
        return None
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if not parse_binding_response(data, transaction_id(request)):
        return None
    return elapsed_ms


@dataclass(frozen=True)
class StunStats:
    rtt_samples_ms: list[float]
    rtt_mean_ms: float
    jitter_ms: float
    loss_rate: float


def measure_rtt_jitter(
    n_measurements: int = DEFAULT_N_MEASUREMENTS,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    sock=None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> StunStats:
    """RTT/jitter/loss over `n_measurements` real STUN round trips.

    jitter_ms is the stdev of inter-packet RTT deltas ("gigue = variance inter-paquets").
    `sock` injectable for testing; defaults to a real UDP socket, opened and closed here.
    """
    owns_socket = sock is None
    sock = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    samples: list[float] = []
    try:
        for _ in range(n_measurements):
            rtt = measure_one_rtt(sock, host, port, timeout_s)
            if rtt is not None:
                samples.append(rtt)
    finally:
        if owns_socket:
            sock.close()

    n_lost = n_measurements - len(samples)
    loss_rate = n_lost / n_measurements if n_measurements else 0.0
    if not samples:
        return StunStats(rtt_samples_ms=[], rtt_mean_ms=float("nan"), jitter_ms=float("nan"), loss_rate=loss_rate)

    rtt_mean = float(np.mean(samples))
    jitter = float(np.std(np.diff(samples))) if len(samples) >= 2 else 0.0
    return StunStats(rtt_samples_ms=samples, rtt_mean_ms=rtt_mean, jitter_ms=jitter, loss_rate=loss_rate)
