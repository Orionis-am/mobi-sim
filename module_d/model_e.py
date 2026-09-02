"""ITU-T G.107 E-model: R-factor and MOS prediction (`docs/SUJET.md` line 227-232).

The spec gives the R-factor skeleton (``R = R0 - Is - Id - Ie + A``, ``R0=93.2``, ``A=10``, and a
per-codec ``Ie`` table) but not the ``Id``/``Ie`` formulas or the ``R -> MOS`` conversion — those
are filled in here with the standard, published ITU-T G.107 formulas (not invented):

- ``Id`` (delay impairment) is the standard piecewise-linear function of one-way delay.
- Packet loss enters through ``Ie,eff`` (the standard loss-adjusted equipment impairment factor),
  since the spec's flat ``Ie`` table has no loss term and ``Is`` is documented as a
  codec/echo-specific term, not a loss-specific one.
- ``Is`` (synchronous impairment — "codage, echo simule" per the spec) defaults to 0: no echo path
  is modeled anywhere in this codebase, so there is nothing else to attribute to it by default.
  Left injectable for a future echo model.
- ``R -> MOS`` is the standard ITU-T G.107 cubic mapping.
"""

from __future__ import annotations

import numpy as np

R0 = 93.2
ACCESS_ADVANTAGE_A = 10.0
IE_TABLE = {"gsm": 20.0, "aac": 25.0, "opus": 7.0}
DEFAULT_BPL = 10.0  # packet-loss robustness factor; no per-codec basis given, single constant used uniformly


def delay_impairment(delay_ms: float) -> float:
    """Standard ITU-T Id formula: 0 below ~177 ms, then a steeper linear ramp."""
    d = max(0.0, delay_ms)
    step = 1.0 if d > 177.3 else 0.0
    return 0.024 * d + 0.11 * (d - 177.3) * step


def effective_equipment_impairment(ie_base: float, loss_pct: float, bpl: float = DEFAULT_BPL) -> float:
    """Standard packet-loss-adjusted equipment impairment factor Ie,eff."""
    loss_pct = max(0.0, loss_pct)
    return ie_base + (95.0 - ie_base) * loss_pct / (loss_pct / bpl + 2.0)


def r_factor(codec: str, delay_ms: float, loss_pct: float, is_impairment: float = 0.0, bpl: float = DEFAULT_BPL) -> float:
    """R = R0 - Is - Id - Ie,eff + A, clamped to [0, 100]."""
    codec = codec.lower()
    if codec not in IE_TABLE:
        raise ValueError(f"Unknown codec {codec!r}, expected one of {sorted(IE_TABLE)}")

    id_ = delay_impairment(delay_ms)
    ie_eff = effective_equipment_impairment(IE_TABLE[codec], loss_pct, bpl)
    r = R0 - is_impairment - id_ - ie_eff + ACCESS_ADVANTAGE_A
    return float(np.clip(r, 0.0, 100.0))


def r_to_mos(r: float) -> float:
    """Standard ITU-T G.107 R -> MOS cubic mapping."""
    if r <= 0.0:
        return 1.0
    if r >= 100.0:
        return 4.5
    return 1.0 + 0.035 * r + r * (r - 60.0) * (100.0 - r) * 7e-6


def mos_from_conditions(codec: str, delay_ms: float, loss_pct: float, is_impairment: float = 0.0, bpl: float = DEFAULT_BPL) -> float:
    """End-to-end: conditions -> R -> MOS."""
    return r_to_mos(r_factor(codec, delay_ms, loss_pct, is_impairment, bpl))
