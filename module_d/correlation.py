"""Global PESQ/MUSHRA/WER/MOS correlation and MOS=f(delay,loss) curves (`docs/SUJET.md` line 231-232, 241-243).

No I/O, no recomputation, no ``plt.show()`` — same discipline as ``module_a/visualize.py``, whose
generic ``plot_correlation_matrix`` (works over any ``{label: {metric: value}}`` mapping, not just
its own PESQ/WER/MUSHRA output) is reused here as-is for the heatmap rather than reimplemented; its
title still reads "PESQ x MUSHRA x WER" even once a "mos" key is folded in — a known cosmetic quirk
of the reused function, not worth a module_a edit for.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from module_a import visualize as a_visualize
from module_d import model_e

DEFAULT_CODECS = ("aac", "gsm", "opus")
R_THRESHOLD_ACCEPTABLE = 60.0
R_THRESHOLD_GOOD = 80.0

plot_correlation_matrix = a_visualize.plot_correlation_matrix


def build_correlation_table(
    pesq_by_codec: dict[str, float],
    wer_by_codec: dict[str, float],
    mushra_summary: dict[str, dict[str, dict[str, float]]],
    mos_by_codec: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Merge module_a's real PESQ/WER/MUSHRA with module_d's predicted MOS, per codec."""
    base = a_visualize.build_correlation_table(pesq_by_codec, wer_by_codec, mushra_summary)
    return {codec: {**metrics, "mos": mos_by_codec[codec]} for codec, metrics in base.items()}


def _annotate_r_thresholds(ax) -> None:
    mos_acceptable = model_e.r_to_mos(R_THRESHOLD_ACCEPTABLE)
    mos_good = model_e.r_to_mos(R_THRESHOLD_GOOD)
    ax.axhline(mos_acceptable, color="orange", linestyle="--", linewidth=1, label=f"R={R_THRESHOLD_ACCEPTABLE:.0f} (acceptable)")
    ax.axhline(mos_good, color="green", linestyle="--", linewidth=1, label=f"R={R_THRESHOLD_GOOD:.0f} (bon)")


def plot_mos_vs_delay(codecs: tuple[str, ...] = DEFAULT_CODECS, delay_range_ms: np.ndarray | None = None, loss_pct: float = 0.0) -> Figure:
    """MOS = f(delay), one line per codec, with R<60/60-80/>80 thresholds annotated (spec line 231-232)."""
    delay_range_ms = delay_range_ms if delay_range_ms is not None else np.linspace(0, 300, 50)
    fig, ax = plt.subplots()
    for codec in codecs:
        mos_values = [model_e.mos_from_conditions(codec, float(d), loss_pct) for d in delay_range_ms]
        ax.plot(delay_range_ms, mos_values, label=codec.upper())
    _annotate_r_thresholds(ax)
    ax.set_xlabel("Delai (ms)")
    ax.set_ylabel("MOS")
    ax.set_ylim(1.0, 4.5)
    ax.set_title(f"MOS = f(delai) a perte={loss_pct:.0f}%")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_mos_vs_loss(codecs: tuple[str, ...] = DEFAULT_CODECS, loss_range_pct: np.ndarray | None = None, delay_ms: float = 0.0) -> Figure:
    """MOS = f(loss rate), one line per codec, with R<60/60-80/>80 thresholds annotated (spec line 231-232)."""
    loss_range_pct = loss_range_pct if loss_range_pct is not None else np.linspace(0, 20, 50)
    fig, ax = plt.subplots()
    for codec in codecs:
        mos_values = [model_e.mos_from_conditions(codec, delay_ms, float(loss)) for loss in loss_range_pct]
        ax.plot(loss_range_pct, mos_values, label=codec.upper())
    _annotate_r_thresholds(ax)
    ax.set_xlabel("Perte de paquets (%)")
    ax.set_ylabel("MOS")
    ax.set_ylim(1.0, 4.5)
    ax.set_title(f"MOS = f(perte) a delai={delay_ms:.0f} ms")
    ax.legend()
    fig.tight_layout()
    return fig
