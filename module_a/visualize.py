"""Matplotlib visualizations for Module A.

Every function here builds and returns a ``Figure`` from data already
computed elsewhere (PESQ/SNR/LSD from ``metrics.py``, WER from
``whisper_eval.py``, MUSHRA summaries from ``mushra_sim.py``) — no I/O, no
recomputation, no ``plt.show()``. Callers decide whether to display, embed,
or save the figure; ``save_figure`` covers the last case, writing into the
gitignored ``results/`` directory the rest of the project uses for
generated output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

DEFAULT_CODEC_ORDER = ("aac", "gsm", "opus")


def _ordered(codecs: dict, order: tuple[str, ...] = DEFAULT_CODEC_ORDER) -> list[str]:
    known = [c for c in order if c in codecs]
    return known + sorted(c for c in codecs if c not in known)


def plot_mushra_comparison(summary: dict[str, dict[str, dict[str, float]]]) -> Figure:
    """Grouped bar chart of mean MUSHRA score (95% CI error bars) per codec and listener group."""
    codecs = _ordered(summary)
    groups = sorted({group for per_codec in summary.values() for group in per_codec})

    fig, ax = plt.subplots()
    x = np.arange(len(codecs))
    width = 0.8 / max(len(groups), 1)

    for i, group in enumerate(groups):
        means = [summary[codec][group]["mean"] for codec in codecs]
        lower_err = [summary[codec][group]["mean"] - summary[codec][group]["ci_low"] for codec in codecs]
        upper_err = [summary[codec][group]["ci_high"] - summary[codec][group]["mean"] for codec in codecs]
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=[lower_err, upper_err], capsize=3, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in codecs])
    ax.set_ylabel("Score MUSHRA")
    ax.set_ylim(0, 100)
    ax.set_title("MUSHRA comparatif par codec (IC 95% bootstrap)")
    ax.legend(title="Groupe")
    fig.tight_layout()
    return fig


def plot_metric_bar(values: dict[str, float], ylabel: str, title: str) -> Figure:
    """Simple bar chart of one scalar metric per codec (e.g. PESQ-NB or WER)."""
    codecs = _ordered(values)
    x = np.arange(len(codecs))

    fig, ax = plt.subplots()
    ax.bar(x, [values[c] for c in codecs])
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in codecs])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def mushra_grand_mean(summary: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    """Average each codec's MUSHRA mean across listener groups, into one scalar per codec."""
    return {codec: float(np.mean([group["mean"] for group in groups.values()])) for codec, groups in summary.items()}


def build_correlation_table(
    pesq_by_codec: dict[str, float],
    wer_by_codec: dict[str, float],
    mushra_summary: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Merge PESQ, WER, and (group-averaged) MUSHRA into ``{codec: {"pesq", "wer", "mushra"}}``."""
    mushra_by_codec = mushra_grand_mean(mushra_summary)
    return {
        codec: {"pesq": pesq_by_codec[codec], "wer": wer_by_codec[codec], "mushra": mushra_by_codec[codec]}
        for codec in pesq_by_codec
    }


def plot_correlation_matrix(metrics_by_label: dict[str, dict[str, float]]) -> Figure:
    """Heatmap of the Pearson correlation matrix between metrics, computed across ``metrics_by_label``'s entries.

    ``metrics_by_label`` is typically ``{codec: {"pesq": ..., "wer": ..., "mushra": ...}}`` from
    ``build_correlation_table``, but any ``{label: {metric_name: value}}`` mapping works — e.g. many
    Module F chromosome evaluations instead of just 3 codecs, for a statistically sturdier matrix.
    """
    metric_names = sorted({name for entry in metrics_by_label.values() for name in entry})
    data = np.array([[entry[name] for name in metric_names] for entry in metrics_by_label.values()])
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(metric_names)

    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            value = corr[i, j]
            text = "n/a" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Corrélation de Pearson")
    ax.set_title("Corrélation PESQ × MUSHRA × WER")
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, filename: str, output_dir: str | Path = "results") -> Path:
    """Save ``fig`` as a PNG under ``output_dir`` (created if missing); returns the file path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
