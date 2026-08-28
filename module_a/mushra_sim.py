"""MUSHRA subjective panel simulation for Module A.

The reference report only publishes each codec's *mean* MUSHRA score per
listener group (English speakers, Native speakers) — not the raw
per-listener ratings. To get something a confidence interval and a
correlation matrix can be computed from, we simulate a plausible panel: each
listener's rating is drawn from a normal distribution centered on the
reported mean, clipped to the [0, 100] MUSHRA scale (ratings are bounded,
unlike the normal distribution). The 95% CI itself is computed by bootstrap
resampling of those simulated ratings rather than assumed from normality,
per the spec.
"""

from __future__ import annotations

import numpy as np

MUSHRA_SCALE = (0.0, 100.0)

# Reported mean scores (English speakers, Native speakers) per codec, from
# the reference experimental report.
REPORTED_MEANS = {
    "opus": {"english": 57.7, "native": 61.4},
    "gsm": {"english": 48.0, "native": 51.4},
    "aac": {"english": 34.9, "native": 36.2},
}

DEFAULT_STD = 15.0
DEFAULT_N_LISTENERS = 20


def simulate_panel_ratings(mean: float, std: float = DEFAULT_STD, n_listeners: int = DEFAULT_N_LISTENERS, seed: int | None = None) -> np.ndarray:
    """Simulate one group's per-listener MUSHRA ratings for one codec, clipped to [0, 100]."""
    rng = np.random.default_rng(seed)
    ratings = rng.normal(mean, std, size=n_listeners)
    return np.clip(ratings, *MUSHRA_SCALE)


def simulate_mushra_panel(
    means: dict[str, dict[str, float]] = REPORTED_MEANS,
    std: float = DEFAULT_STD,
    n_listeners: int = DEFAULT_N_LISTENERS,
    seed: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Simulate ratings for every (codec, group) pair in ``means``.

    Returns ``{codec: {group: ratings_array}}``. Each (codec, group) draws
    from its own RNG stream (offset from ``seed``) so results are
    reproducible without sharing sample noise across codecs/groups.
    """
    ratings: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for codec, groups in means.items():
        ratings[codec] = {}
        for group, mean in groups.items():
            sub_seed = None if seed is None else seed + offset
            ratings[codec][group] = simulate_panel_ratings(mean, std, n_listeners, seed=sub_seed)
            offset += 1
    return ratings


def bootstrap_ci(ratings: np.ndarray, n_bootstrap: int = 2000, ci: float = 0.95, seed: int | None = None) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of ``ratings``."""
    rng = np.random.default_rng(seed)
    n = len(ratings)
    boot_means = rng.choice(ratings, size=(n_bootstrap, n), replace=True).mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return lower, upper


def summarize_mushra(
    ratings_by_codec_group: dict[str, dict[str, np.ndarray]],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean + bootstrap CI for every (codec, group) in ``ratings_by_codec_group``."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    offset = 0
    for codec, groups in ratings_by_codec_group.items():
        summary[codec] = {}
        for group, ratings in groups.items():
            sub_seed = None if seed is None else seed + offset
            lower, upper = bootstrap_ci(ratings, n_bootstrap, ci, seed=sub_seed)
            summary[codec][group] = {"mean": float(np.mean(ratings)), "ci_low": lower, "ci_high": upper}
            offset += 1
    return summary
