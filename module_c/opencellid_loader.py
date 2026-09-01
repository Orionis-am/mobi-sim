"""OpenCelliD CSV loading for Module C — real BTS coordinates, offline (no API quota).

`docs/SUJET.md` explicitly recommends the offline dump over OpenCelliD's live API (1000 req/day)
to sidestep quota entirely; the repo ships a France (MCC=208) dump at ``docs/208.csv``. That file
has no header row and, per OpenCelliD's public schema, lists **lon before lat** — a well-known
footgun worth naming explicitly in ``COLUMNS`` rather than trusting callers to remember.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "docs" / "208.csv"

COLUMNS = [
    "radio", "mcc", "net", "area", "cell", "unit",
    "lon", "lat", "range", "samples", "changeable", "created", "updated", "average_signal",
]

# Rodez / Aveyron (departement 12): lon_min, lon_max, lat_min, lat_max.
# 6,815 real BTS rows fall in this box (verified against docs/208.csv), comfortably over the
# spec's 500-2000 BTS target for the simulation terrain.
AVEYRON_BBOX = (1.5, 3.5, 43.8, 44.9)

_csv_cache: dict[tuple[str, int | None], pd.DataFrame] = {}


def load_opencellid_csv(csv_path: str | Path = DEFAULT_CSV_PATH, mcc: int | None = 208, use_cache: bool = True) -> pd.DataFrame:
    """Load the OpenCelliD dump into a DataFrame, optionally filtered to one MCC.

    Cached in memory per ``(csv_path, mcc)`` so repeated calls (e.g. across several
    ``sample_bts`` calls in the same process) don't re-parse the multi-hundred-thousand-row CSV
    each time — same idea as ``module_a/fitness.py``'s ``_reference_cache``, not a new on-disk
    cache format.
    """
    key = (str(csv_path), mcc)
    if use_cache and key in _csv_cache:
        return _csv_cache[key].copy()

    df = pd.read_csv(csv_path, header=None, names=COLUMNS)
    if mcc is not None:
        df = df[df["mcc"] == mcc].reset_index(drop=True)

    if use_cache:
        _csv_cache[key] = df
    return df.copy()


def sample_bts(
    df: pd.DataFrame,
    bbox: tuple[float, float, float, float] = AVEYRON_BBOX,
    n: int = 1000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Filter ``df`` to a ``(lon_min, lon_max, lat_min, lat_max)`` box and sample up to ``n`` rows.

    Samples without replacement; if fewer than ``n`` rows fall in the box, returns all of them
    rather than raising — a demo terrain built from what's actually there beats a startup crash.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    in_box = df[(df["lon"] >= lon_min) & (df["lon"] <= lon_max) & (df["lat"] >= lat_min) & (df["lat"] <= lat_max)]
    return in_box.sample(n=min(n, len(in_box)), random_state=seed).reset_index(drop=True)
