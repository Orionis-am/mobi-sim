"""Matplotlib/Folium visualizations for Module F: convergence curves, hyperparameter heatmaps,
Pareto fronts, hypervolume curves, and the Pb2 BTS-placement map (`docs/SUJET.md` lines 317-319,
327-328, 339-340, 356-357).

Same discipline as `module_a/visualize.py`/`module_c/map_viz.py`: every function takes data
already computed elsewhere and returns a `Figure`/`Map` — no I/O, no recomputation, no
`plt.show()`. `save_figure`/`save_map_html` are reused by identity rather than reimplemented,
mirroring `module_d/correlation.py`'s `plot_correlation_matrix = module_a.visualize.
plot_correlation_matrix` precedent.
"""

from __future__ import annotations

import folium
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from module_a import visualize as a_visualize
from module_c import map_viz as c_map_viz
from module_c import terrain_sim
from module_c.terrain_sim import Terrain

save_figure = a_visualize.save_figure
save_map_html = c_map_viz.save_map_html


def plot_convergence(
    histories: dict[str, list[float]],
    xlabel: str = "Génération",
    ylabel: str = "Fitness",
    title: str = "",
    x_values: dict[str, list[float]] | None = None,
) -> Figure:
    """One line per algorithm/run label, e.g. `{"AG": [...], "random_search": [...]}`.

    Plots `range(len(values))` by default — correct only when every history shares the same
    per-point unit. That's true for `plot_hypervolume_curve` (NSGA-II/MOEA-D both report one point
    per pymoo generation) but **not** for `benchmark.run_benchmark_suite`'s three series:
    `random_search`/`hill_climbing` log one point per fitness evaluation, while `ag_scratch.run_ga`
    logs one point per generation (each already `pop_size` evaluations) — plotting both against a
    raw `range(len(values))` index understates the GA's x-axis by a factor of `pop_size` and makes
    it look like it stopped early despite an equal evaluation budget. Pass `x_values` (one
    coordinate list per label, e.g. `{"AG": [i * pop_size for i in range(n_generations)], ...}`) to
    plot every series on a shared, correctly-scaled axis instead.
    """
    fig, ax = plt.subplots()
    for label, values in histories.items():
        xs = x_values[label] if x_values is not None else range(len(values))
        ax.plot(xs, values, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hyperparam_heatmap(grid: np.ndarray, row_labels: list, col_labels: list, row_name: str, col_name: str, title: str) -> Figure:
    """2D heatmap of a hyperparameter sweep's mean final score (`grid.shape == (len(row_labels), len(col_labels))`).

    `benchmark.hyperparameter_sweep` returns a 3D array (population x crossover x mutation) — the
    spec's 3 hyperparameters can't collapse into one 2D heatmap, so callers render one of these
    per population size, slicing the 3D array down to 2D first (documented resolution, see
    REPORT.md's `benchmark.py` section).
    """
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pareto_front_2d(fronts: dict[str, np.ndarray], xlabel: str, ylabel: str, title: str) -> Figure:
    """Scatter one Pareto front per algorithm/N label, each an (n_points, 2) array of (f1, f2)."""
    fig, ax = plt.subplots()
    for label, points in fronts.items():
        points = np.asarray(points)
        ax.scatter(points[:, 0], points[:, 1], label=label, s=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_hypervolume_curve(histories: dict[str, list[float]], title: str = "Hypervolume par génération") -> Figure:
    """One hypervolume-vs-generation line per algorithm (e.g. NSGA-II vs. MOEA/D)."""
    return plot_convergence(histories, xlabel="Génération", ylabel="Hypervolume", title=title)


def build_bts_placement_map(terrain: Terrain, new_positions_xy: np.ndarray, zoom_start: int = c_map_viz.DEFAULT_ZOOM_START) -> folium.Map:
    """Existing real BTS (blue) + Pb2's newly placed BTS (red), converted back to lon/lat.

    New, not adapted from `module_c.map_viz.build_position_map` — that function is shaped for one
    estimated position + one optional true position + POIs, not N new BTS placed alongside M
    existing real ones.
    """
    existing_xy = terrain.positions_xy
    existing_lon, existing_lat = terrain_sim.xy_to_lonlat(existing_xy[:, 0], existing_xy[:, 1], terrain.lon0, terrain.lat0)
    new_positions_xy = np.asarray(new_positions_xy)
    new_lon, new_lat = terrain_sim.xy_to_lonlat(new_positions_xy[:, 0], new_positions_xy[:, 1], terrain.lon0, terrain.lat0)

    m = folium.Map(location=[terrain.lat0, terrain.lon0], zoom_start=zoom_start)
    for lat, lon in zip(np.atleast_1d(existing_lat), np.atleast_1d(existing_lon)):
        folium.Marker(location=[float(lat), float(lon)], popup="BTS existante", icon=folium.Icon(color="blue", icon="signal", prefix="fa")).add_to(m)
    for lat, lon in zip(np.atleast_1d(new_lat), np.atleast_1d(new_lon)):
        folium.Marker(location=[float(lat), float(lon)], popup="Nouvelle BTS (Pb2)", icon=folium.Icon(color="red", icon="tower-broadcast", prefix="fa")).add_to(m)
    return m
