"""Unit tests for Module F (ag_scratch, benchmark, pb1_codec, pb2_bts, pb3_qos, compare,
visualize).

Built incrementally alongside each module_f file, mirroring test_module_d.py's conventions: one
Test<Function> class per function under test, seed-reproducibility checks for anything stochastic,
hand-computed reference values where the formula is simple enough to check by hand. Module F has
no external API of its own (`docs/SUJET.md` line 309-310: "toutes gratuites, pip, aucune cle API")
so there is no manual_*_check.py counterpart here, unlike modules A-D.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: these tests must never pop up a GUI window

import folium
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from module_a import visualize as a_visualize
from module_c import map_viz as c_map_viz
from module_c.terrain_sim import Terrain
from module_f import ag_scratch, visualize

# --- ag_scratch ----------------------------------------------------------------


class TestTournamentSelection:
    def test_best_individual_selected_more_than_chance(self):
        rng = np.random.default_rng(0)
        population = np.arange(20).reshape(20, 1).astype(float)
        fitnesses = population[:, 0].copy()  # index 0 is the global best (minimize)

        wins = sum(1 for _ in range(500) if ag_scratch.tournament_selection(population, fitnesses, k=3, rng=rng)[0] == 0.0)
        # uniform random picks would land on individual 0 ~1/20 of the time; a 3-way tournament
        # among 20 should do meaningfully better than that baseline.
        assert wins / 500 > 1 / 20

    def test_returns_a_copy(self):
        rng = np.random.default_rng(0)
        population = np.array([[1.0], [2.0], [3.0]])
        fitnesses = np.array([3.0, 2.0, 1.0])
        picked = ag_scratch.tournament_selection(population, fitnesses, k=3, rng=rng)
        picked[0] = 999.0
        assert population[2, 0] == 3.0


class TestSbxCrossover:
    def test_children_within_bounds(self):
        rng = np.random.default_rng(1)
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        p1, p2 = np.array([-4.9, 4.9]), np.array([4.9, -4.9])
        for _ in range(50):
            c1, c2 = ag_scratch.sbx_crossover(p1, p2, eta=20.0, bounds=bounds, rng=rng)
            assert np.all(c1 >= -5.0) and np.all(c1 <= 5.0)
            assert np.all(c2 >= -5.0) and np.all(c2 <= 5.0)

    def test_large_eta_keeps_children_close_to_parents(self):
        # beta -> 1 as eta -> infinity, so child1 -> parent1 and child2 -> parent2 (SBX's
        # exploitation limit: near-identical offspring rather than blending toward the midpoint).
        rng = np.random.default_rng(2)
        bounds = [(-5.0, 5.0)]
        p1, p2 = np.array([0.0]), np.array([2.0])
        c1, c2 = ag_scratch.sbx_crossover(p1, p2, eta=1000.0, bounds=bounds, rng=rng)
        assert c1[0] == pytest.approx(0.0, abs=0.2)
        assert c2[0] == pytest.approx(2.0, abs=0.2)


class TestGaussianMutation:
    def test_zero_sigma_is_identity(self):
        rng = np.random.default_rng(0)
        individual = np.array([1.0, -2.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        mutated = ag_scratch.gaussian_mutation(individual, sigma=0.0, bounds=bounds, rng=rng)
        assert np.array_equal(mutated, individual)

    def test_stays_within_bounds(self):
        rng = np.random.default_rng(0)
        bounds = [(-1.0, 1.0)]
        for _ in range(50):
            mutated = ag_scratch.gaussian_mutation(np.array([0.99]), sigma=5.0, bounds=bounds, rng=rng)
            assert -1.0 <= mutated[0] <= 1.0


class TestAdaptiveSigma:
    def test_starts_at_init_ends_at_final(self):
        assert ag_scratch._adaptive_sigma(0, 10, 0.5, 0.01) == pytest.approx(0.5)
        assert ag_scratch._adaptive_sigma(9, 10, 0.5, 0.01) == pytest.approx(0.01)

    def test_single_generation_returns_final(self):
        assert ag_scratch._adaptive_sigma(0, 1, 0.5, 0.01) == 0.01


class TestRunGa:
    def test_history_best_is_non_increasing(self):
        config = ag_scratch.GAConfig(pop_size=20, n_generations=30, seed=0)
        bounds = [(-5.12, 5.12)] * 2
        result = ag_scratch.run_ga(_rastrigin, bounds, config)
        history = result.history_best_fitness
        assert all(a >= b for a, b in zip(history, history[1:]))

    def test_converges_on_2d_rastrigin(self):
        config = ag_scratch.GAConfig(pop_size=50, n_generations=50, seed=0)
        bounds = [(-5.12, 5.12)] * 2
        result = ag_scratch.run_ga(_rastrigin, bounds, config)
        assert result.best_fitness < 5.0

    def test_n_evaluations_matches_formula(self):
        config = ag_scratch.GAConfig(pop_size=20, n_generations=15, seed=0)
        bounds = [(-5.12, 5.12)] * 2
        result = ag_scratch.run_ga(_rastrigin, bounds, config)
        assert result.n_evaluations == 20 * 15

    def test_history_length_matches_n_generations(self):
        config = ag_scratch.GAConfig(pop_size=10, n_generations=15, seed=0)
        bounds = [(-5.12, 5.12)] * 2
        result = ag_scratch.run_ga(_rastrigin, bounds, config)
        assert len(result.history_best_fitness) == 15
        assert len(result.history_mean_fitness) == 15

    def test_reproducible_with_same_seed(self):
        config = ag_scratch.GAConfig(pop_size=15, n_generations=10, seed=42)
        bounds = [(-5.12, 5.12)] * 2
        r1 = ag_scratch.run_ga(_rastrigin, bounds, config)
        r2 = ag_scratch.run_ga(_rastrigin, bounds, config)
        assert r1.best_fitness == r2.best_fitness
        assert np.array_equal(r1.best_x, r2.best_x)


def _rastrigin(x: np.ndarray) -> float:
    a = 10.0
    return float(a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x)))


# --- visualize -------------------------------------------------------------


def _markers(m: folium.Map) -> list:
    return [c for c in m._children.values() if isinstance(c, folium.Marker)]


def _placement_terrain() -> Terrain:
    # 4 real BTS at the corners of a 10km square, centered on lon0=0, lat0=0.
    df = pd.DataFrame({"x": [0.0, 10_000.0, 0.0, 10_000.0], "y": [0.0, 0.0, 10_000.0, 10_000.0]})
    return Terrain(bts=df, lon0=0.0, lat0=0.0)


class TestSaveFigureReuse:
    def test_is_module_a_save_figure(self):
        assert visualize.save_figure is a_visualize.save_figure


class TestSaveMapHtmlReuse:
    def test_is_module_c_save_map_html(self):
        assert visualize.save_map_html is c_map_viz.save_map_html


class TestPlotConvergence:
    def test_returns_figure_with_one_line_per_history(self):
        fig = visualize.plot_convergence({"AG": [5.0, 3.0, 1.0], "random_search": [5.0, 4.5, 4.0]})
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) == 2


class TestPlotHyperparamHeatmap:
    def test_returns_figure_with_expected_ticks(self):
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        fig = visualize.plot_hyperparam_heatmap(grid, ["a", "b"], ["x", "y"], "row", "col", "title")
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert len(ax.get_xticks()) == 2
        assert len(ax.get_yticks()) == 2


class TestPlotParetoFront2d:
    def test_returns_figure_with_one_scatter_per_front(self):
        fronts = {"NSGA-II": np.array([[1.0, 2.0], [1.5, 1.8]]), "MOEA/D": np.array([[1.1, 2.1]])}
        fig = visualize.plot_pareto_front_2d(fronts, "f1", "f2", "title")
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].collections) == 2


class TestPlotHypervolumeCurve:
    def test_returns_figure_with_one_line_per_algorithm(self):
        fig = visualize.plot_hypervolume_curve({"NSGA-II": [0.1, 0.2], "MOEA/D": [0.1, 0.15]})
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) == 2
        assert fig.axes[0].get_ylabel() == "Hypervolume"


class TestBuildBtsPlacementMap:
    def test_returns_folium_map(self):
        m = visualize.build_bts_placement_map(_placement_terrain(), np.array([[5_000.0, 5_000.0]]))
        assert isinstance(m, folium.Map)

    def test_marker_count_is_existing_plus_new(self):
        terrain = _placement_terrain()
        new_positions = np.array([[5_000.0, 5_000.0], [2_000.0, 2_000.0]])
        m = visualize.build_bts_placement_map(terrain, new_positions)
        assert len(_markers(m)) == len(terrain.bts) + len(new_positions)
