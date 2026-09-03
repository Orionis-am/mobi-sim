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
from module_a.fitness import codec_fitness
from module_d.fitness import DEFAULT_TOTAL_CAPACITY_KBPS, qos_fitness_components
from module_f import ag_scratch, benchmark, compare, pb1_codec, pb2_bts, pb3_qos, visualize

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


# --- benchmark ---------------------------------------------------------------


class TestRastrigin:
    def test_global_minimum_at_origin(self):
        assert benchmark.rastrigin(np.zeros(5)) == pytest.approx(0.0)

    def test_positive_away_from_origin(self):
        assert benchmark.rastrigin(np.array([1.5, -2.0])) > 0.0


class TestRosenbrock:
    def test_global_minimum_at_ones(self):
        assert benchmark.rosenbrock(np.ones(4)) == pytest.approx(0.0)

    def test_positive_away_from_ones(self):
        assert benchmark.rosenbrock(np.zeros(4)) > 0.0


class TestRandomSearch:
    def test_history_is_non_increasing(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        result = benchmark.random_search(benchmark.rastrigin, bounds, n_evaluations=30, seed=0)
        history = result.history_best_fitness
        assert all(a >= b for a, b in zip(history, history[1:]))

    def test_n_evaluations_matches_request(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        result = benchmark.random_search(benchmark.rastrigin, bounds, n_evaluations=17, seed=0)
        assert result.n_evaluations == 17
        assert len(result.history_best_fitness) == 17

    def test_reproducible_with_same_seed(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        r1 = benchmark.random_search(benchmark.rastrigin, bounds, n_evaluations=20, seed=5)
        r2 = benchmark.random_search(benchmark.rastrigin, bounds, n_evaluations=20, seed=5)
        assert r1.best_fitness == r2.best_fitness
        assert np.array_equal(r1.best_x, r2.best_x)


class TestHillClimbing:
    def test_history_is_non_increasing(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        result = benchmark.hill_climbing(benchmark.rastrigin, bounds, n_iterations=30, seed=0)
        history = result.history_best_fitness
        assert all(a >= b for a, b in zip(history, history[1:]))

    def test_n_evaluations_matches_request(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        result = benchmark.hill_climbing(benchmark.rastrigin, bounds, n_iterations=12, seed=0)
        assert result.n_evaluations == 12
        assert len(result.history_best_fitness) == 12

    def test_reproducible_with_same_seed(self):
        bounds = [benchmark.RASTRIGIN_BOUNDS] * 2
        r1 = benchmark.hill_climbing(benchmark.rastrigin, bounds, n_iterations=15, seed=3)
        r2 = benchmark.hill_climbing(benchmark.rastrigin, bounds, n_iterations=15, seed=3)
        assert r1.best_fitness == r2.best_fitness


class TestRunBenchmarkSuite:
    def test_returns_both_functions_with_matched_budgets(self):
        config = ag_scratch.GAConfig(pop_size=10, n_generations=5, seed=0)
        results = benchmark.run_benchmark_suite(dim=2, seed=0, ga_config=config)
        assert set(results.keys()) == {"rastrigin", "rosenbrock"}
        for name in results:
            budget = config.pop_size * config.n_generations
            assert results[name]["ga"].n_evaluations == budget
            assert results[name]["random_search"].n_evaluations == budget
            assert results[name]["hill_climbing"].n_evaluations == budget


class TestHyperparameterSweep:
    def test_shape_and_finiteness(self):
        grid = benchmark.hyperparameter_sweep(
            benchmark.rastrigin,
            benchmark.RASTRIGIN_BOUNDS,
            dim=2,
            pop_sizes=(5, 10),
            crossover_rates=(0.6, 0.9),
            mutation_rates=(0.05, 0.2),
            n_runs=2,
            n_generations=5,
            seed=0,
        )
        assert grid.shape == (2, 2, 2)
        assert np.all(np.isfinite(grid))


# --- pb1_codec ---------------------------------------------------------------

_CODEC_KWARGS = {"seed": 0}  # keep codec_fitness deterministic and cheap across pb1_codec's tests


class TestDeapGaCodec:
    def test_history_is_non_decreasing(self):
        result = pb1_codec.deap_ga_codec(pop_size=6, n_generations=4, seed=0)
        history = result.history_best_fitness
        assert all(a <= b for a, b in zip(history, history[1:]))

    def test_best_fitness_matches_recomputed_chromosome(self):
        result = pb1_codec.deap_ga_codec(pop_size=6, n_generations=4, seed=0)
        recomputed = codec_fitness(result.best_chromosome, seed=0)
        assert result.best_fitness == pytest.approx(recomputed)

    def test_n_evaluations_matches_formula(self):
        result = pb1_codec.deap_ga_codec(pop_size=6, n_generations=4, seed=0)
        assert result.n_evaluations == 6 * 4

    def test_reproducible_with_same_seed(self):
        r1 = pb1_codec.deap_ga_codec(pop_size=6, n_generations=4, seed=1)
        r2 = pb1_codec.deap_ga_codec(pop_size=6, n_generations=4, seed=1)
        assert r1.best_chromosome == r2.best_chromosome
        assert r1.best_fitness == r2.best_fitness


class TestRandomSearchCodec:
    def test_n_evaluations_matches_request(self):
        result = pb1_codec.random_search_codec(n_evaluations=5, **_CODEC_KWARGS)
        assert result.n_evaluations == 5
        assert len(result.history_best_fitness) == 5

    def test_history_is_non_decreasing(self):
        result = pb1_codec.random_search_codec(n_evaluations=8, **_CODEC_KWARGS)
        history = result.history_best_fitness
        assert all(a <= b for a, b in zip(history, history[1:]))


class TestCreatorRegistrationIsIdempotent:
    def test_reload_does_not_raise(self):
        # pb1_codec registers DEAP's process-global `creator.FitnessMaxCodec`/`IndividualCodec`
        # once at import time, guarded by hasattr so a second import (e.g. test collection
        # re-importing, or a notebook re-running the cell) doesn't try to redefine them.
        import importlib

        importlib.reload(pb1_codec)
        assert hasattr(pb1_codec.creator, "FitnessMaxCodec")


class TestGridSearchCodec:
    def test_evaluation_count_matches_grid_size(self):
        result = pb1_codec.grid_search_codec(plc_steps=2, **_CODEC_KWARGS)
        assert result.n_evaluations == 5 * 4 * 2 * 3

    def test_best_chromosome_is_a_grid_point(self):
        result = pb1_codec.grid_search_codec(plc_steps=2, **_CODEC_KWARGS)
        assert result.best_chromosome[0] in [float(b) for b in pb1_codec.BITRATE_CHOICES_KBPS]
        assert result.best_chromosome[1] in [float(f) for f in pb1_codec.FRAME_SIZE_CHOICES_MS]
        assert result.best_chromosome[2] in [0.0, 1.0]
        assert result.best_chromosome[3] in [0.0, 1.0, 2.0]


# --- pb2_bts -----------------------------------------------------------------

_PB2_FITNESS_KWARGS = {"n_test_points": 20}  # small n_test_points keeps tests fast


class TestBtsPlacementProblem:
    def test_dimensions_match_n_new_bts(self):
        problem = pb2_bts.BtsPlacementProblem(n_new_bts=3, terrain=_placement_terrain())
        assert problem.n_var == 6
        assert problem.n_obj == 3

    def test_evaluate_fills_f_with_expected_shape(self):
        problem = pb2_bts.BtsPlacementProblem(n_new_bts=2, terrain=_placement_terrain(), **_PB2_FITNESS_KWARGS)
        X = np.random.default_rng(0).uniform(0.0, 1.0, size=(5, 4))
        out = {}
        problem._evaluate(X, out)
        assert out["F"].shape == (5, 3)


class TestRunNsga2:
    def test_final_front_shape_and_budget(self):
        result = pb2_bts.run_nsga2(n_new_bts=1, terrain=_placement_terrain(), pop_size=8, n_generations=3, seed=0, **_PB2_FITNESS_KWARGS)
        assert result.final_F.shape[1] == 3
        assert len(result.final_F) <= 8
        assert result.n_evaluations == 8 * 3

    def test_reproducible_with_same_seed(self):
        kwargs = dict(n_new_bts=1, terrain=_placement_terrain(), pop_size=8, n_generations=3, seed=0, **_PB2_FITNESS_KWARGS)
        r1 = pb2_bts.run_nsga2(**kwargs)
        r2 = pb2_bts.run_nsga2(**kwargs)
        np.testing.assert_array_equal(r1.final_F, r2.final_F)


class TestRunMoead:
    def test_final_front_shape_and_budget(self):
        result = pb2_bts.run_moead(n_new_bts=1, terrain=_placement_terrain(), n_partitions=3, n_generations=3, seed=0, **_PB2_FITNESS_KWARGS)
        assert result.final_F.shape[1] == 3
        n_ref_dirs = len(result.F_history[0])
        assert result.n_evaluations == n_ref_dirs * 3


class TestHypervolumeHistory:
    def test_length_matches_n_generations_and_is_finite(self):
        result = pb2_bts.run_nsga2(n_new_bts=1, terrain=_placement_terrain(), pop_size=8, n_generations=4, seed=0, **_PB2_FITNESS_KWARGS)
        history = pb2_bts.hypervolume_history(result)
        assert len(history) == 4
        assert all(np.isfinite(v) for v in history)

    def test_explicit_ref_point_is_used(self):
        result = pb2_bts.Pb2Result(
            F_history=[np.array([[1.0, 1.0, 1.0]])],
            final_F=np.array([[1.0, 1.0, 1.0]]),
            final_X=np.array([[0]]),
            n_evaluations=1,
        )
        history = pb2_bts.hypervolume_history(result, ref_point=np.array([2.0, 2.0, 2.0]))
        assert history == [pytest.approx(1.0)]


class TestSelectBestCompromise:
    def test_returns_point_closest_to_ideal(self):
        result = pb2_bts.Pb2Result(
            F_history=[],
            final_F=np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [1.0, 1.0, 1.0]]),
            final_X=np.array([[0], [1], [2], [3]]),
            n_evaluations=0,
        )
        best = pb2_bts.select_best_compromise(result)
        assert best[0] == 3


# --- pb3_qos -----------------------------------------------------------------


class TestBounds:
    def test_length_matches_twice_profile_count(self):
        bounds = pb3_qos._bounds()
        assert len(bounds) == 20


class TestRunDe:
    def test_finite_and_non_decreasing_history(self):
        result = pb3_qos.run_de(maxiter=5, popsize=5, seed=0)
        assert np.isfinite(result.best_fitness)
        history = result.history_best_fitness
        assert all(a <= b for a, b in zip(history, history[1:]))

    def test_reproducible_with_same_seed(self):
        r1 = pb3_qos.run_de(maxiter=5, popsize=5, seed=0)
        r2 = pb3_qos.run_de(maxiter=5, popsize=5, seed=0)
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_chromosome, r2.best_chromosome)


class TestRunPso:
    def test_history_length_matches_maxiter_plus_one(self):
        result = pb3_qos.run_pso(swarmsize=10, maxiter=5, seed=0)
        assert len(result.history_best_fitness) == 6

    def test_history_is_non_decreasing(self):
        result = pb3_qos.run_pso(swarmsize=10, maxiter=5, seed=0)
        history = result.history_best_fitness
        assert all(a <= b for a, b in zip(history, history[1:]))

    def test_reproducible_with_same_seed(self):
        r1 = pb3_qos.run_pso(swarmsize=10, maxiter=5, seed=0)
        r2 = pb3_qos.run_pso(swarmsize=10, maxiter=5, seed=0)
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_chromosome, r2.best_chromosome)


class TestConstraintRespected:
    def test_total_bandwidth_close_to_or_under_capacity(self):
        result = pb3_qos.run_de(maxiter=40, popsize=15, seed=0)
        components = qos_fitness_components(result.best_chromosome)
        assert components["total_bandwidth_kbps"] < DEFAULT_TOTAL_CAPACITY_KBPS * 1.2


# --- compare -----------------------------------------------------------------


class _FakeResult:
    def __init__(self, best_fitness: float, n_evaluations: int = 10):
        self.best_fitness = best_fitness
        self.n_evaluations = n_evaluations


class TestSummarizeRuns:
    def test_zero_variance_when_runs_identical(self):
        summary = compare.summarize_runs("p", "algo", lambda seed: _FakeResult(5.0), n_runs=4, seed=0)
        assert summary.variance_over_runs == 0.0
        assert summary.best_value == 5.0

    def test_positive_variance_when_runs_differ(self):
        summary = compare.summarize_runs("p", "algo", lambda seed: _FakeResult(float(seed)), n_runs=4, seed=0)
        assert summary.variance_over_runs > 0.0

    def test_maximize_false_picks_minimum(self):
        summary = compare.summarize_runs("p", "algo", lambda seed: _FakeResult(float(seed)), n_runs=4, seed=0, maximize=False)
        assert summary.best_value == 0.0


class TestBuildComparisonTable:
    def test_one_row_per_summary_with_expected_keys(self):
        summaries = [
            compare.AlgoRunSummary("p1", "AG", 1.0, 0.1, 0.0, {}, 10),
            compare.AlgoRunSummary("p1", "DE", 2.0, 0.2, 0.1, {}, 20),
        ]
        table = compare.build_comparison_table(summaries)
        assert len(table) == 2
        assert set(table[0].keys()) == {
            "problem",
            "algorithm",
            "best_value",
            "convergence_time_s",
            "variance_over_runs",
            "recommended_params",
            "n_evaluations",
        }


class TestRunFixedBudgetComparison:
    def test_pb3_returns_de_and_pso_histories(self):
        result = compare.run_fixed_budget_comparison("pb3_qos", n_evaluations=200, n_runs=2, seed=0)
        assert set(result.keys()) == {"de", "pso"}
        assert len(result["de"]) > 0
        assert len(result["pso"]) > 0

    def test_unknown_problem_raises(self):
        with pytest.raises(ValueError):
            compare.run_fixed_budget_comparison("unknown", n_evaluations=100)
