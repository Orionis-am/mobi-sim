"""Unit tests for Module C (opencellid_loader, terrain_sim, cell_id, toa, wifi_fp, ipinfo_client,
lbs_poi, map_viz, fitness).

External I/O (ipinfo.io, Nominatim) is mocked throughout via hand-rolled fakes, mirroring
module_b/test_module_b.py's FakeTwilioClient approach — no network access required to run this
file. Tests against the real docs/208.csv (392k rows) live outside pytest, exercised manually —
these tests use small synthetic CSVs instead, for speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from module_c import cell_id, opencellid_loader, terrain_sim

# --- opencellid_loader -------------------------------------------------------

_SAMPLE_ROWS = [
    "GSM,208,1,100,1,0,2.50,44.30,10000,20,1,1600000000,1700000000,0",
    "GSM,208,1,100,2,0,2.55,44.35,10000,20,1,1600000000,1700000000,0",
    "LTE,208,1,100,3,0,2.60,44.40,5000,30,1,1600000000,1700000000,0",
    "UMTS,1,1,100,4,0,2.65,44.45,8000,25,1,1600000000,1700000000,0",  # different MCC
    "LTE,208,1,100,5,0,10.0,50.0,5000,30,1,1600000000,1700000000,0",  # outside Aveyron bbox
]


def _write_sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("\n".join(_SAMPLE_ROWS) + "\n")
    return path


class TestLoadOpencellidCsv:
    def test_loads_named_columns(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=None, use_cache=False)
        assert list(df.columns) == opencellid_loader.COLUMNS
        assert len(df) == 5

    def test_filters_by_mcc(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=208, use_cache=False)
        assert len(df) == 4
        assert (df["mcc"] == 208).all()

    def test_lon_before_lat_column_order(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=208, use_cache=False)
        first = df.iloc[0]
        assert first["lon"] == 2.50
        assert first["lat"] == 44.30

    def test_cache_returns_independent_copies(self, tmp_path):
        csv_path = _write_sample_csv(tmp_path)
        df1 = opencellid_loader.load_opencellid_csv(csv_path, mcc=208, use_cache=True)
        df1.loc[0, "lon"] = -999.0
        df2 = opencellid_loader.load_opencellid_csv(csv_path, mcc=208, use_cache=True)
        assert df2.iloc[0]["lon"] != -999.0


class TestSampleBts:
    def test_filters_to_bbox(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=208, use_cache=False)
        sampled = opencellid_loader.sample_bts(df, bbox=opencellid_loader.AVEYRON_BBOX, n=10, seed=0)
        assert len(sampled) == 3
        assert (sampled["lon"] <= 3.5).all()

    def test_caps_at_available_rows_without_raising(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=208, use_cache=False)
        sampled = opencellid_loader.sample_bts(df, bbox=opencellid_loader.AVEYRON_BBOX, n=1000, seed=0)
        assert len(sampled) == 3

    def test_seed_reproducibility(self, tmp_path):
        df = opencellid_loader.load_opencellid_csv(_write_sample_csv(tmp_path), mcc=208, use_cache=False)
        a = opencellid_loader.sample_bts(df, bbox=opencellid_loader.AVEYRON_BBOX, n=2, seed=42)
        b = opencellid_loader.sample_bts(df, bbox=opencellid_loader.AVEYRON_BBOX, n=2, seed=42)
        assert a["cell"].tolist() == b["cell"].tolist()


# --- terrain_sim -------------------------------------------------------------


class TestLonlatToXy:
    def test_origin_maps_to_zero(self):
        x, y = terrain_sim.lonlat_to_xy(2.5, 44.3, lon0=2.5, lat0=44.3)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_latitude_is_about_111km(self):
        _, y = terrain_sim.lonlat_to_xy(2.5, 45.3, lon0=2.5, lat0=44.3)
        assert y == pytest.approx(111_195.0, rel=0.01)

    def test_roundtrip_through_xy_to_lonlat(self):
        lon0, lat0 = 2.5, 44.3
        x, y = terrain_sim.lonlat_to_xy(2.7, 44.5, lon0, lat0)
        lon, lat = terrain_sim.xy_to_lonlat(x, y, lon0, lat0)
        assert lon == pytest.approx(2.7, abs=1e-9)
        assert lat == pytest.approx(44.5, abs=1e-9)

    def test_accepts_numpy_arrays(self):
        lons = np.array([2.5, 2.6, 2.7])
        lats = np.array([44.3, 44.4, 44.5])
        x, y = terrain_sim.lonlat_to_xy(lons, lats, lon0=2.5, lat0=44.3)
        assert x.shape == (3,)
        assert y.shape == (3,)


class TestBuildTerrain:
    def _sample_df(self):
        return pd.DataFrame({"lon": [2.4, 2.5, 2.6], "lat": [44.2, 44.3, 44.4], "cell": [1, 2, 3]})

    def test_centers_on_bts_centroid(self):
        terrain = terrain_sim.build_terrain(self._sample_df())
        assert terrain.lon0 == pytest.approx(2.5)
        assert terrain.lat0 == pytest.approx(44.3)

    def test_positions_xy_shape_and_centroid_at_origin(self):
        terrain = terrain_sim.build_terrain(self._sample_df())
        positions = terrain.positions_xy
        assert positions.shape == (3, 2)
        assert positions.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-6)

    def test_bbox_xy_matches_min_max(self):
        terrain = terrain_sim.build_terrain(self._sample_df())
        x_min, x_max, y_min, y_max = terrain.bbox_xy
        x, y = terrain.positions_xy[:, 0], terrain.positions_xy[:, 1]
        assert x_min == pytest.approx(x.min())
        assert x_max == pytest.approx(x.max())
        assert y_min == pytest.approx(y.min())
        assert y_max == pytest.approx(y.max())

    def test_original_columns_preserved(self):
        terrain = terrain_sim.build_terrain(self._sample_df())
        assert terrain.bts["cell"].tolist() == [1, 2, 3]


# --- cell_id -----------------------------------------------------------------


def _square_terrain() -> terrain_sim.Terrain:
    # 4 BTS at the corners of a 10x10 square — the Voronoi diagram is exactly the two
    # perpendicular bisectors x=5, y=5, so each corner's clipped cell is a hand-computable
    # 5x5 quadrant (e.g. BTS at (0,0) -> the square [0,5]x[0,5], centroid (2.5, 2.5)).
    df = pd.DataFrame({"x": [0.0, 0.0, 10.0, 10.0], "y": [0.0, 10.0, 0.0, 10.0]})
    return terrain_sim.Terrain(bts=df, lon0=0.0, lat0=0.0)


class TestClipPolygonToBbox:
    def test_triangle_extending_past_box_is_clipped_to_box(self):
        triangle = np.array([[-5.0, 5.0], [15.0, 5.0], [5.0, 20.0]])
        clipped = cell_id._clip_polygon_to_bbox(triangle, (0.0, 10.0, 0.0, 10.0))
        assert np.all(clipped[:, 0] >= -1e-9)
        assert np.all(clipped[:, 0] <= 10.0 + 1e-9)
        assert np.all(clipped[:, 1] <= 10.0 + 1e-9)

    def test_polygon_fully_inside_box_is_unchanged(self):
        square = np.array([[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0]])
        clipped = cell_id._clip_polygon_to_bbox(square, (0.0, 10.0, 0.0, 10.0))
        np.testing.assert_allclose(sorted(clipped.tolist()), sorted(square.tolist()))

    def test_polygon_fully_outside_box_clips_to_empty(self):
        square = np.array([[20.0, 20.0], [20.0, 25.0], [25.0, 25.0], [25.0, 20.0]])
        clipped = cell_id._clip_polygon_to_bbox(square, (0.0, 10.0, 0.0, 10.0))
        assert len(clipped) == 0


class TestPolygonCentroid:
    def test_unit_square_centroid(self):
        square = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        centroid = cell_id._polygon_centroid(square)
        np.testing.assert_allclose(centroid, [0.5, 0.5])

    def test_degenerate_polygon_falls_back_to_vertex_mean(self):
        line = np.array([[0.0, 0.0], [2.0, 0.0]])
        centroid = cell_id._polygon_centroid(line)
        np.testing.assert_allclose(centroid, [1.0, 0.0])

    def test_zero_area_collinear_points_falls_back_to_vertex_mean(self):
        collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        centroid = cell_id._polygon_centroid(collinear)
        np.testing.assert_allclose(centroid, [1.0, 0.0])


class TestVoronoiCellCentroids:
    def test_matches_hand_computed_quadrant_centroids(self):
        centroids = cell_id.voronoi_cell_centroids(_square_terrain())
        expected = np.array([[2.5, 2.5], [2.5, 7.5], [7.5, 2.5], [7.5, 7.5]])
        np.testing.assert_allclose(centroids, expected, atol=1e-6)


class TestEstimatePosition:
    def test_returns_centroid_of_nearest_bts(self):
        terrain = _square_terrain()
        estimate = cell_id.estimate_position(np.array([1.0, 1.0]), terrain)
        np.testing.assert_allclose(estimate, [2.5, 2.5], atol=1e-6)

    def test_accepts_precomputed_centroids(self):
        terrain = _square_terrain()
        centroids = cell_id.voronoi_cell_centroids(terrain)
        estimate = cell_id.estimate_position(np.array([9.0, 9.0]), terrain, centroids=centroids)
        np.testing.assert_allclose(estimate, [7.5, 7.5], atol=1e-6)


class TestEvaluateAccuracy:
    def test_returns_finite_median_and_matching_error_count(self):
        result = cell_id.evaluate_accuracy(_square_terrain(), n_positions=50, seed=0)
        assert result["median_error_m"] >= 0.0
        assert len(result["errors_m"]) == 50

    def test_seed_reproducibility(self):
        a = cell_id.evaluate_accuracy(_square_terrain(), n_positions=20, seed=7)
        b = cell_id.evaluate_accuracy(_square_terrain(), n_positions=20, seed=7)
        assert a["median_error_m"] == pytest.approx(b["median_error_m"])
