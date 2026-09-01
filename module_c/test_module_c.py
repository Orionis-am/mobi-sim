"""Unit tests for Module C (opencellid_loader, terrain_sim, cell_id, toa, wifi_fp, ipinfo_client,
lbs_poi, map_viz, fitness).

External I/O (ipinfo.io, Nominatim) is mocked throughout via hand-rolled fakes, mirroring
module_b/test_module_b.py's FakeTwilioClient approach — no network access required to run this
file. Tests against the real docs/208.csv (392k rows) live outside pytest, exercised manually —
these tests use small synthetic CSVs instead, for speed.
"""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import pytest
import requests

from module_c import cell_id, fitness, ipinfo_client, lbs_poi, map_viz, opencellid_loader, terrain_sim, toa, wifi_fp

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


class TestCellIdEvaluateAccuracy:
    def test_returns_finite_median_and_matching_error_count(self):
        result = cell_id.evaluate_accuracy(_square_terrain(), n_positions=50, seed=0)
        assert result["median_error_m"] >= 0.0
        assert len(result["errors_m"]) == 50

    def test_seed_reproducibility(self):
        a = cell_id.evaluate_accuracy(_square_terrain(), n_positions=20, seed=7)
        b = cell_id.evaluate_accuracy(_square_terrain(), n_positions=20, seed=7)
        assert a["median_error_m"] == pytest.approx(b["median_error_m"])


# --- toa -----------------------------------------------------------------


def _corner_terrain() -> terrain_sim.Terrain:
    # 4 corners of a 100x100 square + 1 center BTS — enough anchors for well-conditioned trilateration.
    df = pd.DataFrame({"x": [0.0, 100.0, 0.0, 100.0, 50.0], "y": [0.0, 0.0, 100.0, 100.0, 50.0]})
    return terrain_sim.Terrain(bts=df, lon0=0.0, lat0=0.0)


class TestNearestAnchors:
    def test_selects_k_closest_by_distance(self):
        terrain = terrain_sim.Terrain(bts=pd.DataFrame({"x": [0.0, 10.0, 20.0, 30.0], "y": [0.0, 0.0, 0.0, 0.0]}), lon0=0.0, lat0=0.0)
        anchors = toa._nearest_anchors(np.array([1.0, 0.0]), terrain, k=2)
        assert sorted(anchors[:, 0].tolist()) == [0.0, 10.0]

    def test_caps_k_at_available_bts(self):
        terrain = terrain_sim.Terrain(bts=pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 0.0]}), lon0=0.0, lat0=0.0)
        anchors = toa._nearest_anchors(np.array([0.0, 0.0]), terrain, k=5)
        assert len(anchors) == 2


class TestSimulatePseudoranges:
    def test_zero_noise_matches_true_distance(self):
        anchors = np.array([[0.0, 0.0], [100.0, 0.0]])
        true_xy = np.array([30.0, 40.0])
        measured = toa.simulate_pseudoranges(true_xy, anchors, timing_noise_std_s=0.0)
        expected = np.linalg.norm(anchors - true_xy, axis=1)
        np.testing.assert_allclose(measured, expected, atol=1e-6)

    def test_noise_perturbs_distance(self):
        anchors = np.array([[0.0, 0.0]])
        true_xy = np.array([100.0, 0.0])
        rng = np.random.default_rng(0)
        measured = toa.simulate_pseudoranges(true_xy, anchors, timing_noise_std_s=50e-9, rng=rng)
        assert measured[0] != pytest.approx(100.0)


class TestTrilaterate:
    def test_recovers_position_exactly_with_zero_noise(self):
        anchors = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        true_xy = np.array([40.0, 60.0])
        measured = np.linalg.norm(anchors - true_xy, axis=1)
        estimate = toa.trilaterate(anchors, measured)
        np.testing.assert_allclose(estimate, true_xy, atol=1e-3)

    def test_accepts_explicit_initial_guess(self):
        anchors = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
        true_xy = np.array([40.0, 60.0])
        measured = np.linalg.norm(anchors - true_xy, axis=1)
        estimate = toa.trilaterate(anchors, measured, initial_guess=np.array([50.0, 50.0]))
        np.testing.assert_allclose(estimate, true_xy, atol=1e-3)


class TestToaEstimatePosition:
    def test_reasonably_close_to_true_position(self):
        terrain = _corner_terrain()
        rng = np.random.default_rng(1)
        estimate = toa.estimate_position(np.array([40.0, 60.0]), terrain, k=4, timing_noise_std_s=1e-9, rng=rng)
        assert np.linalg.norm(estimate - np.array([40.0, 60.0])) < 50.0


class TestToaEvaluateAccuracy:
    def test_returns_finite_median_and_matching_error_count(self):
        result = toa.evaluate_accuracy(_corner_terrain(), n_positions=20, seed=0)
        assert result["median_error_m"] >= 0.0
        assert len(result["errors_m"]) == 20

    def test_seed_reproducibility(self):
        a = toa.evaluate_accuracy(_corner_terrain(), n_positions=10, seed=5)
        b = toa.evaluate_accuracy(_corner_terrain(), n_positions=10, seed=5)
        assert a["median_error_m"] == pytest.approx(b["median_error_m"])


# --- wifi_fp -------------------------------------------------------------


def _large_square_terrain() -> terrain_sim.Terrain:
    # 500x500 area, 100m grid cells -> 5x5 = 25 fingerprint grid cells.
    df = pd.DataFrame({"x": [0.0, 500.0, 0.0, 500.0, 250.0], "y": [0.0, 0.0, 500.0, 500.0, 250.0]})
    return terrain_sim.Terrain(bts=df, lon0=0.0, lat0=0.0)


class TestSimulateRssi:
    def test_rssi_decreases_with_distance(self):
        bts = np.array([[0.0, 0.0]])
        near = wifi_fp.simulate_rssi(np.array([10.0, 0.0]), bts)
        far = wifi_fp.simulate_rssi(np.array([1000.0, 0.0]), bts)
        assert near[0] > far[0]

    def test_zero_noise_is_deterministic(self):
        bts = np.array([[0.0, 0.0], [100.0, 0.0]])
        a = wifi_fp.simulate_rssi(np.array([10.0, 10.0]), bts, noise_std_db=0.0)
        b = wifi_fp.simulate_rssi(np.array([10.0, 10.0]), bts, noise_std_db=0.0)
        np.testing.assert_allclose(a, b)

    def test_noise_perturbs_reading(self):
        bts = np.array([[0.0, 0.0]])
        rng = np.random.default_rng(0)
        rssi = wifi_fp.simulate_rssi(np.array([10.0, 0.0]), bts, noise_std_db=4.0, rng=rng)
        noiseless = wifi_fp.simulate_rssi(np.array([10.0, 0.0]), bts, noise_std_db=0.0)
        assert rssi[0] != pytest.approx(noiseless[0])


class TestBuildFingerprintGrid:
    def test_grid_size_matches_cell_and_zone_size(self):
        terrain = _large_square_terrain()
        centroids, fingerprints = wifi_fp.build_fingerprint_grid(terrain, zone_center_xy=np.array([250.0, 250.0]), zone_size_m=500.0, cell_size_m=100.0)
        assert len(centroids) == 25
        assert fingerprints.shape == (25, 5)

    def test_default_zone_is_scoped_not_full_terrain_bbox(self):
        # A terrain far larger than the default 2km zone must not blow up the grid size.
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({"x": rng.uniform(0, 200_000, n), "y": rng.uniform(0, 150_000, n)})
        terrain = terrain_sim.Terrain(bts=df, lon0=0.0, lat0=0.0)
        centroids, _ = wifi_fp.build_fingerprint_grid(terrain)
        assert len(centroids) < 1_000

    def test_zone_smaller_than_cell_size_still_yields_one_cell(self):
        # zone_size_m < cell_size_m means np.arange would produce zero points on that axis;
        # build_fingerprint_grid must fall back to a single centroid rather than an empty grid.
        terrain = _large_square_terrain()
        centroids, fingerprints = wifi_fp.build_fingerprint_grid(terrain, zone_center_xy=np.array([0.0, 0.0]), zone_size_m=50.0, cell_size_m=100.0)
        assert len(centroids) == 1


class TestWifiEstimatePosition:
    def test_noiseless_query_at_grid_centroid_recovers_it_exactly(self):
        terrain = _large_square_terrain()
        centroids, fingerprints = wifi_fp.build_fingerprint_grid(terrain, zone_center_xy=np.array([250.0, 250.0]), zone_size_m=500.0)
        model = wifi_fp.fit_knn(fingerprints, k=3)

        query = wifi_fp.simulate_rssi(centroids[10], terrain.positions_xy)
        estimate = wifi_fp.estimate_position(query, centroids, model)
        np.testing.assert_allclose(estimate, centroids[10], atol=1e-6)


class TestWifiEvaluateAccuracy:
    _ZONE = dict(zone_center_xy=np.array([250.0, 250.0]), zone_size_m=500.0)

    def test_returns_finite_median_and_matching_error_count(self):
        result = wifi_fp.evaluate_accuracy(_large_square_terrain(), n_positions=30, seed=0, **self._ZONE)
        assert result["median_error_m"] >= 0.0
        assert len(result["errors_m"]) == 30

    def test_seed_reproducibility(self):
        a = wifi_fp.evaluate_accuracy(_large_square_terrain(), n_positions=20, seed=5, **self._ZONE)
        b = wifi_fp.evaluate_accuracy(_large_square_terrain(), n_positions=20, seed=5, **self._ZONE)
        assert a["median_error_m"] == pytest.approx(b["median_error_m"])

    def test_more_noise_degrades_accuracy(self):
        terrain = _large_square_terrain()
        low_noise = wifi_fp.evaluate_accuracy(terrain, n_positions=50, seed=0, noise_std_db=4.0, **self._ZONE)
        high_noise = wifi_fp.evaluate_accuracy(terrain, n_positions=50, seed=0, noise_std_db=50.0, **self._ZONE)
        assert high_noise["median_error_m"] > low_noise["median_error_m"]


class TestSweepNoise:
    def test_one_result_per_noise_level_with_matching_field(self):
        results = wifi_fp.sweep_noise(_large_square_terrain(), [0.0, 4.0, 10.0], n_positions=10, seed=0)
        assert [r["noise_std_db"] for r in results] == [0.0, 4.0, 10.0]
        assert all("median_error_m" in r for r in results)

    def test_seed_reproducibility(self):
        a = wifi_fp.sweep_noise(_large_square_terrain(), [0.0, 10.0], n_positions=10, seed=3)
        b = wifi_fp.sweep_noise(_large_square_terrain(), [0.0, 10.0], n_positions=10, seed=3)
        assert [r["median_error_m"] for r in a] == [r["median_error_m"] for r in b]


# --- ipinfo_client -----------------------------------------------------------


class FakeIpinfoResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_data


class FakeIpinfoSession:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.get_calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.get_calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeIpinfoResponse(self._json_data, self.status_code)


class TestLocateIp:
    def test_parses_full_response(self, monkeypatch):
        monkeypatch.setenv("IPINFO_TOKEN", "test-token")
        session = FakeIpinfoSession({"ip": "8.8.8.8", "city": "Mountain View", "region": "California", "country": "US", "loc": "37.4056,-122.0775"})
        location = ipinfo_client.locate_ip("8.8.8.8", session=session)
        assert location.ip == "8.8.8.8"
        assert location.city == "Mountain View"
        assert location.region == "California"
        assert location.country == "US"
        assert location.lat == pytest.approx(37.4056)
        assert location.lon == pytest.approx(-122.0775)

    def test_own_ip_when_none_given(self, monkeypatch):
        monkeypatch.setenv("IPINFO_TOKEN", "test-token")
        session = FakeIpinfoSession({"ip": "1.2.3.4", "loc": "48.85,2.35"})
        ipinfo_client.locate_ip(session=session)
        assert session.get_calls[0]["url"] == "https://ipinfo.io/json"

    def test_specific_ip_included_in_url(self, monkeypatch):
        monkeypatch.setenv("IPINFO_TOKEN", "test-token")
        session = FakeIpinfoSession({"ip": "8.8.8.8", "loc": "37.0,-122.0"})
        ipinfo_client.locate_ip("8.8.8.8", session=session)
        assert session.get_calls[0]["url"] == "https://ipinfo.io/8.8.8.8/json"

    def test_missing_loc_field_yields_none_coordinates(self, monkeypatch):
        monkeypatch.setenv("IPINFO_TOKEN", "test-token")
        session = FakeIpinfoSession({"ip": "8.8.8.8", "city": "Mountain View"})
        location = ipinfo_client.locate_ip("8.8.8.8", session=session)
        assert location.lat is None
        assert location.lon is None

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("IPINFO_TOKEN", raising=False)
        with pytest.raises(RuntimeError):
            ipinfo_client.locate_ip("8.8.8.8", session=FakeIpinfoSession({}))

    def test_explicit_token_bypasses_env(self, monkeypatch):
        monkeypatch.delenv("IPINFO_TOKEN", raising=False)
        session = FakeIpinfoSession({"ip": "8.8.8.8", "loc": "1.0,2.0"})
        ipinfo_client.locate_ip("8.8.8.8", token="explicit-token", session=session)
        assert session.get_calls[0]["params"]["token"] == "explicit-token"

    def test_http_error_status_raises(self, monkeypatch):
        monkeypatch.setenv("IPINFO_TOKEN", "test-token")
        session = FakeIpinfoSession({}, status_code=403)
        with pytest.raises(requests.HTTPError):
            ipinfo_client.locate_ip("8.8.8.8", session=session)


# --- lbs_poi -----------------------------------------------------------------


def _make_element(lat, lon, name="Test POI", amenity="cafe"):
    return {"lat": lat, "lon": lon, "tags": {"name": name, "amenity": amenity}}


class FakeOverpassResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._json_data


class FakeOverpassSession:
    def __init__(self, elements=None, status_code=200):
        self._elements = elements if elements is not None else []
        self.status_code = status_code
        self.post_calls: list[dict] = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.post_calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        return FakeOverpassResponse({"elements": self._elements}, self.status_code)


class TestHaversineDistance:
    def test_one_degree_latitude_is_about_111km(self):
        d = lbs_poi._haversine_distance_m(48.0, 2.0, 49.0, 2.0)
        assert d == pytest.approx(111_195.0, rel=0.01)

    def test_same_point_is_zero(self):
        d = lbs_poi._haversine_distance_m(48.0, 2.0, 48.0, 2.0)
        assert d == pytest.approx(0.0, abs=1e-6)


class TestNearbyPois:
    def test_sends_query_with_lat_lon_radius(self):
        session = FakeOverpassSession([])
        lbs_poi.nearby_pois(48.0, 2.0, radius_m=500.0, session=session)
        payload = session.post_calls[0]["data"]["data"]
        assert "48.0" in payload
        assert "2.0" in payload
        assert "500.0" in payload

    def test_sends_mandatory_user_agent_header(self):
        session = FakeOverpassSession([])
        lbs_poi.nearby_pois(48.0, 2.0, session=session)
        assert session.post_calls[0]["headers"]["User-Agent"] == lbs_poi.USER_AGENT

    def test_sorts_by_distance_nearest_first(self):
        elements = [
            _make_element(48.01, 2.0, name="Far"),
            _make_element(48.0005, 2.0, name="Near"),
            _make_element(48.002, 2.0, name="Mid"),
        ]
        session = FakeOverpassSession(elements)
        pois = lbs_poi.nearby_pois(48.0, 2.0, session=session)
        assert [p.name for p in pois] == ["Near", "Mid", "Far"]

    def test_truncates_to_limit(self):
        elements = [_make_element(48.0 + i * 0.0001, 2.0, name=f"P{i}") for i in range(20)]
        session = FakeOverpassSession(elements)
        pois = lbs_poi.nearby_pois(48.0, 2.0, limit=5, session=session)
        assert len(pois) == 5

    def test_skips_elements_missing_coordinates(self):
        elements = [{"tags": {"name": "NoCoords"}}, _make_element(48.001, 2.0, name="HasCoords")]
        session = FakeOverpassSession(elements)
        pois = lbs_poi.nearby_pois(48.0, 2.0, session=session)
        assert [p.name for p in pois] == ["HasCoords"]

    def test_missing_tags_default_to_placeholder(self):
        elements = [{"lat": 48.001, "lon": 2.0, "tags": {}}]
        session = FakeOverpassSession(elements)
        pois = lbs_poi.nearby_pois(48.0, 2.0, session=session)
        assert pois[0].name == "?"
        assert pois[0].category == "?"

    def test_http_error_status_raises(self):
        session = FakeOverpassSession([], status_code=500)
        with pytest.raises(requests.HTTPError):
            lbs_poi.nearby_pois(48.0, 2.0, session=session)


# --- map_viz -------------------------------------------------------------


class _FakePoi:
    def __init__(self, lat, lon, name="Cafe X", category="cafe", distance_m=42.0):
        self.lat, self.lon, self.name, self.category, self.distance_m = lat, lon, name, category, distance_m


def _markers(m: folium.Map) -> list:
    return [c for c in m._children.values() if isinstance(c, folium.Marker)]


def _circles(m: folium.Map) -> list:
    return [c for c in m._children.values() if isinstance(c, folium.Circle)]


class TestBuildPositionMap:
    def test_returns_folium_map(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        assert isinstance(m, folium.Map)

    def test_estimated_marker_always_added(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        assert len(_markers(m)) == 1

    def test_true_position_marker_added_when_given(self):
        m = map_viz.build_position_map(44.351, 2.571, 44.35, 2.57)
        assert len(_markers(m)) == 2

    def test_true_position_marker_omitted_when_none(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        assert len(_markers(m)) == 1

    def test_uncertainty_circle_added_when_given(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57, uncertainty_radius_m=100.0)
        assert len(_circles(m)) == 1

    def test_uncertainty_circle_omitted_when_none(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        assert len(_circles(m)) == 0

    def test_poi_markers_added(self):
        pois = [_FakePoi(44.351, 2.572), _FakePoi(44.349, 2.569)]
        m = map_viz.build_position_map(None, None, 44.35, 2.57, pois=pois)
        assert len(_markers(m)) == 1 + len(pois)

    def test_no_pois_defaults_to_none_gracefully(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57, pois=None)
        assert len(_markers(m)) == 1


class TestAddToaRangeCircles:
    def test_adds_one_circle_per_anchor(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        anchors = [(44.351, 2.571), (44.349, 2.569)]
        map_viz.add_toa_range_circles(m, anchors, [100.0, 150.0])
        assert len(_circles(m)) == 2

    def test_returns_the_same_map_instance(self):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        result = map_viz.add_toa_range_circles(m, [(44.351, 2.571)], [100.0])
        assert result is m


class TestSaveMapHtml:
    def test_creates_file_and_returns_path(self, tmp_path):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        path = map_viz.save_map_html(m, "test_map.html", output_dir=tmp_path)
        assert path.exists()
        assert "<html" in path.read_text(encoding="utf-8").lower()

    def test_creates_output_dir_if_missing(self, tmp_path):
        m = map_viz.build_position_map(None, None, 44.35, 2.57)
        out_dir = tmp_path / "nested" / "results"
        map_viz.save_map_html(m, "test_map.html", output_dir=out_dir)
        assert out_dir.exists()


# --- fitness ---------------------------------------------------------------


def _fitness_terrain() -> terrain_sim.Terrain:
    # 4 real BTS at the corners of a 10km square.
    df = pd.DataFrame({"x": [0.0, 10_000.0, 0.0, 10_000.0], "y": [0.0, 0.0, 10_000.0, 10_000.0]})
    return terrain_sim.Terrain(bts=df, lon0=0.0, lat0=0.0)


class TestDecodeChromosome:
    def test_zero_maps_to_bbox_min_one_maps_to_bbox_max(self):
        terrain = _fitness_terrain()
        positions = fitness.decode_chromosome([0.0, 0.0, 1.0, 1.0], terrain)
        np.testing.assert_allclose(positions[0], [0.0, 0.0])
        np.testing.assert_allclose(positions[1], [10_000.0, 10_000.0])

    def test_out_of_range_genes_are_clipped(self):
        terrain = _fitness_terrain()
        positions = fitness.decode_chromosome([-0.5, 1.5], terrain)
        np.testing.assert_allclose(positions[0], [0.0, 10_000.0])

    def test_multiple_bts_shape(self):
        terrain = _fitness_terrain()
        positions = fitness.decode_chromosome([0.0, 0.0, 0.5, 0.5, 1.0, 1.0], terrain)
        assert positions.shape == (3, 2)

    def test_odd_length_raises(self):
        terrain = _fitness_terrain()
        with pytest.raises(ValueError):
            fitness.decode_chromosome([0.0, 0.5, 1.0], terrain)


class TestCoverageFraction:
    def test_huge_radius_covers_everything(self):
        terrain = _fitness_terrain()
        new_positions = np.array([[5_000.0, 5_000.0]])
        result = fitness.coverage_fraction(terrain, new_positions, n_test_points=50, coverage_radius_m=1_000_000.0, seed=0)
        assert result == pytest.approx(1.0)

    def test_tiny_radius_covers_almost_nothing(self):
        terrain = _fitness_terrain()
        new_positions = np.array([[5_000.0, 5_000.0]])
        result = fitness.coverage_fraction(terrain, new_positions, n_test_points=200, coverage_radius_m=0.001, seed=0)
        assert result < 0.01

    def test_seed_reproducibility(self):
        terrain = _fitness_terrain()
        new_positions = np.array([[5_000.0, 5_000.0]])
        a = fitness.coverage_fraction(terrain, new_positions, n_test_points=50, seed=3)
        b = fitness.coverage_fraction(terrain, new_positions, n_test_points=50, seed=3)
        assert a == pytest.approx(b)


class TestMeanInterference:
    def test_closer_new_bts_pair_has_more_interference(self):
        close_pair = np.array([[5_000.0, 5_000.0], [5_010.0, 5_000.0]])
        far_pair = np.array([[1_000.0, 1_000.0], [9_000.0, 9_000.0]])
        existing = np.array([[-100_000.0, -100_000.0]])  # far away, irrelevant to either pair
        close_interference = fitness.mean_interference(close_pair, existing, interference_radius_m=1_000.0)
        far_interference = fitness.mean_interference(far_pair, existing, interference_radius_m=1_000.0)
        assert close_interference > far_interference

    def test_no_neighbors_within_radius_is_zero(self):
        isolated = np.array([[5_000.0, 5_000.0]])
        existing = np.array([[-100_000.0, -100_000.0]])
        assert fitness.mean_interference(isolated, existing, interference_radius_m=1_000.0) == 0.0

    def test_empty_new_positions_is_zero(self):
        existing = np.array([[0.0, 0.0]])
        assert fitness.mean_interference(np.empty((0, 2)), existing) == 0.0


class TestMeanCostToInfrastructure:
    def test_new_bts_at_existing_bts_is_near_zero_cost(self):
        existing = np.array([[0.0, 0.0], [10_000.0, 10_000.0]])
        new_positions = np.array([[0.0, 0.0]])
        assert fitness.mean_cost_to_infrastructure(new_positions, existing) == pytest.approx(0.0, abs=1e-6)

    def test_far_new_bts_has_higher_cost(self):
        existing = np.array([[0.0, 0.0]])
        near = fitness.mean_cost_to_infrastructure(np.array([[100.0, 0.0]]), existing)
        far = fitness.mean_cost_to_infrastructure(np.array([[10_000.0, 0.0]]), existing)
        assert far > near

    def test_empty_new_positions_is_zero(self):
        existing = np.array([[0.0, 0.0]])
        assert fitness.mean_cost_to_infrastructure(np.empty((0, 2)), existing) == 0.0


class TestBtsCoverageFitnessComponents:
    def test_returns_expected_keys_with_finite_values(self):
        terrain = _fitness_terrain()
        components = fitness.bts_coverage_fitness_components([0.5, 0.5], terrain=terrain, n_test_points=50, seed=0)
        for key in ("coverage_pct", "interference", "cost_m", "f1", "f2", "f3"):
            assert key in components
            assert np.isfinite(components[key])

    def test_f1_is_negative_coverage_pct(self):
        terrain = _fitness_terrain()
        components = fitness.bts_coverage_fitness_components([0.5, 0.5], terrain=terrain, n_test_points=50, seed=0)
        assert components["f1"] == pytest.approx(-components["coverage_pct"])

    def test_seed_reproducibility(self):
        terrain = _fitness_terrain()
        a = fitness.bts_coverage_fitness_components([0.5, 0.5], terrain=terrain, n_test_points=50, seed=7)
        b = fitness.bts_coverage_fitness_components([0.5, 0.5], terrain=terrain, n_test_points=50, seed=7)
        assert a == b


class TestBtsCoverageFitness:
    def test_matches_components_f1_f2_f3(self):
        terrain = _fitness_terrain()
        components = fitness.bts_coverage_fitness_components([0.5, 0.5], terrain=terrain, n_test_points=50, seed=0)
        result = fitness.bts_coverage_fitness([0.5, 0.5], terrain=terrain, n_test_points=50, seed=0)
        assert result == (components["f1"], components["f2"], components["f3"])

    def test_returns_a_3_tuple(self):
        terrain = _fitness_terrain()
        result = fitness.bts_coverage_fitness([0.5, 0.5, 0.2, 0.8], terrain=terrain, n_test_points=50, seed=0)
        assert len(result) == 3


def _fake_aveyron_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({"lon": rng.uniform(2.0, 3.0, n), "lat": rng.uniform(44.0, 44.5, n), "mcc": 208})


class TestGetDefaultTerrain:
    def test_terrain_is_generated_once_and_cached(self, monkeypatch):
        fitness._terrain_cache = None
        calls = []

        def _fake_load(*args, **kwargs):
            calls.append(1)
            return _fake_aveyron_df()

        monkeypatch.setattr(opencellid_loader, "load_opencellid_csv", _fake_load)

        fitness.bts_coverage_fitness([0.5, 0.5], n_test_points=20, seed=0)
        fitness.bts_coverage_fitness([0.2, 0.8], n_test_points=20, seed=1)

        assert len(calls) == 1
        fitness._terrain_cache = None
