"""Unit tests for Module C (opencellid_loader, terrain_sim, cell_id, toa, wifi_fp, ipinfo_client,
lbs_poi, map_viz, fitness).

External I/O (ipinfo.io, Nominatim) is mocked throughout via hand-rolled fakes, mirroring
module_b/test_module_b.py's FakeTwilioClient approach — no network access required to run this
file. Tests against the real docs/208.csv (392k rows) live outside pytest, exercised manually —
these tests use small synthetic CSVs instead, for speed.
"""

from __future__ import annotations

from module_c import opencellid_loader

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
