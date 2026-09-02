"""Unit tests for Module D (model_e, stun_probe, session_sim, dashboard, correlation, fitness).

Built incrementally alongside each module_d file, mirroring test_module_c.py's conventions: one
Test<Function> class per function under test, hand-computed reference values where the formula is
simple enough to check by hand, seed-reproducibility tests for anything stochastic. External I/O
(real STUN UDP calls) will be mocked via a FakeStunSocket once stun_probe.py lands — no network
access required to run this file.
"""

from __future__ import annotations

import csv
import io
import socket
import struct

import matplotlib

matplotlib.use("Agg")  # headless: these tests must never pop up a GUI window

import numpy as np
import pytest
from matplotlib.figure import Figure
from rich.console import Console

from module_a import visualize as a_visualize
from module_d import correlation, dashboard, model_e, session_sim, stun_probe

# --- model_e -----------------------------------------------------------------


class TestDelayImpairment:
    def test_zero_delay_is_zero(self):
        assert model_e.delay_impairment(0.0) == 0.0

    def test_negative_delay_clamped_to_zero(self):
        assert model_e.delay_impairment(-50.0) == 0.0

    def test_below_threshold_is_linear_only(self):
        assert model_e.delay_impairment(100.0) == pytest.approx(0.024 * 100.0)

    def test_above_threshold_hand_computed(self):
        # d=200: Id = 0.024*200 + 0.11*(200-177.3) = 4.8 + 0.11*22.7 = 4.8 + 2.497 = 7.297
        assert model_e.delay_impairment(200.0) == pytest.approx(7.297)

    def test_monotonically_increasing_with_delay(self):
        assert model_e.delay_impairment(300.0) > model_e.delay_impairment(150.0) > model_e.delay_impairment(50.0)


class TestEffectiveEquipmentImpairment:
    def test_zero_loss_returns_base_ie(self):
        assert model_e.effective_equipment_impairment(20.0, 0.0) == pytest.approx(20.0)

    def test_loss_increases_impairment(self):
        base = model_e.effective_equipment_impairment(20.0, 0.0)
        degraded = model_e.effective_equipment_impairment(20.0, 5.0)
        assert degraded > base

    def test_hand_computed_value(self):
        # ie_base=20, loss_pct=5, bpl=10: Ie_eff = 20 + (95-20)*5/(5/10+2) = 20 + 75*5/2.5 = 20+150=170
        assert model_e.effective_equipment_impairment(20.0, 5.0, bpl=10.0) == pytest.approx(170.0)

    def test_monotonically_increasing_with_loss(self):
        low = model_e.effective_equipment_impairment(20.0, 1.0)
        mid = model_e.effective_equipment_impairment(20.0, 5.0)
        high = model_e.effective_equipment_impairment(20.0, 10.0)
        assert low < mid < high


class TestRFactor:
    def test_zero_delay_zero_loss_hand_computed(self):
        # opus: Ie=7, Id=0, Ie_eff=7 (loss=0) -> R = 93.2 - 0 - 0 - 7 + 10 = 96.2
        assert model_e.r_factor("opus", delay_ms=0.0, loss_pct=0.0) == pytest.approx(96.2)

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError):
            model_e.r_factor("mp3", delay_ms=0.0, loss_pct=0.0)

    def test_codec_name_case_insensitive(self):
        assert model_e.r_factor("OPUS", 0.0, 0.0) == model_e.r_factor("opus", 0.0, 0.0)

    def test_clamped_to_zero_under_extreme_conditions(self):
        assert model_e.r_factor("aac", delay_ms=5000.0, loss_pct=90.0) == 0.0

    def test_clamped_to_hundred_ceiling(self):
        assert model_e.r_factor("opus", delay_ms=0.0, loss_pct=0.0, is_impairment=-1000.0) == 100.0

    def test_worse_conditions_lower_r(self):
        good = model_e.r_factor("gsm", delay_ms=20.0, loss_pct=0.0)
        bad = model_e.r_factor("gsm", delay_ms=300.0, loss_pct=10.0)
        assert bad < good


class TestRToMos:
    def test_r_zero_gives_mos_one(self):
        assert model_e.r_to_mos(0.0) == 1.0

    def test_r_below_zero_gives_mos_one(self):
        assert model_e.r_to_mos(-10.0) == 1.0

    def test_r_hundred_gives_mos_four_point_five(self):
        assert model_e.r_to_mos(100.0) == 4.5

    def test_r_above_hundred_gives_mos_four_point_five(self):
        assert model_e.r_to_mos(150.0) == 4.5

    def test_hand_computed_value_at_r_93_2(self):
        # MOS = 1 + 0.035*93.2 + 93.2*33.2*6.8*7e-6 = 1 + 3.262 + 0.147286 = 4.409286
        assert model_e.r_to_mos(93.2) == pytest.approx(4.409286, abs=1e-4)

    def test_monotonically_increasing_with_r(self):
        assert model_e.r_to_mos(80.0) > model_e.r_to_mos(60.0) > model_e.r_to_mos(30.0)


class TestMosFromConditions:
    def test_matches_r_factor_then_r_to_mos(self):
        expected = model_e.r_to_mos(model_e.r_factor("gsm", 50.0, 2.0))
        assert model_e.mos_from_conditions("gsm", 50.0, 2.0) == pytest.approx(expected)

    def test_worse_conditions_give_lower_mos(self):
        good = model_e.mos_from_conditions("aac", delay_ms=20.0, loss_pct=0.0)
        bad = model_e.mos_from_conditions("aac", delay_ms=300.0, loss_pct=10.0)
        assert bad < good

    def test_codec_ordering_matches_ie_table_at_zero_loss(self):
        # Ie: opus=7 (best) < gsm=20 < aac=25 (worst) -> same ordering in MOS at equal delay/loss
        mos_opus = model_e.mos_from_conditions("opus", delay_ms=50.0, loss_pct=0.0)
        mos_gsm = model_e.mos_from_conditions("gsm", delay_ms=50.0, loss_pct=0.0)
        mos_aac = model_e.mos_from_conditions("aac", delay_ms=50.0, loss_pct=0.0)
        assert mos_opus > mos_gsm > mos_aac


# --- stun_probe ----------------------------------------------------------------


class FakeStunSocket:
    """Echoes a valid Binding Success Response for whatever request was last sent.

    `timeout_at_indices` makes the recvfrom call at those 0-based call indices raise
    socket.timeout instead, simulating loss without any real network I/O.
    """

    def __init__(self, timeout_at_indices: set[int] | None = None):
        self.timeout_at_indices = timeout_at_indices or set()
        self.sent: list[bytes] = []
        self.closed = False
        self._call_count = 0

    def settimeout(self, timeout_s):
        self.timeout_s = timeout_s

    def sendto(self, data, addr):
        self.sent.append(data)

    def recvfrom(self, bufsize):
        idx = self._call_count
        self._call_count += 1
        if idx in self.timeout_at_indices:
            raise socket.timeout("simulated timeout")
        txn_id = stun_probe.transaction_id(self.sent[-1])
        response = struct.pack(">HHI12s", stun_probe.BINDING_SUCCESS_RESPONSE, 0, stun_probe.MAGIC_COOKIE, txn_id)
        return response, ("127.0.0.1", stun_probe.DEFAULT_PORT)

    def close(self):
        self.closed = True


class TestBuildBindingRequest:
    def test_has_correct_header_fields(self):
        packet = stun_probe.build_binding_request()
        msg_type, length, cookie = struct.unpack(">HHI", packet[:8])
        assert msg_type == stun_probe.BINDING_REQUEST
        assert length == 0
        assert cookie == stun_probe.MAGIC_COOKIE
        assert len(packet) == 20

    def test_transaction_id_is_random_each_call(self):
        a = stun_probe.transaction_id(stun_probe.build_binding_request())
        b = stun_probe.transaction_id(stun_probe.build_binding_request())
        assert a != b


class TestParseBindingResponse:
    def test_valid_response_matches(self):
        txn_id = b"0" * 12
        data = struct.pack(">HHI12s", stun_probe.BINDING_SUCCESS_RESPONSE, 0, stun_probe.MAGIC_COOKIE, txn_id)
        assert stun_probe.parse_binding_response(data, txn_id) is True

    def test_mismatched_transaction_id_rejected(self):
        data = struct.pack(">HHI12s", stun_probe.BINDING_SUCCESS_RESPONSE, 0, stun_probe.MAGIC_COOKIE, b"0" * 12)
        assert stun_probe.parse_binding_response(data, b"1" * 12) is False

    def test_wrong_message_type_rejected(self):
        txn_id = b"0" * 12
        data = struct.pack(">HHI12s", stun_probe.BINDING_REQUEST, 0, stun_probe.MAGIC_COOKIE, txn_id)
        assert stun_probe.parse_binding_response(data, txn_id) is False

    def test_too_short_rejected(self):
        assert stun_probe.parse_binding_response(b"short", b"0" * 12) is False


class TestMeasureOneRtt:
    def test_valid_response_returns_nonnegative_float(self):
        rtt = stun_probe.measure_one_rtt(FakeStunSocket())
        assert isinstance(rtt, float)
        assert rtt >= 0.0

    def test_timeout_returns_none(self):
        rtt = stun_probe.measure_one_rtt(FakeStunSocket(timeout_at_indices={0}))
        assert rtt is None

    def test_garbage_response_returns_none(self):
        class GarbageSocket(FakeStunSocket):
            def recvfrom(self, bufsize):
                return b"not a stun packet", ("127.0.0.1", stun_probe.DEFAULT_PORT)

        rtt = stun_probe.measure_one_rtt(GarbageSocket())
        assert rtt is None


class TestMeasureRttJitter:
    def test_all_succeed_no_loss(self):
        stats = stun_probe.measure_rtt_jitter(n_measurements=5, sock=FakeStunSocket())
        assert len(stats.rtt_samples_ms) == 5
        assert stats.loss_rate == 0.0
        assert stats.rtt_mean_ms >= 0.0

    def test_partial_timeouts_counted_as_loss(self):
        stats = stun_probe.measure_rtt_jitter(n_measurements=5, sock=FakeStunSocket(timeout_at_indices={1, 3}))
        assert len(stats.rtt_samples_ms) == 3
        assert stats.loss_rate == pytest.approx(2 / 5)

    def test_all_timeouts_yields_nan_stats(self):
        stats = stun_probe.measure_rtt_jitter(n_measurements=3, sock=FakeStunSocket(timeout_at_indices={0, 1, 2}))
        assert stats.rtt_samples_ms == []
        assert stats.loss_rate == 1.0
        assert stats.rtt_mean_ms != stats.rtt_mean_ms  # NaN

    def test_injected_socket_is_not_closed(self):
        fake = FakeStunSocket()
        stun_probe.measure_rtt_jitter(n_measurements=2, sock=fake)
        assert fake.closed is False

    def test_creates_and_closes_own_socket_when_none_given(self, monkeypatch):
        fake = FakeStunSocket()
        monkeypatch.setattr(stun_probe.socket, "socket", lambda *a, **kw: fake)
        stats = stun_probe.measure_rtt_jitter(n_measurements=2, sock=None)
        assert len(stats.rtt_samples_ms) == 2
        assert fake.closed is True


# --- session_sim ---------------------------------------------------------------


class TestSimulateSession:
    def test_sufficient_bandwidth_adds_no_congestion_delay(self):
        session = session_sim.simulate_session("opus", bandwidth_kbps=64, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert session.delay_ms == pytest.approx(40.0 / 2.0 + 5.0)

    def test_full_deficit_adds_more_loss_than_partial_deficit(self):
        full = session_sim.simulate_session("gsm", bandwidth_kbps=0, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        partial = session_sim.simulate_session("gsm", bandwidth_kbps=16, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        sufficient = session_sim.simulate_session("gsm", bandwidth_kbps=32, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert full.loss_pct > partial.loss_pct > sufficient.loss_pct

    def test_loss_clipped_to_hundred(self):
        session = session_sim.simulate_session("gsm", bandwidth_kbps=-1000, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert session.loss_pct <= 100.0

    def test_seed_reproducibility(self):
        a = session_sim.simulate_session("aac", bandwidth_kbps=10, required_bandwidth_kbps=32, rtt_ms=60.0, jitter_ms=8.0, seed=42)
        b = session_sim.simulate_session("aac", bandwidth_kbps=10, required_bandwidth_kbps=32, rtt_ms=60.0, jitter_ms=8.0, seed=42)
        assert a == b

    def test_worse_deficit_gives_lower_mos(self):
        starved = session_sim.simulate_session("opus", bandwidth_kbps=0, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        healthy = session_sim.simulate_session("opus", bandwidth_kbps=32, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert starved.mos < healthy.mos

    def test_r_and_mos_consistent_with_model_e(self):
        session = session_sim.simulate_session("gsm", bandwidth_kbps=32, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        expected_r = model_e.r_factor("gsm", session.delay_ms, session.loss_pct)
        assert session.r_factor == pytest.approx(expected_r)
        assert session.mos == pytest.approx(model_e.r_to_mos(expected_r))

    def test_codec_field_preserved(self):
        session = session_sim.simulate_session("aac", bandwidth_kbps=32, required_bandwidth_kbps=32, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert session.codec == "aac"

    def test_zero_required_bandwidth_means_no_deficit(self):
        session = session_sim.simulate_session("opus", bandwidth_kbps=0, required_bandwidth_kbps=0, rtt_ms=40.0, jitter_ms=5.0, seed=1)
        assert session.delay_ms == pytest.approx(40.0 / 2.0 + 5.0)


# --- dashboard -------------------------------------------------------------------

_silent_console = lambda: Console(file=io.StringIO())


def _render(renderable) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=120).print(renderable)
    return buffer.getvalue()


class TestBuildTable:
    def test_has_expected_columns(self):
        table = dashboard._build_table(50.0, 2.0, 0.0, 4.1, "opus")
        assert [c.header for c in table.columns] == ["RTT (ms)", "Gigue (ms)", "Perte (%)", "MOS", "Codec"]

    def test_renders_expected_values(self):
        rendered = _render(dashboard._build_table(50.0, 2.0, 1.5, 4.1, "opus"))
        assert "50.0" in rendered
        assert "opus" in rendered

    def test_nan_rtt_displayed_as_dash(self):
        rendered = _render(dashboard._build_table(float("nan"), 0.0, 0.0, 1.0, "opus"))
        assert "nan" not in rendered.lower()


class TestRunDashboard:
    def test_writes_one_csv_row_per_iteration(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        result_path = dashboard.run_dashboard(
            n_iterations=3, csv_path=csv_path, sock=FakeStunSocket(), sleep=lambda s: None, console=_silent_console()
        )
        assert result_path == csv_path
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert list(rows[0].keys()) == dashboard.CSV_FIELDS

    def test_creates_output_dir_if_missing(self, tmp_path):
        csv_path = tmp_path / "nested" / "out.csv"
        dashboard.run_dashboard(n_iterations=1, csv_path=csv_path, sock=FakeStunSocket(), sleep=lambda s: None, console=_silent_console())
        assert csv_path.exists()

    def test_injected_socket_not_closed(self, tmp_path):
        fake = FakeStunSocket()
        dashboard.run_dashboard(n_iterations=2, csv_path=tmp_path / "out.csv", sock=fake, sleep=lambda s: None, console=_silent_console())
        assert fake.closed is False

    def test_closes_own_socket_when_none_given(self, tmp_path, monkeypatch):
        fake = FakeStunSocket()
        monkeypatch.setattr(dashboard.socket, "socket", lambda *a, **kw: fake)
        dashboard.run_dashboard(n_iterations=2, csv_path=tmp_path / "out.csv", sock=None, sleep=lambda s: None, console=_silent_console())
        assert fake.closed is True

    def test_partial_loss_reflected_in_later_rows(self, tmp_path):
        csv_path = tmp_path / "out.csv"
        dashboard.run_dashboard(
            n_iterations=3, csv_path=csv_path, sock=FakeStunSocket(timeout_at_indices={0}), sleep=lambda s: None, console=_silent_console()
        )
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["loss_pct"]) > 0.0

    def test_sleeps_between_but_not_after_last_iteration(self, tmp_path):
        sleep_calls = []
        dashboard.run_dashboard(
            n_iterations=3, csv_path=tmp_path / "out.csv", sock=FakeStunSocket(), sleep=sleep_calls.append, console=_silent_console()
        )
        assert len(sleep_calls) == 2


# --- correlation -----------------------------------------------------------------


class TestBuildCorrelationTable:
    def test_merges_pesq_wer_mushra_and_mos(self):
        pesq = {"aac": 4.0, "gsm": 3.5}
        wer = {"aac": 0.1, "gsm": 0.2}
        mushra = {"aac": {"english": {"mean": 40.0}}, "gsm": {"english": {"mean": 50.0}}}
        mos = {"aac": 3.0, "gsm": 3.2}
        table = correlation.build_correlation_table(pesq, wer, mushra, mos)
        assert table == {
            "aac": {"pesq": 4.0, "wer": 0.1, "mushra": 40.0, "mos": 3.0},
            "gsm": {"pesq": 3.5, "wer": 0.2, "mushra": 50.0, "mos": 3.2},
        }


class TestPlotCorrelationMatrixReuse:
    def test_is_module_a_visualize_function(self):
        assert correlation.plot_correlation_matrix is a_visualize.plot_correlation_matrix


class TestPlotMosVsDelay:
    def test_returns_figure_with_one_line_per_codec(self):
        fig = correlation.plot_mos_vs_delay(codecs=("aac", "opus"), delay_range_ms=np.linspace(0, 300, 10))
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) >= 2  # 2 codec lines + 2 threshold lines

    def test_mos_decreases_with_delay(self):
        fig = correlation.plot_mos_vs_delay(codecs=("opus",), delay_range_ms=np.linspace(0, 300, 10))
        line = fig.axes[0].lines[0]
        y = line.get_ydata()
        assert y[0] > y[-1]


class TestPlotMosVsLoss:
    def test_returns_figure_with_one_line_per_codec(self):
        fig = correlation.plot_mos_vs_loss(codecs=("aac", "opus"), loss_range_pct=np.linspace(0, 20, 10))
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) >= 2

    def test_mos_decreases_with_loss(self):
        fig = correlation.plot_mos_vs_loss(codecs=("gsm",), loss_range_pct=np.linspace(0, 20, 10))
        line = fig.axes[0].lines[0]
        y = line.get_ydata()
        assert y[0] > y[-1]
