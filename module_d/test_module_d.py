"""Unit tests for Module D (model_e, stun_probe, session_sim, dashboard, correlation, fitness).

Built incrementally alongside each module_d file, mirroring test_module_c.py's conventions: one
Test<Function> class per function under test, hand-computed reference values where the formula is
simple enough to check by hand, seed-reproducibility tests for anything stochastic. External I/O
(real STUN UDP calls) will be mocked via a FakeStunSocket once stun_probe.py lands — no network
access required to run this file.
"""

from __future__ import annotations

import pytest

from module_d import model_e

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
