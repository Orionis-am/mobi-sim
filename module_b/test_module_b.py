"""Unit tests for Module B (entities, pdu, network_sim, twilio_client, compare, fitness).

External I/O (the Twilio API) is mocked throughout via a hand-rolled fake
client, mirroring module_a/test_module_a.py's approach for the OpenAI
client — no network access and no Twilio trial credit are required to run
this file.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from module_b import compare, entities, network_sim, pdu, twilio_client

# --- entities ---------------------------------------------------------------


class TestHlr:
    def test_register_then_query_returns_same_subscriber(self):
        hlr = entities.Hlr()
        registered = hlr.register_ms("+15551234567", imsi="001010000000001")
        assert hlr.query_hlr("+15551234567") == registered

    def test_unknown_msisdn_returns_none(self):
        hlr = entities.Hlr()
        assert hlr.query_hlr("+15550000000") is None

    def test_re_registering_overwrites_previous_record(self):
        hlr = entities.Hlr()
        hlr.register_ms("+15551234567", imsi="001010000000001")
        second = hlr.register_ms("+15551234567", imsi="001010000000002")
        assert hlr.query_hlr("+15551234567") == second


class TestVlr:
    def test_not_attached_by_default(self):
        vlr = entities.Vlr()
        assert vlr.is_attached("+15551234567") is False
        assert vlr.location_of("+15551234567") is None

    def test_attach_then_detach(self):
        vlr = entities.Vlr()
        vlr.attach("+15551234567", "LAC-42")
        assert vlr.is_attached("+15551234567") is True
        assert vlr.location_of("+15551234567") == "LAC-42"

        vlr.detach("+15551234567")
        assert vlr.is_attached("+15551234567") is False

    def test_detach_unknown_subscriber_is_a_noop(self):
        vlr = entities.Vlr()
        vlr.detach("+15550000000")  # must not raise


class TestMsc:
    def test_route_requires_both_hlr_registration_and_vlr_attachment(self):
        hlr, vlr = entities.Hlr(), entities.Vlr()
        msc = entities.Msc(hlr, vlr)

        assert msc.route("+15551234567") is False  # neither

        hlr.register_ms("+15551234567", imsi="001010000000001")
        assert msc.route("+15551234567") is False  # registered but not attached

        vlr.attach("+15551234567", "LAC-42")
        assert msc.route("+15551234567") is True  # both

    def test_attached_but_never_registered_is_not_routable(self):
        hlr, vlr = entities.Hlr(), entities.Vlr()
        msc = entities.Msc(hlr, vlr)
        vlr.attach("+15551234567", "LAC-42")
        assert msc.route("+15551234567") is False


class TestSmsc:
    def _reachable_msc(self, msisdn: str = "+15551234567") -> entities.Msc:
        hlr, vlr = entities.Hlr(), entities.Vlr()
        hlr.register_ms(msisdn, imsi="001010000000001")
        vlr.attach(msisdn, "LAC-42")
        return entities.Msc(hlr, vlr)

    def test_send_enqueues_a_queued_message_with_incrementing_ids(self):
        smsc = entities.Smsc(self._reachable_msc())
        first = smsc.send("+15550000000", "+15551234567", "hello")
        second = smsc.send("+15550000000", "+15551234567", "world")

        assert first.id == 1 and second.id == 2
        assert first.status == entities.MessageStatus.QUEUED
        assert smsc.queue == [first, second]

    def test_deliver_succeeds_when_destination_is_routable(self):
        smsc = entities.Smsc(self._reachable_msc())
        message = smsc.send("+15550000000", "+15551234567", "hello")

        assert smsc.deliver(message) is True
        assert message.status == entities.MessageStatus.DELIVERED

    def test_deliver_fails_when_destination_is_not_routable(self):
        unreachable_msc = entities.Msc(entities.Hlr(), entities.Vlr())
        smsc = entities.Smsc(unreachable_msc)
        message = smsc.send("+15550000000", "+15551234567", "hello")

        assert smsc.deliver(message) is False
        assert message.status == entities.MessageStatus.UNDELIVERABLE


# --- pdu ---------------------------------------------------------------


class TestGsm7Alphabet:
    def test_round_trips_basic_charset_text(self):
        text = "Hello, World! 123 :-)"
        assert pdu.gsm7_decode(pdu.gsm7_encode(text)) == text

    def test_unsupported_character_raises(self):
        with pytest.raises(ValueError):
            pdu.gsm7_encode("price: 5€")

    def test_at_sign_is_index_zero_not_ascii_nul(self):
        # The one-character difference that makes GSM7 not "just ASCII":
        # index 0x00 is '@', unlike ASCII's NUL.
        assert pdu.gsm7_encode("@") == [0x00]


class TestSeptetPacking:
    def test_two_septets_pack_as_hand_computed(self):
        # 'A'=0x41=0b1000001, 'B'=0x42=0b1000010. Concatenating their 7 bits
        # LSB-first and slicing into two padded octets gives 0x41, 0x21 —
        # verified by hand in REPORT.md.
        assert pdu.pack_septets([0x41, 0x42]) == bytes([0x41, 0x21])

    @pytest.mark.parametrize("n_chars", [1, 7, 8, 9, 16, 17])
    def test_round_trip_across_octet_boundary(self, n_chars):
        septets = [(0x41 + i) % 0x5B for i in range(n_chars)]  # stays within 'A'-'Z'-ish range
        packed = pdu.pack_septets(septets)
        assert pdu.unpack_septets(packed, n_chars) == septets


class TestAddressField:
    @pytest.mark.parametrize("number", ["+33612345678", "0612345678", "+123", "12345"])
    def test_round_trip(self, number):
        encoded = pdu.encode_address_field(number)
        decoded, offset = pdu.decode_address_field(encoded, 0)
        assert decoded == number
        assert offset == len(encoded)


class TestSmscField:
    def test_none_encodes_as_single_zero_octet(self):
        assert pdu.encode_smsc_field(None) == bytes([0x00])
        decoded, offset = pdu.decode_smsc_field(bytes([0x00]), 0)
        assert decoded is None and offset == 1

    @pytest.mark.parametrize("smsc", ["+33689004000", "+1234567"])
    def test_round_trip(self, smsc):
        encoded = pdu.encode_smsc_field(smsc)
        decoded, offset = pdu.decode_smsc_field(encoded, 0)
        assert decoded == smsc
        assert offset == len(encoded)


class TestValidityPeriod:
    @pytest.mark.parametrize("hours", [5 / 60, 1.0, 24.0, 72.0, 336.0, 24.0 * 7 * 10])
    def test_round_trip_on_grid_points(self, hours):
        assert pdu.vp_to_hours(pdu.hours_to_vp(hours)) == pytest.approx(hours)

    def test_out_of_range_is_clamped_not_raised(self):
        assert pdu.hours_to_vp(100_000) == 255  # far beyond the max representable 63 weeks
        assert pdu.hours_to_vp(0) == 0


class TestScts:
    def test_round_trip(self):
        dt = datetime(2026, 8, 29, 14, 5, 9)
        assert pdu.decode_scts(pdu.encode_scts(dt)) == dt


class TestSubmitPdu:
    def test_round_trip_minimal(self):
        original = pdu.SubmitPdu(destination="+33612345678", text="Hi")
        decoded = pdu.decode_submit(pdu.encode_submit(original))
        assert decoded == original

    def test_round_trip_with_smsc_and_validity_period(self):
        original = pdu.SubmitPdu(
            destination="0612345678",
            text="The quick brown fox jumps over the lazy dog",
            message_reference=7,
            validity_period_hours=24.0,
            smsc="+33689004000",
        )
        decoded = pdu.decode_submit(pdu.encode_submit(original))
        assert decoded == original

    def test_hex_round_trip(self):
        original = pdu.SubmitPdu(destination="+33612345678", text="Hi")
        hex_pdu = pdu.encode_submit_hex(original)
        assert hex_pdu == hex_pdu.upper()
        assert pdu.decode_submit_hex(hex_pdu) == original

    def test_decode_rejects_wrong_mti(self):
        deliver_bytes = pdu.encode_deliver(pdu.DeliverPdu(originator="+33612345678", text="hi", timestamp=datetime(2026, 1, 1)))
        with pytest.raises(ValueError):
            pdu.decode_submit(deliver_bytes)


class TestDeliverPdu:
    def test_round_trip(self):
        original = pdu.DeliverPdu(
            originator="+33612345678",
            text="hello there",
            timestamp=datetime(2026, 8, 29, 14, 5, 9),
            smsc="+33689004000",
        )
        decoded = pdu.decode_deliver(pdu.encode_deliver(original))
        assert decoded == original

    def test_decode_rejects_wrong_mti(self):
        submit_bytes = pdu.encode_submit(pdu.SubmitPdu(destination="+33612345678", text="hi"))
        with pytest.raises(ValueError):
            pdu.decode_deliver(submit_bytes)

    def test_hex_round_trip(self):
        original = pdu.DeliverPdu(originator="+33612345678", text="hi", timestamp=datetime(2026, 8, 29, 14, 5, 9))
        hex_pdu = pdu.encode_deliver_hex(original)
        assert pdu.decode_deliver_hex(hex_pdu) == original


# --- network_sim ---------------------------------------------------------


def _reachable_smsc(destination: str = "+15551234567") -> entities.Smsc:
    hlr, vlr = entities.Hlr(), entities.Vlr()
    hlr.register_ms(destination, imsi="001010000000001")
    vlr.attach(destination, "LAC-1")
    return entities.Smsc(entities.Msc(hlr, vlr))


class TestSimulateDelivery:
    def test_reachable_and_no_loss_delivers_on_first_attempt(self):
        smsc = _reachable_smsc()
        message = smsc.send("+15550000000", "+15551234567", "hi")
        rng = np.random.default_rng(0)

        outcome = network_sim.simulate_delivery(smsc, message, rng, loss_probability=0.0)

        assert outcome.delivered is True
        assert outcome.attempts == 1
        assert message.status == entities.MessageStatus.DELIVERED

    def test_unreachable_destination_expires_after_retry_window(self):
        unreachable_smsc = entities.Smsc(entities.Msc(entities.Hlr(), entities.Vlr()))
        message = unreachable_smsc.send("+15550000000", "+15551234567", "hi")
        rng = np.random.default_rng(0)

        outcome = network_sim.simulate_delivery(
            unreachable_smsc, message, rng,
            mean_delay_s=0.01, retry_backoff_s=1.0, backoff_factor=2.0, max_retry_window_s=5.0,
        )

        assert outcome.delivered is False
        assert outcome.attempts > 1
        assert outcome.delay_s > 5.0
        assert message.status == entities.MessageStatus.UNDELIVERABLE

    def test_seed_reproducibility(self):
        def run():
            smsc = _reachable_smsc()
            message = smsc.send("+15550000000", "+15551234567", "hi")
            return network_sim.simulate_delivery(smsc, message, np.random.default_rng(42))

        a, b = run(), run()
        assert a == b


class TestSimulateBatchDelivery:
    def test_always_reachable_and_no_loss_gives_full_delivery_on_first_attempt(self):
        stats = network_sim.simulate_batch_delivery(20, reachable_probability=1.0, loss_probability=0.0, seed=0)
        assert stats.delivery_rate == pytest.approx(1.0)
        assert stats.mean_attempts == pytest.approx(1.0)

    def test_never_reachable_gives_zero_delivery_rate(self):
        stats = network_sim.simulate_batch_delivery(
            5, reachable_probability=0.0, seed=0,
            mean_delay_s=0.01, retry_backoff_s=1.0, backoff_factor=2.0, max_retry_window_s=5.0,
        )
        assert stats.delivery_rate == pytest.approx(0.0)

    def test_seed_reproducibility(self):
        a = network_sim.simulate_batch_delivery(10, reachable_probability=0.7, seed=7)
        b = network_sim.simulate_batch_delivery(10, reachable_probability=0.7, seed=7)
        assert a == b


class TestSimulateOverload:
    def test_light_load_has_negligible_loss(self):
        result = network_sim.simulate_overload(
            arrival_rate_msgs_per_s=5.0, duration_s=2.0, throughput_msgs_per_s=100.0, queue_capacity=500, seed=0,
        )
        assert result.loss_rate == pytest.approx(0.0)

    def test_sustained_overload_causes_loss(self):
        result = network_sim.simulate_overload(
            arrival_rate_msgs_per_s=500.0, duration_s=2.0, throughput_msgs_per_s=50.0, queue_capacity=20, seed=0,
        )
        assert result.loss_rate > 0.0

    def test_arrival_accounting_is_conserved(self):
        result = network_sim.simulate_overload(
            arrival_rate_msgs_per_s=80.0, duration_s=2.0, throughput_msgs_per_s=50.0, queue_capacity=30, seed=0,
        )
        assert result.n_delivered + result.n_dropped + result.n_still_queued == result.n_arrived

    def test_seed_reproducibility(self):
        kwargs = dict(arrival_rate_msgs_per_s=80.0, duration_s=2.0, throughput_msgs_per_s=50.0, queue_capacity=30, seed=3)
        assert network_sim.simulate_overload(**kwargs) == network_sim.simulate_overload(**kwargs)


class TestSweepOverload:
    def test_returns_one_result_per_arrival_rate_in_order(self):
        rates = [5.0, 50.0, 500.0]
        results = network_sim.sweep_overload(rates, duration_s=1.0, throughput_msgs_per_s=50.0, queue_capacity=20, seed=0)
        assert [r.arrival_rate_msgs_per_s for r in results] == rates

    def test_heavy_load_loses_far_more_than_light_load(self):
        results = network_sim.sweep_overload(
            [5.0, 500.0], duration_s=2.0, throughput_msgs_per_s=50.0, queue_capacity=20, seed=0,
        )
        light, heavy = results
        assert heavy.loss_rate > light.loss_rate


# --- twilio_client -------------------------------------------------------


class FakeTwilioMessage:
    def __init__(self, sid, status, to, from_, body):
        self.sid = sid
        self.status = status
        self.to = to
        self.from_ = from_
        self.body = body
        self.date_created = None
        self.date_updated = None
        self.error_code = None


class FakeMessageContext:
    def __init__(self, message: FakeTwilioMessage):
        self._message = message

    def fetch(self) -> FakeTwilioMessage:
        return self._message


class FakeMessagesResource:
    def __init__(self):
        self._messages: dict[str, FakeTwilioMessage] = {}
        self._next_id = 1
        self.create_calls: list[dict] = []

    def create(self, to: str, from_: str, body: str) -> FakeTwilioMessage:
        sid = f"SM{self._next_id:032d}"
        self._next_id += 1
        message = FakeTwilioMessage(sid=sid, status="queued", to=to, from_=from_, body=body)
        self._messages[sid] = message
        self.create_calls.append({"to": to, "from_": from_, "body": body})
        return message

    def __call__(self, sid: str) -> FakeMessageContext:
        return FakeMessageContext(self._messages[sid])


class FakeTwilioClient:
    def __init__(self):
        self.messages = FakeMessagesResource()


class TestSendSms:
    def test_sends_with_explicit_sender_and_returns_result(self):
        client = FakeTwilioClient()
        result = twilio_client.send_sms("+15551234567", "hello", from_="+15550000000", client=client)

        assert result.status == "queued"
        assert result.to == "+15551234567"
        assert result.from_ == "+15550000000"
        assert client.messages.create_calls == [{"to": "+15551234567", "from_": "+15550000000", "body": "hello"}]

    def test_falls_back_to_env_sender(self, monkeypatch):
        monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15559999999")
        client = FakeTwilioClient()
        twilio_client.send_sms("+15551234567", "hi", client=client)
        assert client.messages.create_calls[0]["from_"] == "+15559999999"

    def test_missing_sender_raises(self, monkeypatch):
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
        with pytest.raises(RuntimeError):
            twilio_client.send_sms("+15551234567", "hi", client=FakeTwilioClient())


class TestGetDeliveryStatus:
    def test_returns_status_from_fetch(self):
        client = FakeTwilioClient()
        sent = twilio_client.send_sms("+15551234567", "hi", from_="+15550000000", client=client)
        client.messages._messages[sent.sid].status = "delivered"

        status = twilio_client.get_delivery_status(sent.sid, client=client)
        assert status.sid == sent.sid
        assert status.status == "delivered"


class TestClient:
    def test_missing_credentials_raises(self, monkeypatch):
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
        with pytest.raises(RuntimeError):
            twilio_client._client()

    def test_built_from_env_vars_when_present(self, monkeypatch):
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACtestsid")
        monkeypatch.setenv("TWILIO_AUTH_TOKEN", "testtoken")
        client = twilio_client._client()
        assert client.username == "ACtestsid"
        assert client.password == "testtoken"


class TestHandleStatusWebhook:
    def test_parses_delivered_payload(self):
        payload = {"MessageSid": "SM123", "MessageStatus": "delivered", "To": "+15551234567", "From": "+15550000000"}
        status = twilio_client.handle_status_webhook(payload)
        assert status.sid == "SM123"
        assert status.status == "delivered"
        assert status.error_code is None

    def test_parses_error_code_as_int(self):
        payload = {"MessageSid": "SM123", "MessageStatus": "failed", "ErrorCode": "30006"}
        status = twilio_client.handle_status_webhook(payload)
        assert status.error_code == 30006


# --- compare ---------------------------------------------------------------


class _FakeClock:
    """now()/sleep() pair where sleep() advances the clock — no real waiting in tests."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class TestMeasureRealDelivery:
    def test_polls_until_terminal_status_and_measures_latencies(self):
        client = FakeTwilioClient()
        clock = _FakeClock()
        first_sid = "SM" + "1".rjust(32, "0")

        def sleep_and_mark_delivered(seconds: float) -> None:
            clock.sleep(seconds)
            client.messages._messages[first_sid].status = "delivered"

        sample = compare.measure_real_delivery(
            "+15551234567", "hi", from_="+15550000000", client=client,
            poll_interval_s=2.0, poll_timeout_s=60.0,
            sleep=sleep_and_mark_delivered, now=clock.now,
        )

        assert sample.accept_latency_s == pytest.approx(0.0)
        assert sample.delivery_latency_s == pytest.approx(2.0)
        assert sample.delivered is True
        assert sample.final_status == "delivered"

    def test_gives_up_at_timeout_without_terminal_status(self):
        client = FakeTwilioClient()
        clock = _FakeClock()

        sample = compare.measure_real_delivery(
            "+15551234567", "hi", from_="+15550000000", client=client,
            poll_interval_s=2.0, poll_timeout_s=3.0,
            sleep=clock.sleep, now=clock.now,
        )

        assert sample.delivery_latency_s is None
        assert sample.delivered is False
        assert sample.final_status == "queued"


class TestSummarizeRealSamples:
    def test_empty_list_gives_zeroed_stats(self):
        stats = compare.summarize_real_samples([])
        assert stats == compare.RealStats(n_samples=0, delivery_rate=0.0, mean_accept_latency_s=0.0, mean_delivery_latency_s=None)

    def test_aggregates_across_samples(self):
        samples = [
            compare.RealDeliverySample(accept_latency_s=1.0, delivery_latency_s=4.0, delivered=True, final_status="delivered"),
            compare.RealDeliverySample(accept_latency_s=3.0, delivery_latency_s=None, delivered=False, final_status="queued"),
        ]
        stats = compare.summarize_real_samples(samples)
        assert stats.n_samples == 2
        assert stats.delivery_rate == pytest.approx(0.5)
        assert stats.mean_accept_latency_s == pytest.approx(2.0)
        assert stats.mean_delivery_latency_s == pytest.approx(4.0)  # only the known one counts


class TestBuildComparisonTable:
    def test_shapes_simulated_and_real_side_by_side(self):
        simulated = network_sim.BatchDeliveryStats(n_messages=100, delivery_rate=0.9, mean_delay_s=3.0, mean_attempts=1.2)
        real = compare.RealStats(n_samples=5, delivery_rate=1.0, mean_accept_latency_s=0.4, mean_delivery_latency_s=6.0)

        table = compare.build_comparison_table(simulated, real)

        assert table["simulated"] == {"mean_delay_s": 3.0, "delivery_rate": 0.9}
        assert table["real"] == {"mean_accept_latency_s": 0.4, "mean_delivery_latency_s": 6.0, "delivery_rate": 1.0, "n_samples": 5}
