"""Unit tests for Module B (entities, pdu, network_sim, twilio_client, compare, fitness).

External I/O (the Twilio API) is mocked throughout via a hand-rolled fake
client, mirroring module_a/test_module_a.py's approach for the OpenAI
client — no network access and no Twilio trial credit are required to run
this file.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from module_b import entities, pdu

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
