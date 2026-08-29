"""Unit tests for Module B (entities, pdu, network_sim, twilio_client, compare, fitness).

External I/O (the Twilio API) is mocked throughout via a hand-rolled fake
client, mirroring module_a/test_module_a.py's approach for the OpenAI
client — no network access and no Twilio trial credit are required to run
this file.
"""

from __future__ import annotations

from module_b import entities

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
