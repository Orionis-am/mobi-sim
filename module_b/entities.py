"""Core SMS network entities for Module B: SMSC, HLR, VLR, MSC.

Pure state-holding classes with a narrow interface (send, deliver, query_hlr,
register_ms, ...) matching the real GSM SMS architecture. No delay, loss, or
retry logic lives here — that's ``network_sim.py``'s job, layered on top,
the same separation ``codecs.py``/``fitness.py`` used in Module A.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Subscriber:
    """A GSM subscriber as known to the HLR: mapping between MSISDN (phone number) and IMSI."""

    msisdn: str
    imsi: str


class MessageStatus(Enum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    UNDELIVERABLE = "undeliverable"


@dataclass
class SmsMessage:
    """An SMS in flight through the simulated SMSC, identified by an auto-incrementing id."""

    id: int
    originator: str
    destination: str
    text: str
    status: MessageStatus = MessageStatus.QUEUED


class Hlr:
    """Home Location Register: the permanent subscriber database.

    Real HLRs store the subscriber's *home* network info and a pointer to
    whichever VLR currently serves them; the pointer itself is looked up via
    ``Vlr``/``Msc`` here rather than duplicated into the HLR record, so
    there's a single source of truth for "where is this subscriber now".
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, Subscriber] = {}

    def register_ms(self, msisdn: str, imsi: str) -> Subscriber:
        """Provision a new subscriber (mobile station) in the home network."""
        subscriber = Subscriber(msisdn=msisdn, imsi=imsi)
        self._subscribers[msisdn] = subscriber
        return subscriber

    def query_hlr(self, msisdn: str) -> Subscriber | None:
        """Look up a subscriber by MSISDN, or ``None`` if never registered."""
        return self._subscribers.get(msisdn)


class Vlr:
    """Visitor Location Register: which location area each attached subscriber currently sits in.

    A subscriber must be both known to the HLR *and* currently attached in
    the VLR to be reachable — modeling the real distinction between "has a
    subscription" and "phone is currently on and registered on the network".
    """

    def __init__(self) -> None:
        self._locations: dict[str, str] = {}

    def attach(self, msisdn: str, location_area: str) -> None:
        self._locations[msisdn] = location_area

    def detach(self, msisdn: str) -> None:
        self._locations.pop(msisdn, None)

    def is_attached(self, msisdn: str) -> bool:
        return msisdn in self._locations

    def location_of(self, msisdn: str) -> str | None:
        return self._locations.get(msisdn)


class Msc:
    """Mobile Switching Center: routes to a subscriber by consulting the HLR then the VLR.

    ``route`` is the single reachability check the rest of Module B builds
    on: a message can only be delivered if the destination is both a real
    subscriber (HLR) and currently attached to the network (VLR) — mirrors
    real GSM call/SMS routing, where the MSC never talks to a handset
    directly without going through both lookups.
    """

    def __init__(self, hlr: Hlr, vlr: Vlr) -> None:
        self.hlr = hlr
        self.vlr = vlr

    def route(self, msisdn: str) -> bool:
        return self.hlr.query_hlr(msisdn) is not None and self.vlr.is_attached(msisdn)


class Smsc:
    """Short Message Service Center: queues and attempts delivery of SMS messages.

    ``send`` only enqueues (mirrors the real TP-SUBMIT step: the SMSC has
    accepted the message from the origin, delivery to the destination is a
    separate, possibly-retried step) — ``deliver`` performs one delivery
    attempt via the MSC and updates the message's status accordingly.
    Retrying, delaying, or dropping messages is ``network_sim.py``'s
    responsibility; ``deliver`` here is a single, synchronous, all-or-nothing
    attempt with no notion of time.
    """

    def __init__(self, msc: Msc) -> None:
        self.msc = msc
        self._next_id = 1
        self.queue: list[SmsMessage] = []

    def send(self, originator: str, destination: str, text: str) -> SmsMessage:
        message = SmsMessage(id=self._next_id, originator=originator, destination=destination, text=text)
        self._next_id += 1
        self.queue.append(message)
        return message

    def deliver(self, message: SmsMessage) -> bool:
        """Attempt one delivery of ``message``, updating its status in place."""
        if self.msc.route(message.destination):
            message.status = MessageStatus.DELIVERED
            return True
        message.status = MessageStatus.UNDELIVERABLE
        return False
