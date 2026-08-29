"""GSM SMS PDU encoding/decoding for Module B (3GPP TS 23.038 / TS 23.040).

No hex test vectors from the course were available in this repo to validate
against (see REPORT.md) — correctness here rests on round-trip tests
(``decode(encode(x)) == x``) plus one hand-verified worked example of the
7-bit packing (the trickiest part) documented in REPORT.md, rather than on
a "known good" reference PDU.

Scope, deliberately limited to what the simulation needs:
- GSM 7-bit default alphabet only (no extension/escape table: `€`, `[`, `{`,
  etc. raise ``ValueError``; no UCS-2).
- No User Data Header / concatenated (multipart) SMS.
- SMS-SUBMIT uses the relative Validity Period format only (no absolute/
  enhanced VP).
- SMS-DELIVER's timestamp always encodes a zero (UTC) timezone offset —
  irrelevant for a local simulation that doesn't cross real time zones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# --- GSM 7-bit default alphabet (3GPP TS 23.038 §6.2.1) --------------------
#
# Indices 0x20-0x3F, 0x41-0x5A and 0x61-0x7A coincide with ASCII; the rest
# (accented letters, currency signs, Greek letters used in physics/maths
# notation) do not, which is exactly why this needs an explicit table rather
# than treating GSM7 as "basically ASCII". 0x1B is the extension-table
# escape code, not a printable character, and is intentionally left out of
# the table below (unsupported, see module docstring).

_GSM7_CHARS: dict[int, str] = {
    0x00: "@", 0x01: "£", 0x02: "$", 0x03: "¥", 0x04: "è", 0x05: "é", 0x06: "ù", 0x07: "ì",
    0x08: "ò", 0x09: "Ç", 0x0A: "\n", 0x0B: "Ø", 0x0C: "ø", 0x0D: "\r", 0x0E: "Å", 0x0F: "å",
    0x10: "Δ", 0x11: "_", 0x12: "Φ", 0x13: "Γ", 0x14: "Λ", 0x15: "Ω", 0x16: "Π", 0x17: "Ψ",
    0x18: "Σ", 0x19: "Θ", 0x1A: "Ξ",
    0x1C: "Æ", 0x1D: "æ", 0x1E: "ß", 0x1F: "É",
    0x20: " ", 0x21: "!", 0x22: '"', 0x23: "#", 0x24: "¤", 0x25: "%", 0x26: "&", 0x27: "'",
    0x28: "(", 0x29: ")", 0x2A: "*", 0x2B: "+", 0x2C: ",", 0x2D: "-", 0x2E: ".", 0x2F: "/",
    **{0x30 + i: str(i) for i in range(10)},
    0x3A: ":", 0x3B: ";", 0x3C: "<", 0x3D: "=", 0x3E: ">", 0x3F: "?",
    0x40: "¡",
    **{0x41 + i: chr(ord("A") + i) for i in range(26)},
    0x5B: "Ä", 0x5C: "Ö", 0x5D: "Ñ", 0x5E: "Ü", 0x5F: "§",
    0x60: "¿",
    **{0x61 + i: chr(ord("a") + i) for i in range(26)},
    0x7B: "ä", 0x7C: "ö", 0x7D: "ñ", 0x7E: "ü", 0x7F: "à",
}
_CHAR_TO_GSM7: dict[str, int] = {char: index for index, char in _GSM7_CHARS.items()}


def gsm7_encode(text: str) -> list[int]:
    """Map each character of ``text`` to its GSM 7-bit default-alphabet septet index."""
    try:
        return [_CHAR_TO_GSM7[c] for c in text]
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is not in the GSM 7-bit default alphabet (extension table not supported)") from exc


def gsm7_decode(septets: list[int]) -> str:
    """Inverse of ``gsm7_encode``."""
    try:
        return "".join(_GSM7_CHARS[s] for s in septets)
    except KeyError as exc:
        raise ValueError(f"septet index {exc.args[0]:#04x} is not a valid GSM 7-bit default-alphabet character") from exc


# --- 7-bit packing -----------------------------------------------------------
#
# Septets are concatenated into a single bitstream, least-significant-bit
# first, then sliced into 8-bit octets (also LSB-first) — every 8 septets
# (56 bits) packs into exactly 7 octets. Because the SMS-SUBMIT/DELIVER
# TP-UDL field carries the exact septet count explicitly, decoding never
# needs the padding-bit disambiguation trick some GSM7 write-ups mention for
# septet counts ≡ 7 (mod 8): unpacking simply reads back UDL septets and
# ignores any leftover padding bits in the final octet.


def pack_septets(septets: list[int]) -> bytes:
    value = 0
    for i, septet in enumerate(septets):
        value |= (septet & 0x7F) << (7 * i)
    n_bytes = (7 * len(septets) + 7) // 8
    return value.to_bytes(n_bytes, byteorder="little") if n_bytes else b""


def unpack_septets(data: bytes, septet_count: int) -> list[int]:
    value = int.from_bytes(data, byteorder="little")
    return [(value >> (7 * i)) & 0x7F for i in range(septet_count)]


# --- BCD-packed digits (addresses, timestamp fields) ------------------------
#
# Digits are packed two per octet, low nibble first (the "semi-octet swap"
# GSM PDU write-ups refer to): digit[0] goes in the low nibble, digit[1] in
# the high nibble of the first octet, and so on. An odd digit count is
# padded with a trailing 0xF nibble.


def _encode_bcd_digits(digits: str) -> bytes:
    padded = digits + "F" if len(digits) % 2 else digits
    return bytes((int(padded[i + 1], 16) << 4) | int(padded[i], 16) for i in range(0, len(padded), 2))


def _decode_bcd_digits(data: bytes, digit_count: int) -> str:
    digits = []
    for byte in data:
        digits.append(format(byte & 0x0F, "X"))
        digits.append(format((byte >> 4) & 0x0F, "X"))
    return "".join(digits[:digit_count])


# --- Addresses (TP-DA / TP-OA / SMSC info) -----------------------------------
#
# Two different length conventions coexist in the same PDU, a well-known
# source of encode/decode bugs: TP-DA/TP-OA's length octet counts *decimal
# digits*, while the SMSC-info length octet counts *octets* (type-of-address
# + BCD bytes). Kept as two separate functions below rather than one
# parametrized one, precisely so this difference can't be papered over.

_TON_INTERNATIONAL_MASK = 0x70
_TON_INTERNATIONAL_VALUE = 0x10  # bits 6-4 = TON=001 (international)


def _address_digits_and_toa(number: str) -> tuple[str, int]:
    international = number.startswith("+")
    digits = number[1:] if international else number
    toa = 0x91 if international else 0x81
    return digits, toa


def _address_prefix(toa: int) -> str:
    return "+" if (toa & _TON_INTERNATIONAL_MASK) == _TON_INTERNATIONAL_VALUE else ""


def encode_address_field(number: str) -> bytes:
    digits, toa = _address_digits_and_toa(number)
    return bytes([len(digits), toa]) + _encode_bcd_digits(digits)


def decode_address_field(data: bytes, offset: int) -> tuple[str, int]:
    digit_count = data[offset]
    toa = data[offset + 1]
    n_bcd_bytes = (digit_count + 1) // 2
    bcd = data[offset + 2 : offset + 2 + n_bcd_bytes]
    return _address_prefix(toa) + _decode_bcd_digits(bcd, digit_count), offset + 2 + n_bcd_bytes


def encode_smsc_field(smsc: str | None) -> bytes:
    if not smsc:
        return bytes([0x00])
    digits, toa = _address_digits_and_toa(smsc)
    bcd = _encode_bcd_digits(digits)
    return bytes([1 + len(bcd), toa]) + bcd


def decode_smsc_field(data: bytes, offset: int) -> tuple[str | None, int]:
    length_octets = data[offset]
    if length_octets == 0:
        return None, offset + 1
    toa = data[offset + 1]
    n_bcd_bytes = length_octets - 1
    bcd = data[offset + 2 : offset + 1 + length_octets]
    digits = _decode_bcd_digits(bcd, n_bcd_bytes * 2).rstrip("F")
    return _address_prefix(toa) + digits, offset + 1 + length_octets


# --- Relative Validity Period (TP-VP, 3GPP TS 23.040 §9.2.3.12.1) -----------


def hours_to_vp(hours: float) -> int:
    """Quantize ``hours`` to the nearest representable relative-VP octet."""
    minutes = hours * 60
    if minutes <= 12 * 60:
        return max(0, min(143, round(minutes / 5) - 1))
    if hours <= 24:
        return max(144, min(167, round((minutes - 12 * 60) / 30) + 143))
    if hours <= 30 * 24:
        return max(168, min(196, round(hours / 24) + 166))
    return max(197, min(255, round(hours / (24 * 7)) + 192))


def vp_to_hours(vp: int) -> float:
    if vp <= 143:
        return (vp + 1) * 5 / 60
    if vp <= 167:
        return 12 + (vp - 143) * 0.5
    if vp <= 196:
        return (vp - 166) * 24
    return (vp - 192) * 24 * 7


# --- SMS-DELIVER timestamp (TP-SCTS, 7 semi-octets) --------------------------


def _encode_bcd_pair(value: int) -> int:
    tens, units = divmod(value, 10)
    return (units << 4) | tens  # semi-octet swap, same convention as _encode_bcd_digits


def _decode_bcd_pair(byte: int) -> int:
    return (byte & 0x0F) * 10 + ((byte >> 4) & 0x0F)


def encode_scts(dt: datetime) -> bytes:
    fields = [dt.year % 100, dt.month, dt.day, dt.hour, dt.minute, dt.second]
    return bytes(_encode_bcd_pair(f) for f in fields) + bytes([0x00])  # timezone: always UTC+0


def decode_scts(data: bytes) -> datetime:
    year, month, day, hour, minute, second = (_decode_bcd_pair(b) for b in data[:6])
    return datetime(2000 + year, month, day, hour, minute, second)


# --- SMS-SUBMIT (mobile-originated) ------------------------------------------

_MTI_SUBMIT = 0b01
_MTI_DELIVER = 0b00


@dataclass
class SubmitPdu:
    destination: str
    text: str
    message_reference: int = 0
    validity_period_hours: float | None = None
    smsc: str | None = None


def _encode_submit_first_octet(vp_present: bool) -> int:
    vpf = 0b10 if vp_present else 0b00
    return _MTI_SUBMIT | (vpf << 3)


def _decode_submit_first_octet(octet: int) -> bool:
    """Returns whether a Validity Period is present; raises if not SMS-SUBMIT."""
    mti = octet & 0b11
    if mti != _MTI_SUBMIT:
        raise ValueError(f"not an SMS-SUBMIT PDU (TP-MTI={mti:#04b})")
    return ((octet >> 3) & 0b11) == 0b10


def encode_submit(pdu: SubmitPdu) -> bytes:
    vp_present = pdu.validity_period_hours is not None
    out = bytearray()
    out += encode_smsc_field(pdu.smsc)
    out.append(_encode_submit_first_octet(vp_present))
    out.append(pdu.message_reference & 0xFF)
    out += encode_address_field(pdu.destination)
    out += bytes([0x00, 0x00])  # TP-PID (normal), TP-DCS (GSM7 default alphabet)
    if vp_present:
        out.append(hours_to_vp(pdu.validity_period_hours))  # type: ignore[arg-type]
    septets = gsm7_encode(pdu.text)
    out.append(len(septets))
    out += pack_septets(septets)
    return bytes(out)


def decode_submit(data: bytes) -> SubmitPdu:
    smsc, offset = decode_smsc_field(data, 0)
    vp_present = _decode_submit_first_octet(data[offset])
    offset += 1
    message_reference = data[offset]
    offset += 1
    destination, offset = decode_address_field(data, offset)
    offset += 2  # skip TP-PID, TP-DCS
    validity_period_hours = None
    if vp_present:
        validity_period_hours = vp_to_hours(data[offset])
        offset += 1
    udl = data[offset]
    offset += 1
    n_octets = (7 * udl + 7) // 8
    text = gsm7_decode(unpack_septets(data[offset : offset + n_octets], udl))
    return SubmitPdu(
        destination=destination,
        text=text,
        message_reference=message_reference,
        validity_period_hours=validity_period_hours,
        smsc=smsc,
    )


def encode_submit_hex(pdu: SubmitPdu) -> str:
    return encode_submit(pdu).hex().upper()


def decode_submit_hex(hex_pdu: str) -> SubmitPdu:
    return decode_submit(bytes.fromhex(hex_pdu))


# --- SMS-DELIVER (network-terminated) ----------------------------------------


@dataclass
class DeliverPdu:
    originator: str
    text: str
    timestamp: datetime
    smsc: str | None = None


def _decode_deliver_first_octet(octet: int) -> None:
    mti = octet & 0b11
    if mti != _MTI_DELIVER:
        raise ValueError(f"not an SMS-DELIVER PDU (TP-MTI={mti:#04b})")


def encode_deliver(pdu: DeliverPdu) -> bytes:
    out = bytearray()
    out += encode_smsc_field(pdu.smsc)
    out.append(_MTI_DELIVER)
    out += encode_address_field(pdu.originator)
    out += bytes([0x00, 0x00])  # TP-PID (normal), TP-DCS (GSM7 default alphabet)
    out += encode_scts(pdu.timestamp)
    septets = gsm7_encode(pdu.text)
    out.append(len(septets))
    out += pack_septets(septets)
    return bytes(out)


def decode_deliver(data: bytes) -> DeliverPdu:
    smsc, offset = decode_smsc_field(data, 0)
    _decode_deliver_first_octet(data[offset])
    offset += 1
    originator, offset = decode_address_field(data, offset)
    offset += 2  # skip TP-PID, TP-DCS
    timestamp = decode_scts(data[offset : offset + 7])
    offset += 7
    udl = data[offset]
    offset += 1
    n_octets = (7 * udl + 7) // 8
    text = gsm7_decode(unpack_septets(data[offset : offset + n_octets], udl))
    return DeliverPdu(originator=originator, text=text, timestamp=timestamp, smsc=smsc)


def encode_deliver_hex(pdu: DeliverPdu) -> str:
    return encode_deliver(pdu).hex().upper()


def decode_deliver_hex(hex_pdu: str) -> DeliverPdu:
    return decode_deliver(bytes.fromhex(hex_pdu))
