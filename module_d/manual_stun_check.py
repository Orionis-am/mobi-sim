"""Manual, real-network smoke test against the live STUN server (`docs/SUJET.md` line 234-239).

Not part of the pytest suite (test_module_d.py mocks all STUN sockets via FakeStunSocket — see its
docstring) — running this script makes real UDP round trips to stun.l.google.com:19302. Not run
automatically for that reason; run it yourself when ready:

    uv run python -m module_d.manual_stun_check

Unlike manual_ipinfo_check.py/manual_whisper_check.py/manual_twilio_check.py, this needs no `.env`
token — STUN Binding Requests are public and unauthenticated.
"""

from __future__ import annotations

from module_d import model_e, stun_probe


def main() -> None:
    print(f"Measuring {stun_probe.DEFAULT_N_MEASUREMENTS} real STUN round trips to {stun_probe.DEFAULT_HOST}:{stun_probe.DEFAULT_PORT}...")
    stats = stun_probe.measure_rtt_jitter()

    print(f"rtt_mean_ms={stats.rtt_mean_ms:.2f}")
    print(f"jitter_ms={stats.jitter_ms:.2f}")
    print(f"loss_rate={stats.loss_rate:.2%}")

    delay_ms = stats.rtt_mean_ms / 2.0 + stats.jitter_ms
    for codec in sorted(model_e.IE_TABLE):
        mos = model_e.mos_from_conditions(codec, delay_ms, stats.loss_rate * 100.0)
        print(f"mos[{codec}]={mos:.2f}")


if __name__ == "__main__":
    main()
