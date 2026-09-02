"""Real-time Rich console dashboard for Module D (`docs/SUJET.md` line 238-239).

Displays RTT / Gigue / Perte / MOS / Codec, refreshed every 500 ms by default, one real STUN
measurement per row, with automatic CSV export as it runs (not buffered until the end, so an
interrupted run still leaves partial data on disk).
"""

from __future__ import annotations

import csv
import math
import socket
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from module_d import model_e, stun_probe

DEFAULT_CSV_PATH = "results/module_d_dashboard.csv"
DEFAULT_N_ITERATIONS = 20
DEFAULT_REFRESH_S = 0.5
CSV_FIELDS = ["iteration", "rtt_ms", "jitter_ms", "loss_pct", "mos", "codec"]


def _build_table(rtt_ms: float, jitter_ms: float, loss_pct: float, mos: float, codec: str) -> Table:
    table = Table(title="Module D — QoS temps reel")
    for column in ("RTT (ms)", "Gigue (ms)", "Perte (%)", "MOS", "Codec"):
        table.add_column(column)
    rtt_display = "-" if math.isnan(rtt_ms) else f"{rtt_ms:.1f}"
    table.add_row(rtt_display, f"{jitter_ms:.1f}", f"{loss_pct:.1f}", f"{mos:.2f}", codec)
    return table


def run_dashboard(
    n_iterations: int = DEFAULT_N_ITERATIONS,
    refresh_s: float = DEFAULT_REFRESH_S,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    host: str = stun_probe.DEFAULT_HOST,
    port: int = stun_probe.DEFAULT_PORT,
    sock=None,
    codec: str = "opus",
    sleep=time.sleep,
    console: Console | None = None,
) -> Path:
    """Run the live dashboard for `n_iterations` real STUN measurements, exporting each row to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    owns_socket = sock is None
    sock = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rtt_samples: list[float] = []
    n_lost = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        try:
            with Live(_build_table(float("nan"), 0.0, 0.0, 1.0, codec), console=console, refresh_per_second=max(1, int(1 / refresh_s) if refresh_s > 0 else 4)) as live:
                for i in range(n_iterations):
                    rtt = stun_probe.measure_one_rtt(sock, host, port)
                    if rtt is None:
                        n_lost += 1
                    else:
                        rtt_samples.append(rtt)

                    jitter_ms = float(np.std(np.diff(rtt_samples))) if len(rtt_samples) >= 2 else 0.0
                    loss_pct = (n_lost / (i + 1)) * 100.0
                    rtt_latest = rtt_samples[-1] if rtt_samples else float("nan")
                    delay_ms = 0.0 if math.isnan(rtt_latest) else rtt_latest / 2.0 + jitter_ms
                    mos = model_e.mos_from_conditions(codec, delay_ms, loss_pct)

                    live.update(_build_table(rtt_latest, jitter_ms, loss_pct, mos, codec))
                    writer.writerow(
                        {
                            "iteration": i,
                            "rtt_ms": rtt_latest,
                            "jitter_ms": jitter_ms,
                            "loss_pct": loss_pct,
                            "mos": mos,
                            "codec": codec,
                        }
                    )
                    csv_file.flush()

                    if i < n_iterations - 1:
                        sleep(refresh_s)
        finally:
            if owns_socket:
                sock.close()

    return csv_path
