"""`POST /qos/predict` + `GET /qos/session/live` (SSE) (`docs/SUJET.md` MOD-E).

Both wrap `module_d.session_sim.simulate_session`, itself built on the ITU-T G.107 E-model
(`module_d.model_e`). The live stream isn't driven by a real per-tick STUN probe — a real
`stun.l.google.com:19302` round trip is too slow for a ~1s SSE cadence — instead
`module_d.stun_probe.measure_rtt_jitter` seeds the *initial* rtt/jitter once, which then drifts via
a small seeded random walk each tick.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from module_d.session_sim import simulate_session
from module_d.stun_probe import measure_rtt_jitter
from module_e.auth import get_current_user
from module_e.models import QosPredictRequest, QosPredictResponse
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(prefix="/qos", tags=["qos"], dependencies=[Depends(get_current_user)])

LIVE_TICK_SECONDS = 1.0
LIVE_DRIFT_STD_MS = 2.0


@router.post("/predict", response_model=QosPredictResponse)
@limiter.limit(RATE_LIMIT)
def predict(request: Request, payload: QosPredictRequest) -> QosPredictResponse:
    result = simulate_session(payload.codec, payload.bandwidth_kbps, payload.required_bandwidth_kbps, payload.rtt_ms, payload.jitter_ms, seed=payload.seed)
    return QosPredictResponse(r_factor=result.r_factor, mos=result.mos, delay_ms=result.delay_ms, loss_pct=result.loss_pct, codec=result.codec)


async def _live_qos_events(
    request: Request, codec: str, bandwidth_kbps: float, required_bandwidth_kbps: float, rtt_ms: float, jitter_ms: float, seed: int | None
):
    rng = np.random.default_rng(seed)
    tick = 0
    while not await request.is_disconnected():
        rtt_ms = max(1.0, rtt_ms + float(rng.normal(0.0, LIVE_DRIFT_STD_MS)))
        jitter_ms = max(0.0, jitter_ms + float(rng.normal(0.0, LIVE_DRIFT_STD_MS / 4)))
        result = simulate_session(codec, bandwidth_kbps, required_bandwidth_kbps, rtt_ms, jitter_ms, seed=int(rng.integers(0, 2**31)))
        payload = {"tick": tick, "r_factor": result.r_factor, "mos": result.mos, "delay_ms": result.delay_ms, "loss_pct": result.loss_pct, "codec": result.codec}
        yield f"data: {json.dumps(payload)}\n\n"
        tick += 1
        await asyncio.sleep(LIVE_TICK_SECONDS)


@router.get("/session/live")
async def session_live(
    request: Request,
    codec: str = "opus",
    bandwidth_kbps: float = 64.0,
    required_bandwidth_kbps: float = 64.0,
    rtt_ms: float = 40.0,
    jitter_ms: float = 5.0,
    seed: int | None = None,
    use_real_stun: bool = False,
) -> StreamingResponse:
    """`use_real_stun=true` seeds the initial rtt/jitter from a real `stun.l.google.com:19302`
    round trip (blocking network I/O, run off the event loop) instead of the query-param defaults;
    every subsequent tick still drifts synthetically — a real probe per SSE tick would be far too
    slow for a ~1s cadence."""
    if use_real_stun:
        stats = await run_in_threadpool(measure_rtt_jitter)
        rtt_ms, jitter_ms = stats.rtt_mean_ms, stats.jitter_ms

    return StreamingResponse(
        _live_qos_events(request, codec, bandwidth_kbps, required_bandwidth_kbps, rtt_ms, jitter_ms, seed), media_type="text/event-stream"
    )
