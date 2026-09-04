"""`POST /codecs/evaluate` — Module A's full pipeline (SNR, PESQ, WER, MUSHRA), per `docs/SUJET.md`.

WER defaults to `module_a.fitness.estimate_wer_proxy` (free, deterministic) rather than a real
Whisper call — same reasoning as `codec_fitness`'s own default: an API-gateway route can be hit far
more often than a single manual check, and Whisper costs real credit per CLAUDE.md. Set
`use_real_whisper: true` to opt into the measured value instead.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from module_a import codecs as codecs_module
from module_a import metrics as metrics_module
from module_a import synth_audio
from module_a.fitness import estimate_wer_proxy
from module_a.mushra_sim import simulate_mushra_panel
from module_a.whisper_eval import evaluate_codecs_intelligibility
from module_e.auth import get_current_user
from module_e.models import CodecEvaluateRequest, CodecEvaluateResponse, CodecMetrics
from module_e.rate_limit import RATE_LIMIT, limiter

router = APIRouter(prefix="/codecs", tags=["codecs"], dependencies=[Depends(get_current_user)])


@router.post("/evaluate", response_model=CodecEvaluateResponse)
@limiter.limit(RATE_LIMIT)
def evaluate(request: Request, payload: CodecEvaluateRequest) -> CodecEvaluateResponse:
    reference_signal, sr = synth_audio.generate_reference_signal(payload.text, target_sr=payload.sample_rate)
    mushra_panel = simulate_mushra_panel(seed=payload.seed)

    degraded_by_codec = {codec: codecs_module.simulate_codec(reference_signal, sr, codec, seed=payload.seed) for codec in payload.codecs}

    real_wer_by_codec: dict[str, dict[str, float | str]] = {}
    if payload.use_real_whisper:
        real_wer_by_codec = evaluate_codecs_intelligibility(degraded_by_codec, sr, payload.text)

    results: dict[str, CodecMetrics] = {}
    for codec, degraded in degraded_by_codec.items():
        base = metrics_module.all_metrics(reference_signal, degraded, sr)
        group_means = mushra_panel.get(codec, {})
        mushra_mean = float(sum(ratings.mean() for ratings in group_means.values()) / len(group_means)) if group_means else 0.0
        wer = float(real_wer_by_codec[codec]["wer"]) if codec in real_wer_by_codec else estimate_wer_proxy(base["pesq_nb"])

        results[codec] = CodecMetrics(
            snr_db=base["snr_db"], pesq_nb=base["pesq_nb"], log_spectral_distortion=base["lsd"], mushra_mean=mushra_mean, wer=wer
        )

    return CodecEvaluateResponse(reference_text=payload.text, sample_rate=sr, results=results)
