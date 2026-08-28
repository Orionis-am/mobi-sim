"""Codec configuration fitness for Module F (`codec_fitness` contract).

Module F's Pb1 (GA via DEAP) evolves a mixed discrete/continuous chromosome
``[bitrate_kbps, frame_size_ms, plc_level, codec_index]`` and needs one
scalar score per individual, per docs/SUJET.md:

    f(x) = w1 * PESQ_sim(x) + w2 * (1 - WER(x)) - w3 * (bitrate / bitrate_max)

``codec_fitness(chromosome)`` decodes that chromosome into an actual
degraded signal (reusing codecs.py + metrics.py) and scores it. Real WER
(whisper_eval.py) is NOT called by default: a GA run can call this function
thousands of times, and each Whisper call costs real API credit and network
latency (see CLAUDE.md's external API table) — completely impractical at
that volume. Instead, ``estimate_wer_proxy`` derives a fast, free,
deterministic WER estimate from the PESQ score. Callers who want the
measured value (e.g. one validation pass over the GA's best individuals,
for the sim-vs-real comparison the project's report is built around) can
inject a real ``wer_fn`` backed by whisper_eval.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from module_a import codecs, metrics, synth_audio

BITRATE_CHOICES_KBPS = (8, 12, 16, 24, 32)
FRAME_SIZE_CHOICES_MS = (10, 20, 30, 40)
CODEC_BY_INDEX = {0: "aac", 1: "gsm", 2: "opus"}

DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)  # (w1: PESQ, w2: 1-WER, w3: bitrate penalty) — needs calibration, see REPORT.md
DEFAULT_BITRATE_MAX_KBPS = float(max(BITRATE_CHOICES_KBPS))
DEFAULT_PACKET_LOSS_RATE = 0.01  # assumed nominal telephony loss rate; not itself a chromosome gene — see REPORT.md

_SNR_AT_MIN_BITRATE_DB = 10.0
_SNR_AT_MAX_BITRATE_DB = 45.0

_reference_cache: dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class CodecConfig:
    bitrate_kbps: float
    frame_size_ms: float
    plc_level: float
    codec: str


def _nearest(value: float, choices: Sequence[float]) -> float:
    return float(min(choices, key=lambda c: abs(c - value)))


def decode_chromosome(chromosome: Sequence[float]) -> CodecConfig:
    """Decode a raw ``[bitrate, frame_size, plc_level, codec]`` chromosome.

    Snaps the two discrete genes (bitrate, frame_size) to their nearest
    valid choice and the codec gene to the nearest of {0, 1, 2} — a GA's
    mutation/crossover operators generally can't be trusted to only ever
    emit exactly-valid discrete values, so decoding tolerates any
    real-valued input near a valid one instead of raising.
    """
    bitrate_raw, frame_size_raw, plc_level_raw, codec_raw = chromosome
    bitrate = _nearest(bitrate_raw, BITRATE_CHOICES_KBPS)
    frame_size = _nearest(frame_size_raw, FRAME_SIZE_CHOICES_MS)
    plc_level = float(np.clip(plc_level_raw, 0.0, 1.0))
    codec_index = int(round(codec_raw)) % len(CODEC_BY_INDEX)
    return CodecConfig(bitrate, frame_size, plc_level, CODEC_BY_INDEX[codec_index])


def _bitrate_to_target_snr_db(bitrate_kbps: float) -> float:
    """Map an encoder bitrate onto a target SNR for codecs.py's noise model.

    Linear interpolation between the discrete bitrate choices' extremes:
    more bits means less quantization noise for any lossy codec, so a
    higher bitrate should relax codecs.py's target_snr_db upward regardless
    of which specific codec is selected.
    """
    lo, hi = BITRATE_CHOICES_KBPS[0], BITRATE_CHOICES_KBPS[-1]
    fraction = float(np.clip((bitrate_kbps - lo) / (hi - lo), 0.0, 1.0))
    return _SNR_AT_MIN_BITRATE_DB + fraction * (_SNR_AT_MAX_BITRATE_DB - _SNR_AT_MIN_BITRATE_DB)


def _apply_packet_loss(
    signal: np.ndarray, sr: int, frame_size_ms: float, plc_level: float, loss_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Drop frames at ``loss_rate`` and conceal them with ``plc_level`` of the last good frame.

    No packet-based coding exists elsewhere in Module A, so this treats
    concealment as faded frame-repetition — a real, simple PLC strategy:
    plc_level=1 repeats the last good frame at full strength (best-case
    concealment), plc_level=0 leaves silence (a raw, unconcealed dropout).
    """
    frame_len = max(1, int(sr * frame_size_ms / 1000))
    out = signal.copy()
    last_good = np.zeros(frame_len, dtype=np.float32)
    for start in range(0, len(signal), frame_len):
        end = min(start + frame_len, len(signal))
        length = end - start
        if rng.random() < loss_rate:
            out[start:end] = last_good[:length] * plc_level
        else:
            last_good = np.zeros(frame_len, dtype=np.float32)
            last_good[:length] = signal[start:end]
    return out.astype(np.float32)


def estimate_wer_proxy(pesq_nb: float) -> float:
    """Cheap, deterministic WER estimate from a PESQ-NB score.

    Used inside codec_fitness by default in place of a real Whisper
    transcription, which would cost API credit and network latency on
    every one of a GA's (potentially thousands of) fitness evaluations.
    Monotonically decreasing and heuristic, not measured: near-perfect
    PESQ maps close to 0 WER, the worst PESQ maps close to 1. Callers
    wanting the measured value should inject a ``wer_fn`` instead.
    """
    normalized_quality = (pesq_nb - metrics.PESQ_MIN) / (metrics.PESQ_MAX - metrics.PESQ_MIN)
    normalized_quality = float(np.clip(normalized_quality, 0.0, 1.0))
    return float((1.0 - normalized_quality) ** 2)


def codec_fitness_components(
    chromosome: Sequence[float],
    reference_signal: np.ndarray,
    sr: int = 8_000,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    bitrate_max_kbps: float = DEFAULT_BITRATE_MAX_KBPS,
    bitrate_cap_kbps: float | None = None,
    constraint_penalty_weight: float = 1.0,
    packet_loss_rate: float = DEFAULT_PACKET_LOSS_RATE,
    wer_fn: Callable[[np.ndarray, int], float] | None = None,
    seed: int | None = None,
) -> dict[str, float | str]:
    """Full breakdown behind ``codec_fitness``'s scalar score, for debugging/plotting."""
    config = decode_chromosome(chromosome)
    rng = np.random.default_rng(seed)

    target_snr_db = _bitrate_to_target_snr_db(config.bitrate_kbps)
    degraded = codecs.simulate_codec(reference_signal, sr, config.codec, target_snr_db=target_snr_db, seed=seed)
    degraded = _apply_packet_loss(degraded, sr, config.frame_size_ms, config.plc_level, packet_loss_rate, rng)

    pesq_nb = metrics.pesq_nb_simplified(reference_signal, degraded, sr)
    wer = wer_fn(degraded, sr) if wer_fn is not None else estimate_wer_proxy(pesq_nb)

    w1, w2, w3 = weights
    fitness = w1 * pesq_nb + w2 * (1.0 - wer) - w3 * (config.bitrate_kbps / bitrate_max_kbps)

    constraint_violation = max(0.0, config.bitrate_kbps - bitrate_cap_kbps) if bitrate_cap_kbps is not None else 0.0
    fitness -= constraint_penalty_weight * constraint_violation

    return {
        "codec": config.codec,
        "bitrate_kbps": config.bitrate_kbps,
        "frame_size_ms": config.frame_size_ms,
        "plc_level": config.plc_level,
        "pesq_nb": pesq_nb,
        "wer": wer,
        "constraint_violation": constraint_violation,
        "fitness": float(fitness),
    }


def _get_reference_signal(sr: int) -> np.ndarray:
    """Lazily generate (once per sample rate) and cache the reference speech signal.

    codec_fitness(chromosome) is called by Module F's GA potentially
    thousands of times per run; regenerating TTS audio (a network call) on
    every evaluation would be wasteful and would make fitness
    non-reproducible run to run.
    """
    if sr not in _reference_cache:
        signal, _ = synth_audio.generate_reference_signal(target_sr=sr)
        _reference_cache[sr] = signal
    return _reference_cache[sr]


def codec_fitness(chromosome: Sequence[float], sr: int = 8_000, **kwargs) -> float:
    """Module F's Pb1 fitness contract: score a ``[bitrate, frame_size, plc_level, codec]`` chromosome.

    Single-argument by design (``codec_fitness(chromosome)``) so a GA's
    evaluate callback can call it directly — the reference signal is
    generated once (see ``_get_reference_signal``) and cached across calls.
    Accepts the same keyword overrides as ``codec_fitness_components``
    (weights, bitrate_cap_kbps, wer_fn, seed, ...) for tuning or testing.
    """
    reference_signal = _get_reference_signal(sr)
    return codec_fitness_components(chromosome, reference_signal, sr=sr, **kwargs)["fitness"]
