"""Synthetic codec degradation models for Module A.

Each function takes a clean reference signal (as produced by
``synth_audio.generate_reference_signal``, 8 kHz mono float32 in [-1, 1]) and
returns a degraded copy approximating what a real encoder/decoder pass would
do to it, without depending on any proprietary codec library:

- AAC (16 kbps): aggressive low-pass filtering plus pre-echo smearing around
  transients (a well-known low-bitrate AAC artifact).
- GSM Full Rate (13 kbps): telephone-band band-pass filtering plus additive
  noise.
- Opus (24 kbps): near-transparent — no band limiting beyond the reference's
  own Nyquist, only light harmonic distortion and low-level noise.

All three accept a ``target_snr_db`` to control overall degradation severity
and an optional ``seed`` for reproducibility.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

CODEC_BITRATES_KBPS = {"aac": 16, "gsm": 13, "opus": 24}


def _butter_filter(x: np.ndarray, sr: int, low: float | None = None, high: float | None = None, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    if low is not None and high is not None:
        b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    elif high is not None:
        b, a = sps.butter(order, high / nyq, btype="low")
    elif low is not None:
        b, a = sps.butter(order, low / nyq, btype="high")
    else:
        return x
    return sps.filtfilt(b, a, x).astype(np.float32)


def _add_noise_at_snr(x: np.ndarray, target_snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(x**2)) or 1e-12
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32)


def _add_pre_echo(x: np.ndarray, sr: int, rng: np.random.Generator, n_transients: int = 6, spread_ms: float = 8.0, gain: float = 0.35) -> np.ndarray:
    """Smear a noisy, decaying replica of each transient backward in time.

    Approximates AAC's pre-echo artifact: the encoder's block-based
    quantization spreads a transient's coding noise across the whole
    analysis window, audible as noise *before* the transient's true onset.
    """
    frame = max(1, int(sr * 0.02))
    energy = np.convolve(x**2, np.ones(frame) / frame, mode="same")
    d_energy = np.diff(energy, prepend=energy[0])
    onset_idx = np.argsort(d_energy)[-n_transients:]

    spread = max(1, int(sr * spread_ms / 1000))
    ramp = np.linspace(gain, 0.0, spread, dtype=np.float32)
    out = x.copy()
    for idx in onset_idx:
        start = max(0, idx - spread)
        seg_len = idx - start
        if seg_len <= 0:
            continue
        noise = rng.normal(0.0, 1.0, size=seg_len).astype(np.float32)
        out[start:idx] += ramp[-seg_len:] * x[idx] * noise
    return out.astype(np.float32)


def _add_harmonic_distortion(x: np.ndarray, drive: float = 0.03) -> np.ndarray:
    """Light odd-harmonic distortion via a cubic soft nonlinearity."""
    peak_in = np.max(np.abs(x)) or 1.0
    y = x + drive * x**3
    peak_out = np.max(np.abs(y)) or 1.0
    return (y * (peak_in / peak_out)).astype(np.float32)


def _match_peak(degraded: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_peak = np.max(np.abs(reference)) or 1.0
    deg_peak = np.max(np.abs(degraded)) or 1.0
    return (degraded * (ref_peak / deg_peak)).astype(np.float32)


def simulate_aac(signal: np.ndarray, sr: int, target_snr_db: float = 30.0, cutoff_hz: float = 3800.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nyq = sr / 2.0
    degraded = _butter_filter(signal, sr, high=min(cutoff_hz, nyq * 0.98), order=6)
    degraded = _add_pre_echo(degraded, sr, rng)
    degraded = _add_noise_at_snr(degraded, target_snr_db, rng)
    return _match_peak(degraded, signal)


def simulate_gsm(signal: np.ndarray, sr: int, target_snr_db: float = 18.0, low_hz: float = 300.0, high_hz: float = 3400.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nyq = sr / 2.0
    degraded = _butter_filter(signal, sr, low=low_hz, high=min(high_hz, nyq * 0.98), order=4)
    degraded = _add_noise_at_snr(degraded, target_snr_db, rng)
    return _match_peak(degraded, signal)


def simulate_opus(signal: np.ndarray, sr: int, target_snr_db: float = 45.0, drive: float = 0.02, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    degraded = _add_harmonic_distortion(signal, drive=drive)
    degraded = _add_noise_at_snr(degraded, target_snr_db, rng)
    return _match_peak(degraded, signal)


SIMULATORS = {"aac": simulate_aac, "gsm": simulate_gsm, "opus": simulate_opus}


def simulate_codec(signal: np.ndarray, sr: int, codec: str, seed: int | None = None, **kwargs) -> np.ndarray:
    """Dispatch to the simulator for ``codec`` (one of ``"aac"``, ``"gsm"``, ``"opus"``)."""
    try:
        fn = SIMULATORS[codec.lower()]
    except KeyError:
        raise ValueError(f"unknown codec: {codec!r}, expected one of {sorted(SIMULATORS)}") from None
    return fn(signal, sr, seed=seed, **kwargs)
