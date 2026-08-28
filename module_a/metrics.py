"""Objective audio quality metrics for Module A.

All metrics compare a degraded signal against the clean reference it was
derived from (same length, same sample rate — see ``codecs.py``):

- ``snr_db``: time-domain signal-to-noise ratio.
- ``log_spectral_distortion``: average frame-wise LSD between magnitude
  spectrograms (scipy.signal.stft).
- ``pesq_nb_simplified``: a simplified narrowband PESQ-like score built from
  spectral correlation and temporal-envelope correlation (scipy.signal /
  scipy.stats) — not the ITU-T P.862 reference algorithm, no proprietary PESQ
  library involved.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import stats

PESQ_MIN = 1.0
PESQ_MAX = 4.5


def _align(reference: np.ndarray, degraded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(reference), len(degraded))
    return reference[:n], degraded[:n]


def snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    reference, degraded = _align(reference, degraded)
    noise = degraded - reference
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(noise**2))
    if noise_power <= 1e-20:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def _magnitude_spectrogram(x: np.ndarray, sr: int, frame_ms: float = 32.0, hop_ms: float = 16.0) -> np.ndarray:
    nperseg = max(8, int(sr * frame_ms / 1000))
    noverlap = max(0, nperseg - int(sr * hop_ms / 1000))
    _, _, zxx = sps.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return np.abs(zxx)


def log_spectral_distortion(reference: np.ndarray, degraded: np.ndarray, sr: int, eps: float = 1e-10) -> float:
    reference, degraded = _align(reference, degraded)
    s_ref = _magnitude_spectrogram(reference, sr)
    s_deg = _magnitude_spectrogram(degraded, sr)
    n_frames = min(s_ref.shape[1], s_deg.shape[1])
    s_ref, s_deg = s_ref[:, :n_frames], s_deg[:, :n_frames]

    log_ratio = 10.0 * np.log10((s_ref**2 + eps) / (s_deg**2 + eps))
    per_frame_lsd = np.sqrt(np.mean(log_ratio**2, axis=0))
    return float(np.mean(per_frame_lsd))


def _spectral_correlation(reference: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    s_ref = _magnitude_spectrogram(reference, sr)
    s_deg = _magnitude_spectrogram(degraded, sr)
    n_frames = min(s_ref.shape[1], s_deg.shape[1])
    corr, _ = stats.pearsonr(s_ref[:, :n_frames].ravel(), s_deg[:, :n_frames].ravel())
    return float(corr) if np.isfinite(corr) else 0.0


def _temporal_envelope_correlation(reference: np.ndarray, degraded: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = min(len(reference), len(degraded)) // frame
    if n_frames < 2:
        return 0.0

    ref_env = np.sqrt(np.mean(reference[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    deg_env = np.sqrt(np.mean(degraded[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    if np.std(ref_env) < 1e-12 or np.std(deg_env) < 1e-12:
        return 0.0

    corr, _ = stats.pearsonr(ref_env, deg_env)
    return float(corr) if np.isfinite(corr) else 0.0


def pesq_nb_simplified(reference: np.ndarray, degraded: np.ndarray, sr: int, spectral_weight: float = 0.6) -> float:
    """Simplified narrowband PESQ-like score in the [1.0, 4.5] MOS-like range.

    Blends spectral correlation (how well the frequency content survived)
    with temporal-envelope correlation (how well the amplitude contour
    survived), then rescales the [-1, 1] combined correlation onto the
    conventional PESQ MOS range.
    """
    reference, degraded = _align(reference, degraded)
    spectral_corr = _spectral_correlation(reference, degraded, sr)
    temporal_corr = _temporal_envelope_correlation(reference, degraded, sr)

    combined = spectral_weight * spectral_corr + (1.0 - spectral_weight) * temporal_corr
    combined = float(np.clip(combined, -1.0, 1.0))
    normalized = (combined + 1.0) / 2.0
    return float(np.clip(PESQ_MIN + (PESQ_MAX - PESQ_MIN) * normalized, PESQ_MIN, PESQ_MAX))


def all_metrics(reference: np.ndarray, degraded: np.ndarray, sr: int) -> dict[str, float]:
    """Convenience bundle of every metric in this module for one (reference, degraded) pair."""
    return {
        "snr_db": snr_db(reference, degraded),
        "lsd": log_spectral_distortion(reference, degraded, sr),
        "pesq_nb": pesq_nb_simplified(reference, degraded, sr),
    }
