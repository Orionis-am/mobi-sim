"""Synthetic reference speech generation for Module A.

Produces a clean 8 kHz mono reference speech signal entirely in memory (no
microphone, no files on disk) via gTTS (online, natural voice) with a
pyttsx3 fallback (offline) when gTTS is unavailable (no network, quota, etc.).
"""

from __future__ import annotations

import io
import wave

import numpy as np

TARGET_SAMPLE_RATE = 8_000
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog near the mobile network base station."


def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    duration = len(signal) / orig_sr
    n_target = int(round(duration * target_sr))
    orig_t = np.linspace(0.0, duration, num=len(signal), endpoint=False)
    target_t = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(target_t, orig_t, signal).astype(np.float32)


def _wav_bytes_to_mono_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    max_val = float(np.iinfo(dtype).max)
    data /= max_val

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    return data, sr


def _synth_via_gtts(text: str) -> tuple[np.ndarray, int]:
    from gtts import gTTS
    from pydub import AudioSegment

    mp3_buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(mp3_buf)
    mp3_buf.seek(0)

    segment = AudioSegment.from_file(mp3_buf, format="mp3").set_channels(1)
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    samples /= float(1 << (8 * segment.sample_width - 1))
    return samples, segment.frame_rate


def _synth_via_pyttsx3(text: str) -> tuple[np.ndarray, int]:
    import os
    import tempfile

    import pyttsx3

    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return _wav_bytes_to_mono_float(wav_bytes)


def generate_reference_signal(
    text: str = DEFAULT_TEXT,
    target_sr: int = TARGET_SAMPLE_RATE,
    backend: str = "auto",
) -> tuple[np.ndarray, int]:
    """Generate a clean reference speech signal, normalized to ``target_sr`` Hz mono.

    Returns ``(signal, sample_rate)`` where ``signal`` is float32 in [-1, 1].
    ``backend`` is one of ``"auto"`` (try gTTS then fall back to pyttsx3),
    ``"gtts"``, or ``"pyttsx3"``.
    """
    if backend not in {"auto", "gtts", "pyttsx3"}:
        raise ValueError(f"unknown backend: {backend!r}")

    signal: np.ndarray
    sr: int

    if backend in {"auto", "gtts"}:
        try:
            signal, sr = _synth_via_gtts(text)
        except Exception:
            if backend == "gtts":
                raise
            signal, sr = _synth_via_pyttsx3(text)
    else:
        signal, sr = _synth_via_pyttsx3(text)

    signal = _resample(signal, sr, target_sr)

    peak = np.max(np.abs(signal)) or 1.0
    signal = (signal / peak).astype(np.float32)

    return signal, target_sr


def signal_to_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 [-1, 1] mono signal as 16-bit PCM WAV bytes (in memory)."""
    pcm = np.clip(signal, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
