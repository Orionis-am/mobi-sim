"""Speech intelligibility via OpenAI Whisper for Module A.

Transcribes a (possibly codec-degraded) signal through the Whisper API and
scores intelligibility against the known reference text with jiwer's Word
Error Rate. Audio is encoded to WAV entirely in memory (``BytesIO``) — no
temp files, matching ``synth_audio.py``'s in-memory pipeline.

Costs real API credit (~$0.006/min per CLAUDE.md) — callers should keep
clips short and pass a shared ``client`` when transcribing many signals in
one run rather than instantiating a new one per call.
"""

from __future__ import annotations

import os

import jiwer
import numpy as np
from openai import OpenAI

from module_a.synth_audio import signal_to_wav_bytes

WHISPER_MODEL = "whisper-1"

# Whisper's output is cased and punctuated regardless of audio quality (it's a
# transcription-style artifact, not an intelligibility failure), so normalize
# both sides before diffing words — otherwise WER would count "Fox" vs "fox."
# as an error even on a perfect transcription.
_WER_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — required to call the Whisper API")
    return OpenAI(api_key=api_key)


def transcribe(signal: np.ndarray, sample_rate: int, client: OpenAI | None = None) -> str:
    """Transcribe a mono float32 signal via Whisper, encoding it as WAV in memory."""
    wav_bytes = signal_to_wav_bytes(signal, sample_rate)
    client = client or _client()
    response = client.audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=("audio.wav", wav_bytes, "audio/wav"),
    )
    return response.text.strip()


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Word Error Rate between the known reference text and a transcription.

    Both texts are lower-cased and stripped of punctuation before comparison
    (see ``_WER_NORMALIZE``).
    """
    return float(
        jiwer.wer(
            reference_text,
            hypothesis_text,
            reference_transform=_WER_NORMALIZE,
            hypothesis_transform=_WER_NORMALIZE,
        )
    )


def evaluate_intelligibility(signal: np.ndarray, sample_rate: int, reference_text: str, client: OpenAI | None = None) -> dict[str, float | str]:
    """Transcribe ``signal`` and score it against ``reference_text``."""
    hypothesis = transcribe(signal, sample_rate, client=client)
    return {"transcription": hypothesis, "wer": word_error_rate(reference_text, hypothesis)}


def evaluate_codecs_intelligibility(
    degraded_by_codec: dict[str, np.ndarray],
    sample_rate: int,
    reference_text: str,
    client: OpenAI | None = None,
) -> dict[str, dict[str, float | str]]:
    """Run ``evaluate_intelligibility`` for each codec's degraded signal, reusing one client."""
    client = client or _client()
    return {
        codec: evaluate_intelligibility(signal, sample_rate, reference_text, client=client)
        for codec, signal in degraded_by_codec.items()
    }
