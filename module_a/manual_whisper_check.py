"""Manual, cost-incurring smoke test against the real Whisper API.

Not part of the pytest suite (test_module_a.py mocks the OpenAI client on
purpose, per its own docstring, so `pytest` never spends API credit). Run
this script directly, once in a while, to sanity-check the whole Module A
pipeline against the real API rather than a fake client:

    uv run python -m module_a.manual_whisper_check

Requires OPENAI_API_KEY in the environment or in a .env file at the repo
root. Uses the short default reference sentence (a few seconds of audio) to
keep the Whisper cost negligible (~$0.006/min per CLAUDE.md).
"""

from __future__ import annotations

from dotenv import load_dotenv

from module_a import codecs, metrics, synth_audio, whisper_eval

CODECS = ["aac", "gsm", "opus"]


def main() -> None:
    load_dotenv()

    print("Generating reference signal...")
    signal, sr = synth_audio.generate_reference_signal()
    reference_text = synth_audio.DEFAULT_TEXT

    print("Degrading through each codec simulator...")
    degraded_by_codec = {codec: codecs.simulate_codec(signal, sr, codec, seed=42) for codec in CODECS}

    print("Scoring local objective metrics (SNR/LSD/PESQ-NB simplified)...")
    local_metrics = {codec: metrics.all_metrics(signal, degraded, sr) for codec, degraded in degraded_by_codec.items()}

    print("Calling the real Whisper API (clean reference + each degraded codec)...")
    client = whisper_eval._client()
    reference_result = whisper_eval.evaluate_intelligibility(signal, sr, reference_text, client=client)
    degraded_results = whisper_eval.evaluate_codecs_intelligibility(degraded_by_codec, sr, reference_text, client=client)

    print(f"\nReference text: {reference_text!r}")
    print(f"Clean transcription: {reference_result['transcription']!r} (WER={reference_result['wer']:.3f})\n")

    header = f"{'codec':6} {'snr_db':>8} {'lsd':>8} {'pesq_nb':>8} {'wer':>6}  transcription"
    print(header)
    print("-" * len(header))
    for codec in CODECS:
        m = local_metrics[codec]
        r = degraded_results[codec]
        print(f"{codec:6} {m['snr_db']:8.2f} {m['lsd']:8.2f} {m['pesq_nb']:8.2f} {r['wer']:6.3f}  {r['transcription']!r}")


if __name__ == "__main__":
    main()
