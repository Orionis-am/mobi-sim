"""Unit tests for Module A (synth_audio, codecs, metrics, whisper_eval).

External I/O (gTTS/pyttsx3 network+engine calls, the Whisper API) is mocked
throughout — these are unit tests for our own logic, not integration tests
for third-party services. No network access and no API credit are required
to run this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from module_a import codecs, metrics, mushra_sim, synth_audio, whisper_eval

SR = 8_000


@pytest.fixture
def reference_signal() -> np.ndarray:
    t = np.linspace(0.0, 1.0, SR, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
    x[::800] += 0.6  # fake transients, to exercise AAC's pre-echo detection
    return x.astype(np.float32)


# --- synth_audio -------------------------------------------------------


class TestResample:
    def test_identity_when_same_rate(self, reference_signal):
        out = synth_audio._resample(reference_signal, SR, SR)
        assert out is reference_signal

    def test_changes_length_for_new_rate(self, reference_signal):
        out = synth_audio._resample(reference_signal, SR, SR // 2)
        assert len(out) == pytest.approx(len(reference_signal) / 2, abs=1)
        assert out.dtype == np.float32


class TestWavRoundtrip:
    def test_signal_survives_16bit_pcm_roundtrip(self, reference_signal):
        wav_bytes = synth_audio.signal_to_wav_bytes(reference_signal, SR)
        decoded, sr = synth_audio._wav_bytes_to_mono_float(wav_bytes)

        assert sr == SR
        assert len(decoded) == len(reference_signal)
        np.testing.assert_allclose(decoded, reference_signal, atol=2e-4)

    def test_wav_bytes_have_riff_header(self, reference_signal):
        wav_bytes = synth_audio.signal_to_wav_bytes(reference_signal, SR)
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_stereo_wav_is_downmixed_to_mono(self):
        import io
        import wave

        left = np.full(100, 0.5, dtype=np.float32)
        right = np.full(100, -0.5, dtype=np.float32)
        stereo_pcm16 = np.empty(200, dtype=np.int16)
        stereo_pcm16[0::2] = (left * 32767).astype(np.int16)
        stereo_pcm16[1::2] = (right * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(stereo_pcm16.tobytes())

        decoded, sr = synth_audio._wav_bytes_to_mono_float(buf.getvalue())
        assert sr == SR
        assert len(decoded) == 100
        np.testing.assert_allclose(decoded, np.zeros(100), atol=1e-3)


class TestGenerateReferenceSignal:
    def test_auto_uses_gtts_when_it_succeeds(self, monkeypatch, reference_signal):
        monkeypatch.setattr(synth_audio, "_synth_via_gtts", lambda text: (reference_signal, SR))

        def _fail_pyttsx3(text):
            raise AssertionError("pyttsx3 should not be called when gTTS succeeds")

        monkeypatch.setattr(synth_audio, "_synth_via_pyttsx3", _fail_pyttsx3)

        signal, sr = synth_audio.generate_reference_signal(backend="auto")
        assert sr == synth_audio.TARGET_SAMPLE_RATE
        assert np.max(np.abs(signal)) == pytest.approx(1.0)

    def test_auto_falls_back_to_pyttsx3_when_gtts_fails(self, monkeypatch, reference_signal):
        def _fail_gtts(text):
            raise RuntimeError("no network")

        monkeypatch.setattr(synth_audio, "_synth_via_gtts", _fail_gtts)
        monkeypatch.setattr(synth_audio, "_synth_via_pyttsx3", lambda text: (reference_signal, SR))

        signal, sr = synth_audio.generate_reference_signal(backend="auto")
        assert sr == synth_audio.TARGET_SAMPLE_RATE
        assert len(signal) == len(reference_signal)

    def test_gtts_backend_does_not_fall_back(self, monkeypatch):
        def _fail_gtts(text):
            raise RuntimeError("no network")

        monkeypatch.setattr(synth_audio, "_synth_via_gtts", _fail_gtts)
        with pytest.raises(RuntimeError):
            synth_audio.generate_reference_signal(backend="gtts")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            synth_audio.generate_reference_signal(backend="carrier-pigeon")

    def test_explicit_pyttsx3_backend_skips_gtts(self, monkeypatch, reference_signal):
        def _fail_gtts(text):
            raise AssertionError("gTTS should not be called for backend='pyttsx3'")

        monkeypatch.setattr(synth_audio, "_synth_via_gtts", _fail_gtts)
        monkeypatch.setattr(synth_audio, "_synth_via_pyttsx3", lambda text: (reference_signal, SR))

        signal, sr = synth_audio.generate_reference_signal(backend="pyttsx3")
        assert sr == synth_audio.TARGET_SAMPLE_RATE
        assert len(signal) == len(reference_signal)


# --- codecs --------------------------------------------------------------


class TestCodecSimulators:
    @pytest.mark.parametrize("codec", ["aac", "gsm", "opus"])
    def test_output_shape_dtype_and_finiteness(self, reference_signal, codec):
        degraded = codecs.simulate_codec(reference_signal, SR, codec, seed=42)
        assert degraded.shape == reference_signal.shape
        assert degraded.dtype == np.float32
        assert np.isfinite(degraded).all()

    @pytest.mark.parametrize("codec", ["aac", "gsm", "opus"])
    def test_peak_matches_reference(self, reference_signal, codec):
        degraded = codecs.simulate_codec(reference_signal, SR, codec, seed=42)
        ref_peak = np.max(np.abs(reference_signal))
        assert np.max(np.abs(degraded)) == pytest.approx(ref_peak, rel=1e-5)

    @pytest.mark.parametrize("codec", ["aac", "gsm", "opus"])
    def test_seed_reproducibility(self, reference_signal, codec):
        a = codecs.simulate_codec(reference_signal, SR, codec, seed=1)
        b = codecs.simulate_codec(reference_signal, SR, codec, seed=1)
        c = codecs.simulate_codec(reference_signal, SR, codec, seed=2)

        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)

    def test_dispatch_is_case_insensitive(self, reference_signal):
        a = codecs.simulate_codec(reference_signal, SR, "GSM", seed=7)
        b = codecs.simulate_codec(reference_signal, SR, "gsm", seed=7)
        np.testing.assert_array_equal(a, b)

    def test_unknown_codec_raises(self, reference_signal):
        with pytest.raises(ValueError):
            codecs.simulate_codec(reference_signal, SR, "mp3")

    def test_aac_survives_signal_shorter_than_pre_echo_frame(self):
        # Regression test: the 20ms energy-smoothing frame in _add_pre_echo
        # (160 samples at 8kHz) used to exceed short signal lengths, making
        # np.convolve(..., mode="same") return a longer array than the
        # signal and producing out-of-bounds transient indices. 100 samples
        # (12.5ms) is short enough to trigger that clamp, while staying
        # above filtfilt's own hard minimum (padlen=21 for a 6th-order
        # filter) so this isolates the pre-echo bug specifically.
        rng = np.random.default_rng(0)
        short_signal = rng.normal(0, 0.3, size=100).astype(np.float32)
        degraded = codecs.simulate_aac(short_signal, SR, seed=0)
        assert degraded.shape == short_signal.shape
        assert np.isfinite(degraded).all()


# --- metrics ---------------------------------------------------------------


class TestMetrics:
    def test_snr_is_infinite_for_identical_signal(self, reference_signal):
        assert metrics.snr_db(reference_signal, reference_signal) == float("inf")

    def test_snr_decreases_as_noise_grows(self, reference_signal):
        rng = np.random.default_rng(0)
        low_noise = reference_signal + rng.normal(0, 0.01, size=reference_signal.shape)
        high_noise = reference_signal + rng.normal(0, 0.2, size=reference_signal.shape)

        assert metrics.snr_db(reference_signal, low_noise) > metrics.snr_db(reference_signal, high_noise)

    def test_lsd_near_zero_for_identical_signal(self, reference_signal):
        assert metrics.log_spectral_distortion(reference_signal, reference_signal, SR) == pytest.approx(0.0, abs=1e-6)

    def test_pesq_nb_at_max_for_identical_signal(self, reference_signal):
        assert metrics.pesq_nb_simplified(reference_signal, reference_signal, SR) == pytest.approx(metrics.PESQ_MAX, abs=1e-6)

    @pytest.mark.parametrize("codec", ["aac", "gsm", "opus"])
    def test_pesq_nb_stays_within_bounds(self, reference_signal, codec):
        degraded = codecs.simulate_codec(reference_signal, SR, codec, seed=3)
        score = metrics.pesq_nb_simplified(reference_signal, degraded, SR)
        assert metrics.PESQ_MIN <= score <= metrics.PESQ_MAX

    def test_all_metrics_has_expected_keys(self, reference_signal):
        degraded = codecs.simulate_gsm(reference_signal, SR, seed=5)
        result = metrics.all_metrics(reference_signal, degraded, SR)
        assert set(result) == {"snr_db", "lsd", "pesq_nb"}

    def test_pesq_nb_handles_silence_without_nan(self):
        # Zero-variance envelope/spectrum would otherwise make pearsonr
        # divide by zero; pesq_nb_simplified must fall back to 0 correlation
        # rather than propagate a NaN.
        silence = np.zeros(SR, dtype=np.float32)
        score = metrics.pesq_nb_simplified(silence, silence, SR)
        assert np.isfinite(score)
        assert metrics.PESQ_MIN <= score <= metrics.PESQ_MAX

    def test_temporal_envelope_correlation_handles_very_short_signal(self):
        # Fewer than 2 frames means correlation is undefined; must return a
        # neutral 0.0 rather than raise or produce a NaN.
        tiny = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        score = metrics.pesq_nb_simplified(tiny, tiny, SR)
        assert np.isfinite(score)


# --- mushra_sim -----------------------------------------------------------


class TestSimulatePanelRatings:
    def test_shape_and_bounds(self):
        ratings = mushra_sim.simulate_panel_ratings(50.0, std=15.0, n_listeners=30, seed=0)
        assert ratings.shape == (30,)
        assert np.all(ratings >= 0.0) and np.all(ratings <= 100.0)

    def test_extreme_mean_is_clipped_to_scale(self):
        ratings = mushra_sim.simulate_panel_ratings(100.0, std=50.0, n_listeners=50, seed=0)
        assert np.all(ratings <= 100.0)
        assert ratings.max() == pytest.approx(100.0)

    def test_seed_reproducibility(self):
        a = mushra_sim.simulate_panel_ratings(50.0, seed=1)
        b = mushra_sim.simulate_panel_ratings(50.0, seed=1)
        np.testing.assert_array_equal(a, b)


class TestSimulateMushraPanel:
    def test_covers_every_codec_and_group(self):
        panel = mushra_sim.simulate_mushra_panel(seed=0)
        assert set(panel) == set(mushra_sim.REPORTED_MEANS)
        for codec, groups in mushra_sim.REPORTED_MEANS.items():
            assert set(panel[codec]) == set(groups)

    def test_ranking_roughly_matches_reported_means(self):
        # With enough listeners, simulated group means should preserve the
        # report's codec ordering (Opus > GSM > AAC) even after clipping.
        panel = mushra_sim.simulate_mushra_panel(n_listeners=200, seed=0)
        opus_mean = np.mean(panel["opus"]["english"])
        gsm_mean = np.mean(panel["gsm"]["english"])
        aac_mean = np.mean(panel["aac"]["english"])
        assert opus_mean > gsm_mean > aac_mean


class TestBootstrapCi:
    def test_constant_ratings_give_a_degenerate_interval(self):
        ratings = np.full(20, 42.0)
        lower, upper = mushra_sim.bootstrap_ci(ratings, n_bootstrap=500, seed=0)
        assert lower == pytest.approx(42.0)
        assert upper == pytest.approx(42.0)

    def test_interval_contains_sample_mean(self):
        ratings = mushra_sim.simulate_panel_ratings(60.0, n_listeners=25, seed=2)
        lower, upper = mushra_sim.bootstrap_ci(ratings, n_bootstrap=1000, seed=3)
        assert lower <= float(np.mean(ratings)) <= upper


class TestSummarizeMushra:
    def test_summary_structure_and_values(self):
        panel = mushra_sim.simulate_mushra_panel(seed=0)
        summary = mushra_sim.summarize_mushra(panel, n_bootstrap=500, seed=0)

        for codec, groups in mushra_sim.REPORTED_MEANS.items():
            for group in groups:
                entry = summary[codec][group]
                assert set(entry) == {"mean", "ci_low", "ci_high"}
                assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]


# --- whisper_eval ------------------------------------------------------


class FakeTranscriptions:
    def __init__(self, text: str):
        self._text = text
        self.last_call: dict | None = None

    def create(self, model, file):
        self.last_call = {"model": model, "file": file}

        class _Response:
            text = self._text

        return _Response()


class FakeClient:
    def __init__(self, text: str = " The quick brown fox. "):
        self.audio = type("_Audio", (), {"transcriptions": FakeTranscriptions(text)})()


class TestWordErrorRate:
    def test_case_and_punctuation_are_ignored(self):
        assert whisper_eval.word_error_rate("the quick brown fox", "The quick brown Fox.") == 0.0

    def test_real_substitution_is_counted(self):
        assert whisper_eval.word_error_rate("the quick brown fox", "the quick brown fax") == pytest.approx(0.25)


class TestTranscribe:
    def test_sends_wav_bytes_and_strips_result(self, reference_signal):
        client = FakeClient(" hello world ")
        text = whisper_eval.transcribe(reference_signal, SR, client=client)

        assert text == "hello world"
        filename, data, content_type = client.audio.transcriptions.last_call["file"]
        assert filename == "audio.wav"
        assert content_type == "audio/wav"
        assert data[:4] == b"RIFF"

    def test_missing_api_key_raises_when_no_client_given(self, monkeypatch, reference_signal):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            whisper_eval.transcribe(reference_signal, SR)

    def test_client_is_built_from_env_var_when_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
        client = whisper_eval._client()
        assert client.api_key == "sk-test-dummy-key"


class TestEvaluateIntelligibility:
    def test_combines_transcription_and_wer(self, reference_signal):
        client = FakeClient("the quick brown fox")
        result = whisper_eval.evaluate_intelligibility(reference_signal, SR, "the quick brown fox", client=client)
        assert result == {"transcription": "the quick brown fox", "wer": 0.0}

    def test_evaluate_codecs_intelligibility_covers_every_codec(self, reference_signal):
        client = FakeClient("the quick brown fox")
        degraded_by_codec = {
            codec: codecs.simulate_codec(reference_signal, SR, codec, seed=9) for codec in ["aac", "gsm", "opus"]
        }
        results = whisper_eval.evaluate_codecs_intelligibility(degraded_by_codec, SR, "the quick brown fox", client=client)

        assert set(results) == {"aac", "gsm", "opus"}
        assert all(r["wer"] == 0.0 for r in results.values())
