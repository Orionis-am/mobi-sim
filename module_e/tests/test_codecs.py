"""`POST /codecs/evaluate` — mocks `synth_audio.generate_reference_signal` (gTTS/pyttsx3, real
network/OS calls) and, for the real-Whisper path, `evaluate_codecs_intelligibility` (real API,
real cost) — same mocking boundary `module_a/test_module_a.py` itself uses. `simulate_codec`,
`all_metrics`, and `simulate_mushra_panel` are pure numpy and run for real.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from module_a import synth_audio
from module_e.routers import codecs as codecs_router

SR = 8_000


def _fake_reference_signal(text: str, target_sr: int = SR, backend: str = "auto"):
    rng = np.random.default_rng(0)
    duration_s = 1.0
    return rng.uniform(-0.5, 0.5, size=int(target_sr * duration_s)).astype(np.float32), target_sr


@pytest.fixture(autouse=True)
def _mock_reference_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synth_audio, "generate_reference_signal", _fake_reference_signal)


class TestEvaluate:
    def test_returns_metrics_per_requested_codec(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post("/codecs/evaluate", headers=auth_headers, json={"text": "bonjour", "codecs": ["gsm", "opus"], "seed": 1})
        assert response.status_code == 200
        body = response.json()
        assert set(body["results"]) == {"gsm", "opus"}
        for metrics in body["results"].values():
            assert {"snr_db", "pesq_nb", "log_spectral_distortion", "mushra_mean", "wer"} <= metrics.keys()
            assert 0.0 <= metrics["wer"] <= 1.0

    def test_defaults_to_all_three_codecs(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post("/codecs/evaluate", headers=auth_headers, json={"text": "bonjour"})
        assert response.status_code == 200
        assert set(response.json()["results"]) == {"aac", "gsm", "opus"}

    def test_reproducible_with_same_seed(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        payload = {"text": "bonjour", "codecs": ["gsm"], "seed": 42}
        first = client.post("/codecs/evaluate", headers=auth_headers, json=payload).json()
        second = client.post("/codecs/evaluate", headers=auth_headers, json=payload).json()
        assert first == second

    def test_real_whisper_opt_in_uses_measured_wer(self, client: TestClient, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_evaluate_codecs_intelligibility(degraded_by_codec, sample_rate, reference_text, client=None):
            return {codec: {"transcription": reference_text, "wer": 0.5} for codec in degraded_by_codec}

        monkeypatch.setattr(codecs_router, "evaluate_codecs_intelligibility", fake_evaluate_codecs_intelligibility)

        response = client.post("/codecs/evaluate", headers=auth_headers, json={"text": "bonjour", "codecs": ["gsm"], "use_real_whisper": True})
        assert response.status_code == 200
        assert response.json()["results"]["gsm"]["wer"] == 0.5

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/codecs/evaluate", json={"text": "bonjour"})
        assert response.status_code == 401
