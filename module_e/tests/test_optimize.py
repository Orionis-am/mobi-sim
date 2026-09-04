"""`/optimize/*` — runs module_f's real (tiny-budget) algorithms; `TestClient` executes
`BackgroundTasks` synchronously within the request, so a job is already `done` by the time
`POST /optimize/run` returns, letting these tests poll once instead of needing to wait/retry.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _run_and_fetch(client: TestClient, headers: dict[str, str], problem: str, algorithm: str, params: dict) -> dict:
    run_response = client.post("/optimize/run", headers=headers, json={"problem": problem, "algorithm": algorithm, "params": params})
    assert run_response.status_code == 202
    job_id = run_response.json()["job_id"]
    return client.get(f"/optimize/{job_id}", headers=headers).json()


class TestPb1Codec:
    def test_ga(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb1_codec", "ga", {"pop_size": 4, "n_generations": 2, "seed": 1})
        assert body["status"] == "done"
        assert body["result"]["best_chromosome"] is not None
        assert len(body["result"]["history_best_fitness"]) == 2  # one entry per generation

    def test_random(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb1_codec", "random", {"n_evaluations": 3, "seed": 1})
        assert body["status"] == "done"
        assert body["result"]["n_evaluations"] == 3

    def test_grid(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb1_codec", "grid", {"plc_steps": 2})
        assert body["status"] == "done"


class TestPb2Bts:
    def test_nsga2(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb2_bts", "nsga2", {"n_new_bts": 2, "pop_size": 4, "n_generations": 2, "seed": 1})
        assert body["status"] == "done"
        assert body["result"]["final_X"] is not None


class TestPb3Qos:
    def test_de(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb3_qos", "de", {"maxiter": 2, "popsize": 2, "seed": 1})
        assert body["status"] == "done"

    def test_pso(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb3_qos", "pso", {"swarmsize": 3, "maxiter": 2, "seed": 1})
        assert body["status"] == "done"


class TestErrors:
    def test_unknown_algorithm_for_problem_is_422(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.post("/optimize/run", headers=auth_headers, json={"problem": "pb1_codec", "algorithm": "nsga2", "params": {}})
        assert response.status_code == 422

    def test_unknown_job_id_is_404(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        response = client.get("/optimize/does-not-exist", headers=auth_headers)
        assert response.status_code == 404

    def test_bad_params_recorded_as_job_error(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        body = _run_and_fetch(client, auth_headers, "pb2_bts", "nsga2", {})  # missing required n_new_bts
        assert body["status"] == "error"
        assert body["error"]

    def test_requires_auth(self, client: TestClient) -> None:
        response = client.post("/optimize/run", json={"problem": "pb3_qos", "algorithm": "de", "params": {}})
        assert response.status_code == 401
