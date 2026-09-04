"""`POST /optimize/run` + `GET /optimize/{job_id}` — orchestrates Module F (`docs/SUJET.md` MOD-E).

module_f's `run_*`/`deap_ga_*`/`random_search_*`/`grid_search_*` functions are synchronous and
CPU-bound (no async I/O inside), so a run is dispatched via `BackgroundTasks` rather than awaited
in the request: `POST /optimize/run` inserts a `queued` `Job` row and returns its id immediately;
the background task flips it to `running`, executes the algorithm, and stores the JSON-serialized
result (or the error) before marking it `done`/`error`. `GET /optimize/{job_id}` just reads the
row — no separate task queue needed, matching the project's single-machine constraint.

Exposes all four algorithm families module_f implements (GA/random/grid for Pb1, NSGA-II/MOEA-D
for Pb2, DE/PSO for Pb3) rather than just the spec prose's narrower "AG / NSGA-II" phrase — see
REPORT.md.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from module_e import database
from module_e.auth import get_current_user
from module_e.database import Job, get_db
from module_e.models import OptimizeJobResponse, OptimizeRunRequest, OptimizeRunResponse
from module_e.rate_limit import RATE_LIMIT, limiter
from module_f import pb1_codec, pb2_bts, pb3_qos

router = APIRouter(prefix="/optimize", tags=["optimize"], dependencies=[Depends(get_current_user)])

ALGORITHM_MAP: dict[str, dict[str, Callable[..., Any]]] = {
    "pb1_codec": {"ga": pb1_codec.deap_ga_codec, "random": pb1_codec.random_search_codec, "grid": pb1_codec.grid_search_codec},
    "pb2_bts": {"nsga2": pb2_bts.run_nsga2, "moead": pb2_bts.run_moead},
    "pb3_qos": {"de": pb3_qos.run_de, "pso": pb3_qos.run_pso},
}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def run_job(job_id: str) -> None:
    """Background task body — owns its own DB session (the request's is long gone by the time this runs).

    Looks up ``database.SessionLocal`` dynamically (module-attribute access, not an imported name)
    so tests that monkeypatch it to an isolated in-memory engine are respected here too — a plain
    ``from module_e.database import SessionLocal`` would freeze the reference at import time.
    """
    with database.SessionLocal() as db:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        db.commit()

        try:
            func = ALGORITHM_MAP[job.problem][job.algorithm]
            params = json.loads(job.params_json)
            result = func(**params)
            job.result_json = json.dumps(_to_jsonable(result))
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 — surfaced via the job row, not raised out of the background task
            job.status = "error"
            job.error = str(exc)
        finally:
            job.finished_at = datetime.now(timezone.utc)
            db.commit()


@router.post("/run", response_model=OptimizeRunResponse, status_code=202)
@limiter.limit(RATE_LIMIT)
def run(request: Request, payload: OptimizeRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> OptimizeRunResponse:
    algorithms = ALGORITHM_MAP.get(payload.problem, {})
    if payload.algorithm not in algorithms:
        raise HTTPException(422, detail=f"algorithm {payload.algorithm!r} not available for problem {payload.problem!r}; choose from {list(algorithms)}")

    job_id = uuid.uuid4().hex
    job = Job(id=job_id, problem=payload.problem, algorithm=payload.algorithm, params_json=json.dumps(payload.params), status="queued")
    db.add(job)
    db.commit()

    background_tasks.add_task(run_job, job_id)
    return OptimizeRunResponse(job_id=job_id, status="queued")


@router.get("/{job_id}", response_model=OptimizeJobResponse)
@limiter.limit(RATE_LIMIT)
def get_job(request: Request, job_id: str, db: Session = Depends(get_db)) -> OptimizeJobResponse:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, detail=f"job {job_id} not found")

    return OptimizeJobResponse(
        job_id=job.id,
        problem=job.problem,
        algorithm=job.algorithm,
        status=job.status,
        created_at=job.created_at,
        finished_at=job.finished_at,
        result=json.loads(job.result_json) if job.result_json else None,
        error=job.error,
    )
