"""FastAPI app, middleware, lifespan (`docs/SUJET.md` MOD-E `main.py`) — MobiSim's REST gateway.

Run with: ``uv run uvicorn module_e.main:app --reload`` (loads `.env` first via `python-dotenv`,
same as the `manual_*_check.py` scripts, so `JWT_SECRET_KEY`/`TWILIO_*`/etc. are available to the
routers below without each of them re-loading it). Swagger UI is FastAPI's default `/docs`, built
straight from `models.py`'s Pydantic v2 schemas and each router's `response_model`/docstrings.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from module_e.database import init_db
from module_e.rate_limit import limiter
from module_e.routers import auth as auth_router
from module_e.routers import catalog, codecs, location, optimize, qos, sms


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="MobiSim API",
    description="Hybrid mobile-communications simulation platform — REST gateway over Modules A-D and F.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(auth_router.router)
app.include_router(catalog.router)
app.include_router(codecs.router)
app.include_router(sms.router)
app.include_router(location.router)
app.include_router(qos.router)
app.include_router(optimize.router)


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"name": "MobiSim API", "docs": "/docs"}
