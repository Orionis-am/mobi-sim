# mobi-sim

Student project revolving around mobile communications protocol, GSM architecture simulation and Evolutionnary Algortihm

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # fill in your own free-tier API keys
```

## Running a module

Each module exposes its own scripts and a `test_module_*.py` test file. For example, once
Module A exists:

```bash
uv run pytest module_a -v
uv run python -m module_a.visualize   # generates plots into results/module_a/
```

## Running the API (Module E)

Module E is a FastAPI gateway exposing Modules A-D directly and orchestrating Module F's
optimization algorithms asynchronously.

```bash
uv run pytest module_e -v
uv run uvicorn module_e.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the Swagger UI. `POST /auth/register` then
`POST /auth/token` (OAuth2 password flow) to get a bearer token for every other route.
