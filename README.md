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
