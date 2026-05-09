# Contributing to dual-bench

Thank you for considering a contribution. This document describes how to
set up a development environment, run the test suite, and add new problems.

## Development setup

```bash
git clone https://github.com/jtorreci/dual-bench.git
cd dual-bench
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,all,analysis]"
```

## Running tests

```bash
# Fast unit tests (skips optimisation runs)
pytest -m "not slow"

# Full suite including slow integration tests
pytest

# With coverage
pytest --cov=dual_bench --cov-report=term-missing
```

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/
```

## Adding a new problem

1. Create a class that inherits from `DualProblem` (single-objective) or
   `CMOProblem` (constrained multi-objective) in `src/dual_bench/problems/`.
2. Implement `evaluate(x)` following the existing examples.
3. Register the problem in the appropriate suite inside
   `src/dual_bench/problems/__init__.py`.
4. Add tests in `tests/` covering evaluation, feasibility, and bounds.
5. If a known optimum exists, add it to
   `src/dual_bench/references/optima.py`.

## Pull requests

- Keep changes focused and well-tested.
- Follow the existing code style (Ruff-clean, type hints, docstrings).
- Update `CHANGELOG.md` with a brief description of your change.
