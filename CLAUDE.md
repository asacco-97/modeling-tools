# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/unit/test_core.py

# Run a single test by name
pytest tests/unit/test_core.py::test_residualize_removes_linear_control

# Run only unit tests (exclude integration)
pytest tests/unit/

# Lint
ruff check src tests

# Type-check
mypy src tests
```

Integration tests in `tests/integration/` are not marked separately but are heavier and slower than unit tests.

## Architecture

This is a single-package library (`pe_tools`) with two distinct tools:

### 1. Residual Signal Finder (`src/pe_tools/signal_finder/`)

The main purpose of the repo. There are two versions with different philosophies:

**V1 (`core.py`) — `ResidualSignalFinder`**
- Takes `(X, y_true, y_pred)` and fits a shallow tree model to the residuals `y_true - y_pred` using cross-validation, holdout, or bootstrap splits.
- Returns `ResidualSignalFinderResult`: feature importance, stability across folds, binned diagnostics, OOF predictions, and optional SHAP interaction importance (only when `max_depth=2`).
- For binary targets (0/1), it automatically switches from modeling residuals to modeling `y_true` directly with `y_pred` injected as a control feature, and emits a `UserWarning`.
- `offbalance=True` shifts per-split predictions to match the target mean before scoring — useful for calibrated binary models.
- Report outputs: `.to_excel()`, `.to_html()`, `.plot_top_residual_signals()`, `.plot_top_interactions()`.
- Also exposes two lower-level functions: `residualize()` (OLS-based, no tree model) and `find_residual_signal()` (residualize both target and signal, return correlation).

**V2 (`v2.py`) — `ResidualSignalFinderV2`**
- Univariate approach: evaluates each feature independently by fitting a single-feature model to residuals across many bootstrap or k-fold splits.
- Adds a null/shadow-feature baseline (`null_strategy`: permuted features, random noise, or shuffled residuals) so each feature's lift can be compared against chance.
- Optional multivariate screening pass (`screening_enabled=True`) reduces the candidate set to `screening_top_k` features before the univariate loop.
- Results stored as attributes after `.fit()`: `summary_`, `bootstrap_results_`, `effect_curves_`, `null_results_`, `screening_results_`.
- Handles categorical features natively; V1 requires all-numeric input.
- Supports custom splits via `splits=` argument (list of dicts with `train`/`validation`/`holdout` keys, or `(train_idx, validation_idx)` tuples).

### 2. Model Evaluator (`src/pe_tools/evaluation/`)

A separate, model-agnostic evaluator (`ModelEvaluator`) for tabular prediction outputs. Supports regression, binary classification, count, and rate tasks. Returns an `EvaluationResult` with calibration, lift, segment breakdowns, and feature-split diagnostics.

### Public imports

Everything intended for callers is re-exported from `pe_tools.signal_finder`:

```python
from pe_tools.signal_finder import (
    ResidualSignalFinder,
    ResidualSignalFinderResult,
    ResidualSignalFinderV2,
    ResidualSignalResult,
    find_residual_signal,
    residualize,
)
```

### Test fixtures

Synthetic data helpers live in `tests/fixtures/`: `synthetic_signal.py` and `synthetic_data.py`. Use these when writing new tests rather than creating inline DataFrames.

### Notebooks

`notebooks/` contains walkthrough notebooks for V1 and V2 against the UCI credit card dataset in `data/`. These are exploration-only; no production logic lives there.
