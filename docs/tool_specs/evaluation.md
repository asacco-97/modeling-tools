# Model Evaluation Module Specification

## Purpose

Build a reusable model evaluation module for tabular prediction workflows.

The module should evaluate model performance across:

1. Overall predictive metrics
2. Calibration
3. Lift / ranking performance
4. Segment-level performance
5. Feature-split / variable-binned diagnostics
6. Stability across folds, groups, time periods, or user-provided splits
7. Visual diagnostics

This module should work with model outputs from any modeling system, including:

* Bayesian models
* GLMs
* Random forests
* XGBoost / LightGBM
* Neural networks
* External/proprietary scoring systems

The module should not require access to the model object itself. It should primarily operate on actual values, predicted values, optional probabilities, optional sample weights, optional exposure weights, and optional grouping/splitting columns.

---

# Core Design Principle

The evaluation module should answer:

> How well does the model perform, where does it perform poorly, and whether performance is stable across meaningful slices of the data?

It should be model-agnostic and output structured results that are easy to inspect in notebooks, export to reports, or use downstream in diagnostics.

---

# Target Use Cases

## Regression

Examples:

* Loss cost prediction
* Severity prediction
* MOIC prediction
* IRR prediction
* Revenue growth prediction
* EBITDA growth prediction

Common metrics:

* RMSE
* MAE
* MAPE / WMAPE
* R²
* Weighted R²
* Spearman correlation
* Pearson correlation
* Gini index / normalized Gini, where appropriate
* Lift by prediction quantile
* Calibration by prediction bin

## Binary Classification

Examples:

* Default probability
* Deal close probability
* Churn probability
* Claim occurrence
* Conversion probability

Common metrics:

* AUC
* Gini = 2 * AUC - 1
* Log loss
* Brier score
* Accuracy
* Precision
* Recall
* F1
* Calibration by probability bin
* Lift / gains / capture curves

## Count / Frequency / Rate Models

Examples:

* Claim frequency
* Conversion count
* Incidents per exposure
* Event rates

Common metrics:

* Mean observed vs expected
* Weighted observed / expected ratio
* Lift by prediction quantile
* Calibration by prediction bin
* Segment O/E ratios

---

# Public API

Implement a primary evaluator object:

```python
from pe_tools.evaluation import ModelEvaluator

evaluator = ModelEvaluator(
    y_true=y_true,
    y_pred=y_pred,
    task="regression",
    sample_weight=None,
    exposure=None,
    group_cols=None,
    split_col=None,
    feature_frame=X,
)

result = evaluator.evaluate(
    metrics="auto",
    calibration=True,
    lift=True,
    segments=["sector", "vintage_year"],
    feature_splits=["company_size", "entry_multiple", "sector"],
    n_bins=10,
)
```

Alternative functional API should also be supported:

```python
from pe_tools.evaluation import evaluate_model

result = evaluate_model(
    y_true=y_true,
    y_pred=y_pred,
    task="binary",
    sample_weight=weights,
    feature_frame=X,
    segments=["sector"],
    feature_splits=["deal_size", "leverage_ratio"],
)
```

---

# Inputs

## Required

### `y_true`

Actual observed target values.

Accepted types:

* pandas Series
* numpy array
* polars Series if supported
* list-like

### `y_pred`

Predicted values.

For regression:

* continuous predictions

For binary classification:

* predicted probability for positive class

For count/rate models:

* predicted expected count or expected rate

---

## Optional

### `sample_weight`

Observation-level weights.

Used for:

* weighted metrics
* weighted calibration
* weighted segment evaluation
* weighted lift analysis

### `exposure`

Exposure or denominator column.

Used for:

* frequency/rate models
* observed-to-expected ratios
* weighted rate calculations

### `feature_frame`

DataFrame containing features and segment columns.

Used for:

* segment evaluation
* variable split diagnostics
* feature-binned performance plots

### `segments`

List of columns to use for segment-level evaluation.

Example:

```python
segments=["sector", "vintage_year", "geography"]
```

### `feature_splits`

List of columns used for variable-level diagnostics.

For each feature, the module should bin or group observations and evaluate performance within each bin/category.

Example:

```python
feature_splits=["company_size", "entry_multiple", "revenue_growth", "sector"]
```

### `split_col`

Optional column identifying evaluation splits.

Example:

```python
split_col="fold"
split_col="vintage_period"
split_col="train_test_split"
split_col="time_period"
```

Used for:

* stability analysis
* fold-level metrics
* time-split evaluation
* comparing performance across predefined validation groups

### `n_bins`

Default number of bins for continuous features and prediction quantiles.

Default:

```python
n_bins=10
```

### `binning_strategy`

Supported values:

```python
"quantile"
"uniform"
"custom"
```

Default:

```python
"quantile"
```

---

# Outputs

The evaluator should return a structured result object:

```python
EvaluationResult
```

The result object should contain:

```python
result.overall_metrics
result.calibration_table
result.lift_table
result.segment_metrics
result.feature_split_metrics
result.stability_metrics
result.figures
result.warnings
```

The result object should have convenience methods:

```python
result.summary()
result.to_dict()
result.to_frame()
result.plot_calibration()
result.plot_lift()
result.plot_segment_performance()
result.plot_feature_split("feature_name")
result.plot_stability()
```

---

# Required Components

## 1. Overall Metrics

Implement overall metric computation for each task type.

### Regression Metrics

Required:

* `mae`
* `rmse`
* `r2`
* `weighted_mae`
* `weighted_rmse`
* `weighted_r2`
* `mape`
* `wmape`
* `pearson_corr`
* `spearman_corr`
* `gini`
* `normalized_gini`

Notes:

* Metrics requiring weights should return `None` or be omitted if weights are unavailable.
* MAPE should handle zero actual values safely.
* WMAPE should avoid division-by-zero failures.
* Gini should support continuous targets where relevant, especially insurance-style ranking evaluation.

### Binary Classification Metrics

Required:

* `auc`
* `gini`
* `log_loss`
* `brier_score`
* `accuracy`
* `precision`
* `recall`
* `f1`
* `average_precision`

Notes:

* `gini = 2 * auc - 1`
* Classification threshold should default to `0.5`, but be configurable.
* Probability metrics should assume `y_pred` is a probability.

### Count / Rate Metrics

Required:

* `mae`
* `rmse`
* `poisson_deviance`
* `mean_observed`
* `mean_predicted`
* `observed_expected_ratio`
* `weighted_observed_expected_ratio`

---

## 2. Calibration Analysis

Calibration should compare predicted values to observed outcomes.

### For Regression

Create a calibration table by prediction bins:

Columns:

* `bin`
* `bin_lower`
* `bin_upper`
* `n`
* `weighted_n`
* `mean_prediction`
* `mean_actual`
* `residual`
* `abs_residual`
* `observed_expected_ratio`

### For Binary Classification

Create a calibration table by predicted probability bins:

Columns:

* `bin`
* `bin_lower`
* `bin_upper`
* `n`
* `weighted_n`
* `mean_predicted_probability`
* `observed_rate`
* `calibration_error`
* `abs_calibration_error`
* `observed_expected_ratio`

### Calibration Metrics

Implement:

* Expected calibration error, `ece`
* Maximum calibration error, `mce`
* Integrated calibration index, optional
* Brier score for binary classification

### Calibration Plots

Implement:

```python
result.plot_calibration()
```

The plot should show:

* x-axis: mean predicted value or probability
* y-axis: mean observed value or observed rate
* 45-degree reference line
* point size optionally proportional to bin count
* optional error bars if binomial/regression uncertainty is available

---

## 3. Lift Analysis

Lift analysis should evaluate whether high predictions correspond to high observed outcomes.

Create prediction quantile bins, ordered from lowest prediction to highest prediction.

### Lift Table Columns

Required:

* `quantile_bin`
* `n`
* `weighted_n`
* `min_prediction`
* `max_prediction`
* `mean_prediction`
* `mean_actual`
* `total_actual`
* `total_predicted`
* `observed_expected_ratio`
* `lift_vs_overall`
* `cumulative_actual`
* `cumulative_predicted`
* `cumulative_capture_rate`

For binary classification:

* `event_count`
* `event_rate`
* `cumulative_event_capture_rate`

### Lift Plots

Implement:

```python
result.plot_lift()
result.plot_cumulative_gain()
```

`plot_lift()` should show:

* x-axis: prediction quantile
* y-axis: observed outcome, event rate, or lift

`plot_cumulative_gain()` should show:

* x-axis: cumulative population share
* y-axis: cumulative actual/event capture share
* baseline diagonal reference line

---

## 4. Segment Evaluation

Support evaluation by one or more user-provided segment columns.

Example:

```python
segments=["sector", "vintage_year"]
```

The module should compute performance separately for each segment.

### Segment Table Columns

Required:

* `segment_column`
* `segment_value`
* `n`
* `weighted_n`
* `mean_actual`
* `mean_prediction`
* `residual`
* `abs_residual`
* `observed_expected_ratio`
* task-specific metrics

For regression:

* `mae`
* `rmse`
* `r2`
* `weighted_mae`
* `weighted_rmse`

For binary:

* `auc`
* `gini`
* `log_loss`
* `brier_score`
* `event_rate`
* `mean_predicted_probability`

For count/rate:

* `poisson_deviance`
* `observed_expected_ratio`

### Segment Reliability Rules

Segments with low sample size should be flagged.

Configurable parameters:

```python
min_segment_n=30
min_segment_events=5
```

If a segment is too small:

* still compute simple aggregates
* suppress unstable metrics like AUC if needed
* add warning flag

### Segment Plots

Implement:

```python
result.plot_segment_performance(metric="observed_expected_ratio")
```

Useful segment plots:

* observed vs predicted by segment
* O/E ratio by segment
* error by segment
* AUC/Gini by segment
* sample size by segment

---

## 5. Feature Split / Variable Diagnostics

The evaluator should support variable-level plots and tabulations.

This is different from model feature importance.

The goal is to answer:

> Where does the model overpredict or underpredict across values of a feature?

Example:

```python
result.feature_split_metrics["entry_multiple"]
```

### Continuous Features

For continuous features:

* bin into quantiles by default
* support custom bins
* compute metrics within each bin

Columns:

* `feature`
* `bin`
* `bin_lower`
* `bin_upper`
* `n`
* `weighted_n`
* `mean_feature_value`
* `mean_actual`
* `mean_prediction`
* `residual`
* `abs_residual`
* `observed_expected_ratio`
* task-specific metrics

### Categorical Features

For categorical features:

* group by category
* optionally combine rare categories into `"Other"`
* compute metrics within each category

Columns:

* `feature`
* `category`
* `n`
* `weighted_n`
* `mean_actual`
* `mean_prediction`
* `residual`
* `abs_residual`
* `observed_expected_ratio`
* task-specific metrics

### Variable Diagnostic Plots

Implement:

```python
result.plot_feature_split("feature_name")
```

For continuous variables:

* x-axis: feature bin or mean feature value
* y-axis: actual and predicted
* secondary optional view: residual or O/E ratio

For categorical variables:

* x-axis: category
* y-axis: actual and predicted
* sort by volume or residual magnitude

Also support:

```python
result.plot_top_feature_errors(top_n=10)
```

This should rank features by magnitude of systematic error across bins/categories.

---

## 6. Stability Evaluation

The module should support stability evaluation across:

* folds
* validation periods
* time periods
* vintages
* groups
* user-supplied split column

Example:

```python
result = evaluate_model(
    y_true=y,
    y_pred=pred,
    split_col=df["fold"],
)
```

### Stability Table Columns

Required:

* `split_value`
* `n`
* `weighted_n`
* `mean_actual`
* `mean_prediction`
* `observed_expected_ratio`
* task-specific metrics
* `metric_delta_vs_overall`

### Stability Metrics

For each selected metric, compute:

* mean across splits
* standard deviation across splits
* min
* max
* range
* coefficient of variation
* worst split
* best split

### Stability Plots

Implement:

```python
result.plot_stability(metric="gini")
```

Plot metric values by split.

Useful metrics:

* AUC
* Gini
* RMSE
* MAE
* O/E ratio
* calibration error
* lift top decile

---

## 7. Gini / AUC Implementation

Implement both AUC-based Gini and continuous-target Gini.

### Binary Gini

For binary classification:

```python
gini = 2 * auc - 1
```

### Continuous / Insurance-Style Gini

For continuous target ranking:

* Sort observations by prediction descending.
* Calculate cumulative share of exposure or population.
* Calculate cumulative share of actual target.
* Compute area between model curve and baseline.
* Normalize by perfect model curve if requested.

Expose:

```python
gini_index(y_true, y_pred, sample_weight=None)
normalized_gini(y_true, y_pred, sample_weight=None)
```

The implementation should be tested carefully.

---

## 8. Warnings and Edge Cases

The module should collect warnings in:

```python
result.warnings
```

Required warning cases:

* Missing values in `y_true`
* Missing values in `y_pred`
* Constant predictions
* Constant target
* Very small sample size
* Segment too small
* Segment has no events
* AUC undefined
* R² undefined
* MAPE division by zero
* Negative predictions for count/rate task
* Non-probability predictions for binary classification
* Duplicate index alignment issues

Do not silently fail.

Prefer:

* clear warnings
* stable fallback behavior
* no hard crashes unless input is unusable

---

## 9. Visualization Standards

All plotting methods should return Matplotlib `Figure` objects.

Do not call `plt.show()` inside library code.

Do not call `plt.close()` inside library code.

The user/notebook should control display.

Example:

```python
fig = result.plot_calibration()
display(fig)
```

For multiple figures:

```python
figures = result.plot_all()
for name, fig in figures.items():
    display(fig)
```

Each plot should include:

* title
* axis labels
* readable tick labels
* reference line where relevant
* legend where relevant

---

## 10. Result Object

Implement:

```python
@dataclass
class EvaluationResult:
    overall_metrics: dict
    calibration_table: pd.DataFrame | None
    lift_table: pd.DataFrame | None
    segment_metrics: pd.DataFrame | None
    feature_split_metrics: dict[str, pd.DataFrame]
    stability_metrics: pd.DataFrame | None
    figures: dict[str, matplotlib.figure.Figure]
    warnings: list[str]
```

Convenience methods:

```python
summary()
to_dict()
to_frame()
plot_calibration()
plot_lift()
plot_cumulative_gain()
plot_segment_performance(metric="observed_expected_ratio")
plot_feature_split(feature)
plot_top_feature_errors(top_n=10)
plot_stability(metric)
plot_all()
```

---

# Suggested File Structure

Implement the module under:

```text
src/pe_tools/evaluation/
```

Suggested files:

```text
src/pe_tools/evaluation/
    __init__.py
    evaluator.py
    result.py
    metrics.py
    calibration.py
    lift.py
    segments.py
    feature_splits.py
    stability.py
    plots.py
    validation.py
```

Tests:

```text
tests/evaluation/
    test_metrics.py
    test_calibration.py
    test_lift.py
    test_segments.py
    test_feature_splits.py
    test_stability.py
    test_evaluator.py
```

---

# Implementation Phases

## Phase 1: Core Metrics

Implement:

* `EvaluationResult`
* `evaluate_model`
* `ModelEvaluator`
* regression metrics
* binary metrics
* basic warnings

Do not implement plots yet.

## Phase 2: Calibration and Lift

Implement:

* calibration tables
* lift tables
* binary cumulative gains
* regression lift

Add tests.

## Phase 3: Segment Evaluation

Implement:

* grouped segment metrics
* low-volume warnings
* support multiple segment columns

Add tests.

## Phase 4: Feature Split Diagnostics

Implement:

* continuous feature binning
* categorical feature grouping
* feature-level metric tables
* top feature error ranking

Add tests.

## Phase 5: Stability Analysis

Implement:

* split-column evaluation
* fold/time/group stability summaries
* metric variation summaries

Add tests.

## Phase 6: Visualizations

Implement plotting methods.

All plotting methods must return Matplotlib `Figure` objects and must not call `plt.show()` or `plt.close()` internally.

---

# Acceptance Criteria

The module is complete when:

1. It can evaluate regression and binary classification models.
2. It can compute overall metrics.
3. It can produce calibration tables.
4. It can produce lift tables.
5. It can evaluate performance by segment.
6. It can evaluate performance by feature bins/categories.
7. It can evaluate metric stability across a supplied split column.
8. It returns a structured `EvaluationResult`.
9. It has tests for all core computations.
10. It does not rely on model objects.
11. It handles bad inputs with explicit warnings.
12. Plots return valid Matplotlib figures.
13. No plotting function closes figures internally.

---

# Example Usage

```python
from pe_tools.evaluation import evaluate_model

result = evaluate_model(
    y_true=df["actual"],
    y_pred=df["prediction"],
    task="regression",
    sample_weight=df.get("weight"),
    feature_frame=df,
    segments=["sector", "vintage_year"],
    feature_splits=["company_size", "entry_multiple", "leverage_ratio"],
    split_col=df.get("fold"),
    n_bins=10,
)

result.overall_metrics
result.calibration_table
result.lift_table
result.segment_metrics
result.feature_split_metrics["entry_multiple"]
result.stability_metrics

fig = result.plot_calibration()
display(fig)

fig = result.plot_lift()
display(fig)

fig = result.plot_feature_split("entry_multiple")
display(fig)
```

Binary classification example:

```python
result = evaluate_model(
    y_true=df["default_flag"],
    y_pred=df["predicted_default_probability"],
    task="binary",
    feature_frame=df,
    segments=["sector"],
    feature_splits=["company_size", "leverage_ratio"],
    split_col=df["validation_period"],
)

result.overall_metrics["auc"]
result.overall_metrics["gini"]
result.plot_calibration()
result.plot_cumulative_gain()
```

---

# Important Notes for Codex

Before coding:

1. Inspect existing repository structure.
2. Reuse existing splitting/binning utilities from the residual signal finder if available.
3. Match naming conventions already used in the repository.
4. Do not duplicate existing validation utilities.
5. Implement incrementally.
6. Add tests before expanding visualization complexity.
7. Keep plotting code separate from metric computation.
8. Do not introduce heavyweight dependencies unless necessary.

The first implementation should prioritize correctness and API cleanliness over visual polish.
