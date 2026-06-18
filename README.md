# Modeling Tools

Minimal Python package for finding residual signals in tabular modeling data.

The intended use case is simple factor research: remove known controls from a
candidate signal and a target, then measure whether the remaining residuals still
move together. This helps separate a genuinely useful signal from exposure that
is already explained by common drivers.

```python
from pe_tools.signal_finder import find_residual_signal

result = find_residual_signal(
    data,
    target="forward_return",
    signal="quality_score",
    controls=["size", "value", "momentum"],
)

print(result.correlation)
```

The implementation is intentionally small: pandas for data handling,
scikit-learn for linear residualization, and a lightweight result object.

`ResidualSignalFinder` can also scan model errors directly. For 0/1 targets it
automatically treats the task as classification, models the target again with
the original prediction included as a control, and emits a warning describing
that behavior. Set `offbalance=True` to level held-out predictions to the target
mean within each split before scoring and binned diagnostics.

When SHAP interaction values are available, `plot_top_interactions()` creates
saved heatmap figures for the strongest feature pairs.

Residual diagnostic plots show out-of-sample binned actuals, predictions, and
paired errors with 95% confidence intervals; the error series is shown in a
separate lower panel with a centered zero reference line.
