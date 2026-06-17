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
