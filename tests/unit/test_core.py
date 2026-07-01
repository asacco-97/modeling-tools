import numpy as np
import pandas as pd
import pytest

from pe_tools.signal_finder import residualize


def test_residualize_removes_linear_control() -> None:
    control = np.arange(8, dtype=float)
    values = pd.Series((2.0 * control) + 5.0, name="returns")

    residuals = residualize(values, pd.DataFrame({"market": control}))

    assert residuals.name == "returns_residual"
    assert np.allclose(residuals.to_numpy(), 0.0)


def test_residualize_requires_complete_rows() -> None:
    values = pd.Series([np.nan], name="returns")
    controls = pd.DataFrame({"market": [1.0]})

    with pytest.raises(ValueError, match="No complete rows"):
        residualize(values, controls)

