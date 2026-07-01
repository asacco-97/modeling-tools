import numpy as np
import pandas as pd

from pe_tools.signal_finder import find_residual_signal


def test_find_residual_signal_identifies_relationship_after_controls() -> None:
    rng = np.random.default_rng(7)
    row_count = 80
    market = rng.normal(size=row_count)
    residual_driver = rng.normal(size=row_count)
    signal = (1.5 * market) + residual_driver
    target = (-0.7 * market) + (0.8 * residual_driver) + rng.normal(scale=0.05, size=row_count)
    data = pd.DataFrame({"target": target, "signal": signal, "market": market})

    result = find_residual_signal(data, target="target", signal="signal", controls=["market"])

    assert result.n_obs == row_count
    assert result.correlation > 0.95
    assert result.target_residual.index.equals(data.index)
    assert result.signal_residual.index.equals(data.index)

