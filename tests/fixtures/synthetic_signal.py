from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticResidualData:
    X: pd.DataFrame
    y_true: pd.Series
    y_pred: pd.Series
    sample_weight: pd.Series
    split_col: pd.Series


def make_synthetic_residual_data(row_count: int = 120) -> SyntheticResidualData:
    rng = np.random.default_rng(123)
    residual_signal = rng.normal(size=row_count)
    noise = rng.normal(scale=0.03, size=row_count)
    weak_signal = rng.normal(size=row_count)
    unrelated = rng.normal(size=row_count)

    y_pred = pd.Series(rng.normal(size=row_count), name="base_prediction")
    residual = (2.5 * residual_signal) + noise
    y_true = pd.Series(y_pred.to_numpy() + residual, name="actual")
    X = pd.DataFrame(
        {
            "residual_signal": residual_signal,
            "weak_signal": weak_signal,
            "unrelated": unrelated,
            "group": np.repeat(np.arange(row_count // 10), 10),
        }
    )
    sample_weight = pd.Series(np.linspace(0.5, 2.0, row_count), name="weight")
    split_col = pd.Series(
        np.where(
            np.arange(row_count) < 80,
            "train",
            np.where(np.arange(row_count) < 100, "validation", "holdout"),
        ),
        name="split",
    )
    return SyntheticResidualData(
        X=X,
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=sample_weight,
        split_col=split_col,
    )

