from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class ResidualSignalResult:
    """Residualized target/signal pair and their correlation."""

    target_residual: pd.Series
    signal_residual: pd.Series
    correlation: float
    n_obs: int


def residualize(values: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """Remove linear exposure to controls from a series."""
    frame = pd.concat([values.rename("value"), controls], axis=1).dropna()
    if frame.empty:
        raise ValueError("No complete rows are available for residualization.")

    target_values = frame["value"]
    control_values = frame.drop(columns="value")
    if control_values.shape[1] == 0:
        residual_values = target_values.to_numpy() - float(target_values.mean())
    else:
        model = LinearRegression()
        control_array = control_values.to_numpy()
        target_array = target_values.to_numpy()
        model.fit(control_array, target_array)
        residual_values = target_array - model.predict(control_array)

    residual_name = f"{values.name}_residual" if values.name else "residual"
    return pd.Series(residual_values, index=frame.index, name=residual_name)


def find_residual_signal(
    data: pd.DataFrame,
    target: str,
    signal: str,
    controls: Sequence[str] = (),
) -> ResidualSignalResult:
    """Residualize a target and signal, then report their correlation."""
    required_columns = [target, signal, *controls]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    frame = data.loc[:, required_columns].dropna()
    if frame.empty:
        raise ValueError("No complete rows are available for signal finding.")

    control_frame = frame.loc[:, list(controls)] if controls else pd.DataFrame(index=frame.index)
    target_residual = residualize(frame[target], control_frame)
    signal_residual = residualize(frame[signal], control_frame)

    residual_frame = pd.concat([target_residual, signal_residual], axis=1).dropna()
    if len(residual_frame) < 2:
        raise ValueError("At least two complete residual observations are required.")

    target_std = float(residual_frame.iloc[:, 0].std())
    signal_std = float(residual_frame.iloc[:, 1].std())
    if np.isclose(target_std, 0.0) or np.isclose(signal_std, 0.0):
        raise ValueError("Residuals must have non-zero variation.")

    correlation = float(
        np.corrcoef(residual_frame.iloc[:, 0].to_numpy(), residual_frame.iloc[:, 1].to_numpy())[
            0, 1
        ]
    )
    return ResidualSignalResult(
        target_residual=target_residual,
        signal_residual=signal_residual,
        correlation=correlation,
        n_obs=len(residual_frame),
    )

