from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

MetricValue = float | None


def regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series | None,
    warnings: list[str],
) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "mae": _safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _safe_metric(lambda: r2_score(y_true, y_pred), warnings, "R2 undefined"),
        "mape": _mape(y_true, y_pred, warnings),
        "wmape": _wmape(y_true, y_pred, sample_weight, warnings),
        "pearson_corr": _correlation(y_true, y_pred, method="pearson"),
        "spearman_corr": _correlation(y_true, y_pred, method="spearman"),
        "gini": gini_index(y_true, y_pred, sample_weight),
        "normalized_gini": normalized_gini(y_true, y_pred, sample_weight),
    }
    if sample_weight is None:
        metrics.update(
            {
                "weighted_mae": None,
                "weighted_rmse": None,
                "weighted_r2": None,
            }
        )
    else:
        metrics.update(
            {
                "weighted_mae": _safe_float(
                    mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
                ),
                "weighted_rmse": _safe_float(
                    np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
                ),
                "weighted_r2": _safe_metric(
                    lambda: r2_score(y_true, y_pred, sample_weight=sample_weight),
                    warnings,
                    "Weighted R2 undefined",
                ),
            }
        )
    return metrics


def binary_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series | None,
    threshold: float,
    warnings: list[str],
) -> dict[str, MetricValue]:
    probabilities = _clipped_probabilities(y_pred)
    labels = (probabilities >= threshold).astype(int)
    auc = _safe_metric(
        lambda: roc_auc_score(y_true, probabilities, sample_weight=sample_weight),
        warnings,
        "AUC undefined",
    )
    metrics: dict[str, MetricValue] = {
        "auc": auc,
        "gini": None if auc is None else (2.0 * auc) - 1.0,
        "log_loss": _safe_metric(
            lambda: log_loss(y_true, probabilities, sample_weight=sample_weight),
            warnings,
            "Log loss undefined",
        ),
        "brier_score": _safe_float(
            brier_score_loss(y_true, probabilities, sample_weight=sample_weight)
        ),
        "accuracy": _safe_float(accuracy_score(y_true, labels, sample_weight=sample_weight)),
        "precision": _safe_float(
            precision_score(y_true, labels, sample_weight=sample_weight, zero_division=0)
        ),
        "recall": _safe_float(
            recall_score(y_true, labels, sample_weight=sample_weight, zero_division=0)
        ),
        "f1": _safe_float(f1_score(y_true, labels, sample_weight=sample_weight, zero_division=0)),
        "average_precision": _safe_metric(
            lambda: average_precision_score(y_true, probabilities, sample_weight=sample_weight),
            warnings,
            "Average precision undefined",
        ),
    }
    return metrics


def count_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series | None,
    warnings: list[str],
) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "mae": _safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_observed": _safe_float(y_true.mean()),
        "mean_predicted": _safe_float(y_pred.mean()),
        "observed_expected_ratio": _observed_expected_ratio(y_true, y_pred),
        "weighted_observed_expected_ratio": (
            None
            if sample_weight is None
            else _observed_expected_ratio(y_true, y_pred, sample_weight)
        ),
    }
    if (y_true < 0).any() or (y_pred <= 0).any():
        warnings.append(
            "Poisson deviance undefined for negative targets or non-positive predictions"
        )
        metrics["poisson_deviance"] = None
    else:
        metrics["poisson_deviance"] = _safe_metric(
            lambda: mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight),
            warnings,
            "Poisson deviance undefined",
        )
    return metrics


def compute_task_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    task: str,
    sample_weight: pd.Series | None,
    threshold: float,
    warnings: list[str],
) -> dict[str, MetricValue]:
    if task == "regression":
        return regression_metrics(y_true, y_pred, sample_weight, warnings)
    if task == "binary":
        return binary_metrics(y_true, y_pred, sample_weight, threshold, warnings)
    if task in {"count", "rate"}:
        return count_metrics(y_true, y_pred, sample_weight, warnings)
    raise ValueError("task must be one of: regression, binary, count, rate")


def gini_index(
    y_true: pd.Series | np.ndarray[Any, Any],
    y_pred: pd.Series | np.ndarray[Any, Any],
    sample_weight: pd.Series | np.ndarray[Any, Any] | None = None,
) -> MetricValue:
    frame = _ranking_frame(y_true, y_pred, sample_weight)
    if frame["actual"].sum() <= 0:
        return None
    ordered = frame.sort_values("prediction", ascending=False)
    cumulative_weight = ordered["weight"].cumsum() / ordered["weight"].sum()
    cumulative_actual = (ordered["actual"] * ordered["weight"]).cumsum()
    cumulative_actual = cumulative_actual / cumulative_actual.iloc[-1]
    model_area = float(np.trapezoid(cumulative_actual, cumulative_weight))
    return model_area - 0.5


def normalized_gini(
    y_true: pd.Series | np.ndarray[Any, Any],
    y_pred: pd.Series | np.ndarray[Any, Any],
    sample_weight: pd.Series | np.ndarray[Any, Any] | None = None,
) -> MetricValue:
    model_gini = gini_index(y_true, y_pred, sample_weight)
    perfect_gini = gini_index(y_true, y_true, sample_weight)
    if model_gini is None or perfect_gini is None or np.isclose(perfect_gini, 0.0):
        return None
    return model_gini / perfect_gini


def _ranking_frame(
    y_true: pd.Series | np.ndarray[Any, Any],
    y_pred: pd.Series | np.ndarray[Any, Any],
    sample_weight: pd.Series | np.ndarray[Any, Any] | None,
) -> pd.DataFrame:
    actual = pd.Series(np.asarray(y_true, dtype=float), name="actual")
    prediction = pd.Series(np.asarray(y_pred, dtype=float), name="prediction")
    if sample_weight is None:
        weight = pd.Series(np.ones(len(actual)), name="weight")
    else:
        weight = pd.Series(np.asarray(sample_weight, dtype=float), name="weight")
    return pd.concat([actual, prediction, weight], axis=1).dropna()


def _safe_metric(
    function: Any,
    warnings: list[str],
    warning_message: str,
) -> MetricValue:
    try:
        value = function()
    except ValueError:
        warnings.append(warning_message)
        return None
    return _safe_float(value)


def _safe_float(value: Any) -> MetricValue:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _mape(y_true: pd.Series, y_pred: pd.Series, warnings: list[str]) -> MetricValue:
    non_zero = y_true != 0
    if not non_zero.all():
        warnings.append("MAPE skipped zero actual values")
    if not non_zero.any():
        return None
    return _safe_float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])))


def _wmape(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series | None,
    warnings: list[str],
) -> MetricValue:
    weight = 1.0 if sample_weight is None else sample_weight
    numerator = float(np.sum(np.abs(y_true - y_pred) * weight))
    denominator = float(np.sum(np.abs(y_true) * weight))
    if np.isclose(denominator, 0.0):
        warnings.append("WMAPE undefined because absolute actual total is zero")
        return None
    return numerator / denominator


def _correlation(y_true: pd.Series, y_pred: pd.Series, method: str) -> MetricValue:
    if y_true.nunique() < 2 or y_pred.nunique() < 2:
        return None
    return _safe_float(y_true.corr(y_pred, method=method))


def _observed_expected_ratio(
    y_true: pd.Series,
    y_pred: pd.Series,
    sample_weight: pd.Series | None = None,
) -> MetricValue:
    weight = 1.0 if sample_weight is None else sample_weight
    observed = float(np.sum(y_true * weight))
    expected = float(np.sum(y_pred * weight))
    if np.isclose(expected, 0.0):
        return None
    return observed / expected


def _clipped_probabilities(values: pd.Series) -> pd.Series:
    return values.clip(1e-15, 1.0 - 1e-15)
