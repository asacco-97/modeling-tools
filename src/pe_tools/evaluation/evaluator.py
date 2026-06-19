from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

from pe_tools.evaluation.metrics import MetricValue, compute_task_metrics
from pe_tools.evaluation.result import EvaluationResult

Task = Literal["regression", "binary", "count", "rate"]
BinningStrategy = Literal["quantile", "uniform", "custom"]
SeriesLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[float]
LabelLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[Any] | str | None


@dataclass
class ModelEvaluator:
    """Model-agnostic evaluator for tabular prediction outputs."""

    y_true: SeriesLike
    y_pred: SeriesLike
    task: Task = "regression"
    sample_weight: SeriesLike | None = None
    exposure: SeriesLike | None = None
    group_cols: Sequence[str] | None = None
    split_col: LabelLike = None
    feature_frame: pd.DataFrame | None = None

    def evaluate(
        self,
        metrics: str | Sequence[str] = "auto",
        calibration: bool = True,
        lift: bool = True,
        segments: Sequence[str] | None = None,
        feature_splits: Sequence[str] | None = None,
        n_bins: int = 10,
        binning_strategy: BinningStrategy = "quantile",
        threshold: float = 0.5,
        min_segment_n: int = 30,
        min_segment_events: int = 5,
    ) -> EvaluationResult:
        if metrics != "auto":
            raise ValueError("Only metrics='auto' is supported in this initial implementation")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if binning_strategy not in {"quantile", "uniform", "custom"}:
            raise ValueError("binning_strategy must be one of: quantile, uniform, custom")
        if binning_strategy == "custom":
            raise ValueError("custom binning requires explicit bins and is not implemented yet")

        warnings: list[str] = []
        frame = _evaluation_frame(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sample_weight=self.sample_weight,
            exposure=self.exposure,
            feature_frame=self.feature_frame,
            split_col=self.split_col,
            warnings=warnings,
        )
        task = _normalize_task(self.task)
        _warn_on_data_issues(frame, task, warnings)
        sample_weight = frame["sample_weight"] if "sample_weight" in frame.columns else None
        overall_metrics = compute_task_metrics(
            frame["y_true"],
            frame["y_pred"],
            task,
            sample_weight,
            threshold,
            warnings,
        )

        calibration_table = (
            _calibration_table(frame, task, n_bins, binning_strategy)
            if calibration
            else None
        )
        if calibration_table is not None and not calibration_table.empty:
            overall_metrics.update(_calibration_metrics(calibration_table, task))

        lift_table = _lift_table(frame, task, n_bins, binning_strategy) if lift else None
        segment_columns = list(segments or self.group_cols or [])
        segment_metrics = (
            _segment_metrics(
                frame,
                task,
                segment_columns,
                threshold,
                min_segment_n,
                min_segment_events,
                warnings,
            )
            if segment_columns
            else None
        )
        feature_split_metrics = (
            _feature_split_metrics(
                frame,
                task,
                list(feature_splits),
                n_bins,
                binning_strategy,
                threshold,
                warnings,
            )
            if feature_splits
            else {}
        )
        stability_metrics = (
            _stability_metrics(frame, task, threshold, overall_metrics, warnings)
            if "split_col" in frame.columns
            else None
        )

        return EvaluationResult(
            overall_metrics=overall_metrics,
            calibration_table=calibration_table,
            lift_table=lift_table,
            segment_metrics=segment_metrics,
            feature_split_metrics=feature_split_metrics,
            stability_metrics=stability_metrics,
            warnings=warnings,
        )


def evaluate_model(
    *,
    y_true: SeriesLike,
    y_pred: SeriesLike,
    task: Task = "regression",
    sample_weight: SeriesLike | None = None,
    exposure: SeriesLike | None = None,
    group_cols: Sequence[str] | None = None,
    split_col: LabelLike = None,
    feature_frame: pd.DataFrame | None = None,
    metrics: str | Sequence[str] = "auto",
    calibration: bool = True,
    lift: bool = True,
    segments: Sequence[str] | None = None,
    feature_splits: Sequence[str] | None = None,
    n_bins: int = 10,
    binning_strategy: BinningStrategy = "quantile",
    threshold: float = 0.5,
    min_segment_n: int = 30,
    min_segment_events: int = 5,
) -> EvaluationResult:
    """Functional wrapper around ModelEvaluator."""
    evaluator = ModelEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        task=task,
        sample_weight=sample_weight,
        exposure=exposure,
        group_cols=group_cols,
        split_col=split_col,
        feature_frame=feature_frame,
    )
    return evaluator.evaluate(
        metrics=metrics,
        calibration=calibration,
        lift=lift,
        segments=segments,
        feature_splits=feature_splits,
        n_bins=n_bins,
        binning_strategy=binning_strategy,
        threshold=threshold,
        min_segment_n=min_segment_n,
        min_segment_events=min_segment_events,
    )


def _evaluation_frame(
    *,
    y_true: SeriesLike,
    y_pred: SeriesLike,
    sample_weight: SeriesLike | None,
    exposure: SeriesLike | None,
    feature_frame: pd.DataFrame | None,
    split_col: LabelLike,
    warnings: list[str],
) -> pd.DataFrame:
    y_true_series = _coerce_numeric_series(y_true, "y_true")
    y_pred_series = _coerce_numeric_series(y_pred, "y_pred")
    if len(y_true_series) != len(y_pred_series):
        raise ValueError("y_true and y_pred must have the same length")
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series) and not y_true.index.equals(
        y_pred.index
    ):
        warnings.append("y_true and y_pred indexes differ; values were aligned by position")

    frame = pd.DataFrame(
        {
            "y_true": y_true_series.to_numpy(),
            "y_pred": y_pred_series.to_numpy(),
        }
    )
    if sample_weight is not None:
        weight = _coerce_numeric_series(sample_weight, "sample_weight")
        if len(weight) != len(frame):
            raise ValueError("sample_weight must have the same length as y_true")
        if (weight < 0).any():
            raise ValueError("sample_weight cannot contain negative values")
        frame["sample_weight"] = weight.to_numpy()
    if exposure is not None:
        exposure_series = _coerce_numeric_series(exposure, "exposure")
        if len(exposure_series) != len(frame):
            raise ValueError("exposure must have the same length as y_true")
        if (exposure_series < 0).any():
            raise ValueError("exposure cannot contain negative values")
        frame["exposure"] = exposure_series.to_numpy()
    if feature_frame is not None:
        if not isinstance(feature_frame, pd.DataFrame):
            raise ValueError("feature_frame must be a pandas DataFrame")
        if len(feature_frame) != len(frame):
            raise ValueError("feature_frame must have the same length as y_true")
        frame = pd.concat([frame, feature_frame.reset_index(drop=True)], axis=1)
    split_values = _resolve_split_col(split_col, feature_frame)
    if split_values is not None:
        if len(split_values) != len(frame):
            raise ValueError("split_col must have the same length as y_true")
        frame["split_col"] = pd.Series(split_values).to_numpy()

    missing_mask = frame.loc[:, ["y_true", "y_pred"]].isna().any(axis=1)
    if "sample_weight" in frame.columns:
        missing_mask = missing_mask | frame["sample_weight"].isna()
    if "exposure" in frame.columns:
        missing_mask = missing_mask | frame["exposure"].isna()
    if missing_mask.any():
        warnings.append(f"Dropped {int(missing_mask.sum())} rows with missing required values")
        frame = frame.loc[~missing_mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete rows are available for evaluation")
    return frame


def _resolve_split_col(
    split_col: LabelLike,
    feature_frame: pd.DataFrame | None,
) -> pd.Series | None:
    if split_col is None:
        return None
    if isinstance(split_col, str):
        if feature_frame is None or split_col not in feature_frame.columns:
            raise ValueError("split_col string must refer to a column in feature_frame")
        return feature_frame[split_col].reset_index(drop=True)
    return pd.Series(np.asarray(split_col), name="split_col")


def _warn_on_data_issues(frame: pd.DataFrame, task: str, warnings: list[str]) -> None:
    if len(frame) < 30:
        warnings.append("Very small sample size")
    if frame["y_true"].nunique() < 2:
        warnings.append("Constant target")
    if frame["y_pred"].nunique() < 2:
        warnings.append("Constant predictions")
    if task == "binary" and ((frame["y_pred"] < 0).any() or (frame["y_pred"] > 1).any()):
        warnings.append("Non-probability predictions for binary classification were clipped")
        frame["y_pred"] = frame["y_pred"].clip(0.0, 1.0)
    if task in {"count", "rate"} and (frame["y_pred"] < 0).any():
        warnings.append("Negative predictions for count/rate task")


def _calibration_table(
    frame: pd.DataFrame,
    task: str,
    n_bins: int,
    binning_strategy: BinningStrategy,
) -> pd.DataFrame:
    source = frame.copy()
    source["bin"] = _numeric_bins(source["y_pred"], n_bins, binning_strategy)
    grouped = source.groupby("bin", observed=True, dropna=False)
    table = grouped.apply(_calibration_row, include_groups=False).reset_index()
    if task == "binary":
        table = table.rename(
            columns={
                "mean_prediction": "mean_predicted_probability",
                "mean_actual": "observed_rate",
                "residual": "calibration_error",
                "abs_residual": "abs_calibration_error",
            }
        )
    return table


def _calibration_row(group: pd.DataFrame) -> pd.Series:
    weight = group["sample_weight"] if "sample_weight" in group.columns else None
    mean_prediction = _weighted_mean(group["y_pred"], weight)
    mean_actual = _weighted_mean(group["y_true"], weight)
    residual = mean_actual - mean_prediction
    return pd.Series(
        {
            "bin_lower": float(group["y_pred"].min()),
            "bin_upper": float(group["y_pred"].max()),
            "n": int(len(group)),
            "weighted_n": _weighted_n(group),
            "mean_prediction": mean_prediction,
            "mean_actual": mean_actual,
            "residual": residual,
            "abs_residual": abs(residual),
            "observed_expected_ratio": _oe_ratio(group["y_true"], group["y_pred"], weight),
        }
    )


def _calibration_metrics(table: pd.DataFrame, task: str) -> dict[str, MetricValue]:
    error_col = "abs_calibration_error" if task == "binary" else "abs_residual"
    weighted_error = table[error_col] * table["n"]
    return {
        "ece": float(weighted_error.sum() / table["n"].sum()),
        "mce": float(table[error_col].max()),
    }


def _lift_table(
    frame: pd.DataFrame,
    task: str,
    n_bins: int,
    binning_strategy: BinningStrategy,
) -> pd.DataFrame:
    source = frame.copy()
    source["quantile_bin"] = _numeric_bins(source["y_pred"], n_bins, binning_strategy)
    table = (
        source.groupby("quantile_bin", observed=True, dropna=False)
        .apply(_lift_row, include_groups=False)
        .reset_index()
        .sort_values("quantile_bin", ignore_index=True)
    )
    overall_actual = float(source["y_true"].mean())
    if np.isclose(overall_actual, 0.0):
        table["lift_vs_overall"] = np.nan
    else:
        table["lift_vs_overall"] = table["mean_actual"] / overall_actual
    table["cumulative_actual"] = table["total_actual"].cumsum()
    table["cumulative_predicted"] = table["total_predicted"].cumsum()
    total_actual = float(table["total_actual"].sum())
    table["cumulative_capture_rate"] = (
        np.nan if np.isclose(total_actual, 0.0) else table["cumulative_actual"] / total_actual
    )
    table["cumulative_population_share"] = table["n"].cumsum() / table["n"].sum()
    if task == "binary":
        table["event_count"] = table["total_actual"]
        table["event_rate"] = table["mean_actual"]
        table["cumulative_event_capture_rate"] = table["cumulative_capture_rate"]
    return table


def _lift_row(group: pd.DataFrame) -> pd.Series:
    weight = group["sample_weight"] if "sample_weight" in group.columns else None
    total_actual = _weighted_sum(group["y_true"], weight)
    total_predicted = _weighted_sum(group["y_pred"], weight)
    return pd.Series(
        {
            "n": int(len(group)),
            "weighted_n": _weighted_n(group),
            "min_prediction": float(group["y_pred"].min()),
            "max_prediction": float(group["y_pred"].max()),
            "mean_prediction": _weighted_mean(group["y_pred"], weight),
            "mean_actual": _weighted_mean(group["y_true"], weight),
            "total_actual": total_actual,
            "total_predicted": total_predicted,
            "observed_expected_ratio": _ratio(total_actual, total_predicted),
        }
    )


def _segment_metrics(
    frame: pd.DataFrame,
    task: str,
    segments: Sequence[str],
    threshold: float,
    min_segment_n: int,
    min_segment_events: int,
    warnings: list[str],
) -> pd.DataFrame:
    rows = []
    for segment in segments:
        if segment not in frame.columns:
            warnings.append(f"Segment column not found: {segment}")
            continue
        for value, group in frame.groupby(segment, dropna=False):
            row = _group_metric_row(group, task, threshold, warnings)
            row.update({"segment_column": segment, "segment_value": value})
            warning_flags = []
            if len(group) < min_segment_n:
                warning_flags.append("low_n")
            if task == "binary" and int(group["y_true"].sum()) < min_segment_events:
                warning_flags.append("low_events")
            row["warning_flag"] = ",".join(warning_flags)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _feature_split_metrics(
    frame: pd.DataFrame,
    task: str,
    feature_splits: Sequence[str],
    n_bins: int,
    binning_strategy: BinningStrategy,
    threshold: float,
    warnings: list[str],
) -> dict[str, pd.DataFrame]:
    results = {}
    for feature in feature_splits:
        if feature not in frame.columns:
            warnings.append(f"Feature split column not found: {feature}")
            continue
        source = frame.copy()
        is_numeric_feature = bool(pd.api.types.is_numeric_dtype(source[feature]))
        if is_numeric_feature and source[feature].nunique() > n_bins:
            source["feature_bin"] = _numeric_bins(source[feature], n_bins, binning_strategy)
            table = (
                source.groupby("feature_bin", observed=True, dropna=False)
                .apply(
                    lambda group, feature_name=feature: _feature_split_row(
                        group,
                        task,
                        threshold,
                        warnings,
                        feature_name,
                        is_numeric_feature=True,
                    )
                )
                .reset_index()
                .rename(columns={"feature_bin": "bin"})
            )
        else:
            table = (
                source.groupby(feature, observed=True, dropna=False)
                .apply(
                    lambda group, feature_name=feature: _feature_split_row(
                        group,
                        task,
                        threshold,
                        warnings,
                        feature_name,
                        is_numeric_feature=False,
                    )
                )
                .reset_index()
                .rename(columns={feature: "category"})
            )
        table.insert(0, "feature", feature)
        results[feature] = table
    return results


def _feature_split_row(
    group: pd.DataFrame,
    task: str,
    threshold: float,
    warnings: list[str],
    feature: str,
    *,
    is_numeric_feature: bool,
) -> pd.Series:
    row = _group_metric_row(group, task, threshold, warnings)
    if is_numeric_feature and feature in group.columns:
        row["bin_lower"] = float(group[feature].min())
        row["bin_upper"] = float(group[feature].max())
        row["mean_feature_value"] = float(group[feature].mean())
    return pd.Series(row)


def _stability_metrics(
    frame: pd.DataFrame,
    task: str,
    threshold: float,
    overall_metrics: dict[str, MetricValue],
    warnings: list[str],
) -> pd.DataFrame:
    rows = []
    primary_metric = _primary_metric(task)
    overall_value = overall_metrics.get(primary_metric)
    for split_value, group in frame.groupby("split_col", dropna=False):
        row = _group_metric_row(group, task, threshold, warnings)
        row["split_value"] = split_value
        metric_value = row.get(primary_metric)
        row["primary_metric"] = primary_metric
        row["metric_delta_vs_overall"] = (
            None
            if metric_value is None or overall_value is None
            else float(metric_value) - float(overall_value)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _group_metric_row(
    group: pd.DataFrame,
    task: str,
    threshold: float,
    warnings: list[str],
) -> dict[str, Any]:
    weight = group["sample_weight"] if "sample_weight" in group.columns else None
    metric_warnings: list[str] = []
    metrics = compute_task_metrics(
        group["y_true"],
        group["y_pred"],
        task,
        weight,
        threshold,
        metric_warnings,
    )
    for warning in metric_warnings:
        warnings.append(f"{warning} in grouped evaluation")
    row: dict[str, Any] = {
        "n": int(len(group)),
        "weighted_n": _weighted_n(group),
        "mean_actual": _weighted_mean(group["y_true"], weight),
        "mean_prediction": _weighted_mean(group["y_pred"], weight),
        "residual": _weighted_mean(group["y_true"], weight)
        - _weighted_mean(group["y_pred"], weight),
        "observed_expected_ratio": _oe_ratio(group["y_true"], group["y_pred"], weight),
    }
    row["abs_residual"] = abs(float(row["residual"]))
    row.update(metrics)
    if task == "binary":
        row["event_rate"] = row["mean_actual"]
        row["mean_predicted_probability"] = row["mean_prediction"]
    return row


def _numeric_bins(
    values: pd.Series,
    n_bins: int,
    strategy: BinningStrategy,
) -> pd.Series:
    if values.nunique(dropna=True) < 2:
        return pd.Series(["all"] * len(values), index=values.index, name="bin")
    if strategy == "uniform":
        return pd.cut(values, bins=min(n_bins, values.nunique()), duplicates="drop")
    return pd.qcut(values, q=min(n_bins, values.nunique()), duplicates="drop")


def _weighted_mean(values: pd.Series, weight: pd.Series | None) -> float:
    if weight is None:
        return float(values.mean())
    if np.isclose(float(weight.sum()), 0.0):
        return float(values.mean())
    return float(np.average(values, weights=weight))


def _weighted_sum(values: pd.Series, weight: pd.Series | None) -> float:
    if weight is None:
        return float(values.sum())
    return float(np.sum(values * weight))


def _weighted_n(group: pd.DataFrame) -> float:
    if "sample_weight" not in group.columns:
        return float(len(group))
    return float(group["sample_weight"].sum())


def _oe_ratio(
    y_true: pd.Series,
    y_pred: pd.Series,
    weight: pd.Series | None,
) -> MetricValue:
    return _ratio(_weighted_sum(y_true, weight), _weighted_sum(y_pred, weight))


def _ratio(numerator: float, denominator: float) -> MetricValue:
    if np.isclose(denominator, 0.0):
        return None
    return float(numerator / denominator)


def _coerce_numeric_series(values: SeriesLike, name: str) -> pd.Series:
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values, name=name)
    try:
        numeric = pd.to_numeric(series)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    numeric.name = name
    return numeric.reset_index(drop=True)


def _normalize_task(task: Task) -> str:
    if task not in {"regression", "binary", "count", "rate"}:
        raise ValueError("task must be one of: regression, binary, count, rate")
    return task


def _primary_metric(task: str) -> str:
    if task == "binary":
        return "gini"
    if task in {"count", "rate"}:
        return "observed_expected_ratio"
    return "rmse"
