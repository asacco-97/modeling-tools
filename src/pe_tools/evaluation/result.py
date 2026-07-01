from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


@dataclass
class EvaluationResult:
    """Structured model evaluation output."""

    overall_metrics: dict[str, float | None]
    calibration_table: pd.DataFrame | None
    lift_table: pd.DataFrame | None
    segment_metrics: pd.DataFrame | None
    feature_split_metrics: dict[str, pd.DataFrame]
    stability_metrics: pd.DataFrame | None
    figures: dict[str, Figure] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for notebook display."""
        return {
            "overall_metrics": self.overall_metrics,
            "n_calibration_bins": 0
            if self.calibration_table is None
            else len(self.calibration_table),
            "n_lift_bins": 0 if self.lift_table is None else len(self.lift_table),
            "n_segments": 0 if self.segment_metrics is None else len(self.segment_metrics),
            "n_feature_splits": len(self.feature_split_metrics),
            "n_stability_splits": 0
            if self.stability_metrics is None
            else len(self.stability_metrics),
            "warnings": self.warnings,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result tables to serializable Python objects."""
        return {
            "overall_metrics": self.overall_metrics,
            "calibration_table": _frame_records(self.calibration_table),
            "lift_table": _frame_records(self.lift_table),
            "segment_metrics": _frame_records(self.segment_metrics),
            "feature_split_metrics": {
                key: _frame_records(frame) for key, frame in self.feature_split_metrics.items()
            },
            "stability_metrics": _frame_records(self.stability_metrics),
            "warnings": list(self.warnings),
        }

    def to_frame(self) -> pd.DataFrame:
        """Return overall metrics as a tidy DataFrame."""
        return pd.DataFrame(
            {
                "metric": list(self.overall_metrics),
                "value": list(self.overall_metrics.values()),
            }
        )

    def plot_calibration(self) -> Figure:
        """Plot observed versus predicted values by calibration bin."""
        if self.calibration_table is None or self.calibration_table.empty:
            raise ValueError("calibration_table is empty")
        table = self.calibration_table
        if "mean_predicted_probability" in table.columns:
            x_col = "mean_predicted_probability"
            y_col = "observed_rate"
            x_label = "Mean predicted probability"
            y_label = "Observed rate"
        else:
            x_col = "mean_prediction"
            y_col = "mean_actual"
            x_label = "Mean prediction"
            y_label = "Mean actual"

        figure = Figure(figsize=(6.5, 5.0))
        axes = figure.subplots()
        sizes = 25 + 150 * (table["n"] / max(float(table["n"].max()), 1.0))
        axes.scatter(table[x_col], table[y_col], s=sizes, alpha=0.8)
        lower = float(min(table[x_col].min(), table[y_col].min()))
        upper = float(max(table[x_col].max(), table[y_col].max()))
        axes.plot([lower, upper], [lower, upper], linestyle=":", color="black")
        axes.set_title("Calibration")
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        self.figures["calibration"] = figure
        return figure

    def plot_lift(self) -> Figure:
        """Plot lift or observed outcome by prediction quantile."""
        if self.lift_table is None or self.lift_table.empty:
            raise ValueError("lift_table is empty")
        table = self.lift_table.sort_values("quantile_bin")
        y_col = "event_rate" if "event_rate" in table.columns else "lift_vs_overall"
        y_label = "Event rate" if y_col == "event_rate" else "Lift vs overall"

        figure = Figure(figsize=(7.0, 4.5))
        axes = figure.subplots()
        x_values = np.arange(len(table))
        axes.plot(x_values, table[y_col], marker="o")
        axes.set_title("Lift by Prediction Quantile")
        axes.set_xlabel("Prediction quantile")
        axes.set_ylabel(y_label)
        axes.set_xticks(x_values)
        axes.set_xticklabels(table["quantile_bin"].astype(str).tolist(), rotation=45, ha="right")
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        self.figures["lift"] = figure
        return figure

    def plot_cumulative_gain(self) -> Figure:
        """Plot cumulative actual capture versus population share."""
        if self.lift_table is None or self.lift_table.empty:
            raise ValueError("lift_table is empty")
        table = self.lift_table.sort_values("quantile_bin")
        y_col = (
            "cumulative_event_capture_rate"
            if "cumulative_event_capture_rate" in table.columns
            else "cumulative_capture_rate"
        )

        figure = Figure(figsize=(6.5, 5.0))
        axes = figure.subplots()
        axes.plot(table["cumulative_population_share"], table[y_col], marker="o")
        axes.plot([0.0, 1.0], [0.0, 1.0], linestyle=":", color="black")
        axes.set_title("Cumulative Gain")
        axes.set_xlabel("Cumulative population share")
        axes.set_ylabel("Cumulative actual capture share")
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        self.figures["cumulative_gain"] = figure
        return figure

    def plot_segment_performance(self, metric: str = "observed_expected_ratio") -> Figure:
        """Plot a segment-level metric."""
        if self.segment_metrics is None or self.segment_metrics.empty:
            raise ValueError("segment_metrics is empty")
        if metric not in self.segment_metrics.columns:
            raise ValueError(f"Unknown segment metric: {metric}")
        table = self.segment_metrics.copy()
        table["label"] = table["segment_column"].astype(str) + "=" + table[
            "segment_value"
        ].astype(str)

        figure = Figure(figsize=(8.0, 4.8))
        axes = figure.subplots()
        axes.bar(table["label"], table[metric])
        axes.set_title(f"Segment Performance: {metric}")
        axes.set_xlabel("Segment")
        axes.set_ylabel(metric)
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        self.figures[f"segment_{metric}"] = figure
        return figure

    def plot_feature_split(self, feature: str) -> Figure:
        """Plot actual and predicted values across one feature split."""
        if feature not in self.feature_split_metrics:
            raise ValueError(f"Unknown feature split: {feature}")
        table = self.feature_split_metrics[feature]
        x_values = np.arange(len(table))
        if "category" in table.columns:
            labels = table["category"].astype(str).tolist()
        else:
            labels = table["bin"].astype(str).tolist()

        figure = Figure(figsize=(8.0, 4.8))
        axes = figure.subplots()
        axes.plot(x_values, table["mean_actual"], marker="o", label="Mean actual")
        axes.plot(x_values, table["mean_prediction"], marker="o", label="Mean prediction")
        axes.set_title(f"Feature Split: {feature}")
        axes.set_xlabel(feature)
        axes.set_ylabel("Mean")
        axes.set_xticks(x_values)
        axes.set_xticklabels(labels, rotation=45, ha="right")
        axes.legend()
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        self.figures[f"feature_split_{feature}"] = figure
        return figure

    def plot_top_feature_errors(self, top_n: int = 10) -> Figure:
        """Rank feature splits by average absolute systematic error."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        rows = []
        for feature, table in self.feature_split_metrics.items():
            if table.empty:
                continue
            rows.append(
                {
                    "feature": feature,
                    "mean_abs_error": float(table["abs_residual"].mean()),
                }
            )
        if not rows:
            raise ValueError("feature_split_metrics is empty")
        ranking = pd.DataFrame(rows).sort_values("mean_abs_error", ascending=False).head(top_n)

        figure = Figure(figsize=(7.0, 4.5))
        axes = figure.subplots()
        axes.bar(ranking["feature"], ranking["mean_abs_error"])
        axes.set_title("Top Feature Split Errors")
        axes.set_xlabel("Feature")
        axes.set_ylabel("Mean absolute residual")
        axes.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        self.figures["top_feature_errors"] = figure
        return figure

    def plot_stability(self, metric: str) -> Figure:
        """Plot a metric across supplied evaluation splits."""
        if self.stability_metrics is None or self.stability_metrics.empty:
            raise ValueError("stability_metrics is empty")
        if metric not in self.stability_metrics.columns:
            raise ValueError(f"Unknown stability metric: {metric}")

        figure = Figure(figsize=(7.0, 4.5))
        axes = figure.subplots()
        axes.plot(
            self.stability_metrics["split_value"].astype(str),
            self.stability_metrics[metric],
            marker="o",
        )
        axes.set_title(f"Stability: {metric}")
        axes.set_xlabel("Split")
        axes.set_ylabel(metric)
        axes.tick_params(axis="x", rotation=45)
        axes.grid(True, alpha=0.25)
        figure.tight_layout()
        self.figures[f"stability_{metric}"] = figure
        return figure

    def plot_all(self) -> dict[str, Figure]:
        """Generate all standard plots that have backing tables."""
        figures: dict[str, Figure] = {}
        if self.calibration_table is not None and not self.calibration_table.empty:
            figures["calibration"] = self.plot_calibration()
        if self.lift_table is not None and not self.lift_table.empty:
            figures["lift"] = self.plot_lift()
            figures["cumulative_gain"] = self.plot_cumulative_gain()
        if self.segment_metrics is not None and not self.segment_metrics.empty:
            figures["segment_performance"] = self.plot_segment_performance()
        if self.stability_metrics is not None and not self.stability_metrics.empty:
            metric = (
                "gini"
                if "gini" in self.stability_metrics.columns
                else "observed_expected_ratio"
            )
            figures["stability"] = self.plot_stability(metric)
        if self.feature_split_metrics:
            first_feature = next(iter(self.feature_split_metrics))
            figures[f"feature_split_{first_feature}"] = self.plot_feature_split(first_feature)
            figures["top_feature_errors"] = self.plot_top_feature_errors()
        return figures


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]] | None:
    if frame is None:
        return None
    return frame.to_dict(orient="records")
