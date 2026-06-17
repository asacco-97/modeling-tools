from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from xgboost import XGBRegressor

ModelType = Literal["xgboost", "random_forest"]
SplitStrategy = Literal["cv", "holdout", "group_cv", "bootstrap"]
VALID_MODEL_TYPES = {"xgboost", "random_forest"}
VALID_SPLIT_STRATEGIES = {"cv", "holdout", "group_cv", "bootstrap"}
INTERACTION_IMPORTANCE_COLUMNS = ["feature_1", "feature_2", "importance", "rank"]
SeriesLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[float]
LabelLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[Any]


@dataclass(frozen=True)
class ResidualSignalResult:
    """Residualized target/signal pair and their correlation."""

    target_residual: pd.Series
    signal_residual: pd.Series
    correlation: float
    n_obs: int
    interaction_importance: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=INTERACTION_IMPORTANCE_COLUMNS)
    )


@dataclass(frozen=True)
class ResidualSignalFinderResult:
    """Result returned by ResidualSignalFinder.fit."""

    residuals: pd.Series
    feature_importance: pd.DataFrame
    feature_stability: pd.DataFrame
    binned_diagnostics: dict[str, pd.DataFrame]
    residual_model_score: dict[str, float]
    fold_scores: pd.DataFrame
    fold_feature_importance: pd.DataFrame
    metadata: dict[str, Any]
    models: list[Any]
    interaction_importance: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=INTERACTION_IMPORTANCE_COLUMNS)
    )

    def to_excel(self, path: str | Path) -> Path:
        """Write the result report to an Excel workbook."""
        output_path = _validate_report_path(path)
        used_sheet_names: set[str] = set()

        def write_sheet(frame: pd.DataFrame, sheet_name: str, writer: pd.ExcelWriter) -> None:
            safe_name = _sanitize_excel_sheet_name(sheet_name, used_sheet_names)
            _safe_table_for_export(frame).to_excel(writer, sheet_name=safe_name, index=False)

        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                write_sheet(_summary_frame(self), "summary", writer)
                write_sheet(self.feature_importance, "feature_importance", writer)
                write_sheet(self.interaction_importance, "interaction_importance", writer)
                write_sheet(self.feature_stability, "feature_stability", writer)
                write_sheet(self.fold_scores, "fold_scores", writer)
                write_sheet(self.fold_feature_importance, "fold_feature_importance", writer)
                for feature in _top_feature_names(self, top_n=None):
                    table = self.binned_diagnostics.get(feature)
                    if table is not None:
                        write_sheet(table, f"binned_{feature}", writer)
                write_sheet(_dict_frame(self.metadata), "metadata", writer)
        except OSError as error:
            raise ValueError(f"Could not write Excel report to {output_path}: {error}") from error

        return output_path

    def to_html(self, path: str | Path, top_n: int = 10) -> Path:
        """Write a simple HTML report."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        output_path = _validate_report_path(path)
        top_features = _top_feature_names(self, top_n=top_n)
        diagnostics_sections = []
        for feature in top_features:
            table = self.binned_diagnostics.get(feature)
            if table is not None:
                diagnostics_sections.append(
                    f"<h3>{html.escape(feature)}</h3>\n{_html_table(table)}"
                )

        fold_distribution = self.fold_scores.describe(include="all").reset_index()
        fold_distribution = fold_distribution.rename(columns={"index": "metric"})
        content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Residual Signal Finder Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Residual Signal Finder Report</h1>
  <h2>Summary</h2>
  {_html_table(_summary_frame(self))}
  <h2>Residual Model Score</h2>
  {_html_table(_dict_frame(self.residual_model_score))}
  <h2>Fold Score Distribution</h2>
  {_html_table(fold_distribution)}
  <h2>Feature Stability</h2>
  {_html_table(self.feature_stability)}
  <h2>Top Feature Binned Diagnostics</h2>
  {''.join(diagnostics_sections)}
  <h2>Warnings / Limitations</h2>
  <p>This minimal v1 report does not include SHAP, plotting, Excel styling,
  interaction logic, categorical feature handling, or missing-value handling.</p>
</body>
</html>
"""
        try:
            output_path.write_text(content, encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Could not write HTML report to {output_path}: {error}") from error

        return output_path

    def plot_top_residual_signals(
        self,
        top_n: int = 10,
        save_dir: str | Path | None = None,
    ) -> dict[str, Figure]:
        """Plot binned actual, predicted, and residual diagnostics for top features."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        output_dir = _validate_plot_dir(save_dir) if save_dir is not None else None
        figures: dict[str, Figure] = {}
        for feature in _top_feature_names(self, top_n=top_n):
            diagnostics = self.binned_diagnostics.get(feature)
            if diagnostics is None or diagnostics.empty:
                continue

            figure = _plot_binned_diagnostics(feature, diagnostics)
            figures[feature] = figure
            if output_dir is not None:
                figure.savefig(
                    output_dir / f"{_safe_filename_stem(feature)}.png",
                    bbox_inches="tight",
                )

        return figures

    def plot_top_signals(
        self,
        top_n: int = 10,
        save_dir: str | Path | None = None,
    ) -> dict[str, Figure]:
        """Alias for plot_top_residual_signals."""
        return self.plot_top_residual_signals(top_n=top_n, save_dir=save_dir)


@dataclass
class ResidualSignalFinder:
    """Public API skeleton for residual signal discovery."""

    model_type: ModelType = "xgboost"
    max_depth: int = 1
    n_estimators: int = 100
    learning_rate: float = 0.05
    n_bins: int = 10
    split_strategy: SplitStrategy = "cv"
    n_splits: int = 5
    n_repeats: int = 1
    sample_fraction: float = 0.8
    stratify_col: str | None = None
    group_col: str | None = None
    random_state: int | None = None
    model_params: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError("model_type must be one of: xgboost, random_forest")
        if self.split_strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError("split_strategy must be one of: cv, holdout, group_cv, bootstrap")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be greater than 0 and no more than 1")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")

    def fit(
        self,
        *,
        X: pd.DataFrame,
        y_true: SeriesLike,
        y_pred: SeriesLike,
        sample_weight: SeriesLike | None = None,
        split_col: LabelLike | None = None,
    ) -> ResidualSignalFinderResult:
        if self.split_strategy == "holdout" and split_col is None:
            raise ValueError("split_col is required when split_strategy='holdout'")
        if self.split_strategy == "group_cv" and self.group_col is None:
            raise ValueError("group_col is required when split_strategy='group_cv'")

        X_valid, y_true_valid, y_pred_valid, weight_values = self._validate_fit_inputs(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            split_col=split_col,
        )
        residuals = pd.Series(y_true_valid - y_pred_valid, index=X_valid.index, name="residual")
        split_series = (
            _coerce_series(split_col, name="split_col", index=X_valid.index)
            if split_col is not None
            else None
        )

        if self.split_strategy == "holdout":
            if split_series is None:
                raise ValueError("split_col is required when split_strategy='holdout'")
            return self._fit_holdout(
                X=X_valid,
                residuals=residuals,
                sample_weight=weight_values,
                split_col=split_series,
                sample_weight_used=sample_weight is not None,
            )

        model_X, groups = self._model_frame_and_groups(X_valid)
        if self.split_strategy == "bootstrap":
            return self._fit_bootstrap(
                X=model_X,
                residuals=residuals,
                sample_weight=weight_values,
                sample_weight_used=sample_weight is not None,
            )

        feature_names = list(model_X.columns)
        fold_scores: list[dict[str, Any]] = []
        fold_importance_frames: list[pd.DataFrame] = []
        fold_interaction_frames: list[pd.DataFrame] = []
        models: list[Any] = []
        oof_predictions = pd.Series(np.nan, index=X_valid.index, name="predicted_residual")
        shap_module, interaction_warnings = _load_shap_for_interactions(self.max_depth)

        for fold, (train_idx, test_idx) in enumerate(self._cv_splits(model_X, groups)):
            X_train = model_X.iloc[train_idx]
            X_test = model_X.iloc[test_idx]
            y_train = residuals.iloc[train_idx]
            y_test = residuals.iloc[test_idx]
            train_weight = weight_values[train_idx] if weight_values is not None else None
            test_weight = weight_values[test_idx] if weight_values is not None else None

            model = self._make_model()
            fit_kwargs = {"sample_weight": train_weight} if train_weight is not None else {}
            model.fit(X_train, y_train, **fit_kwargs)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            oof_predictions.iloc[test_idx] = test_pred

            train_r2 = float(r2_score(y_train, train_pred, sample_weight=train_weight))
            test_r2 = float(r2_score(y_test, test_pred, sample_weight=test_weight))
            fold_scores.append(
                {
                    "fold": fold,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "validation_score": test_r2,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "test_groups": (
                        _unique_sorted_tuple(groups.iloc[test_idx]) if groups is not None else ()
                    ),
                }
            )
            fold_importance_frames.append(self._feature_importance(model, feature_names, fold))
            interaction_frame = _fold_interaction_importance(
                shap_module=shap_module,
                model=model,
                X_eval=X_test,
                feature_names=feature_names,
                fold=fold,
                random_state=self.random_state,
            )
            if not interaction_frame.empty:
                fold_interaction_frames.append(interaction_frame)
            models.append(model)

        fold_scores_frame = pd.DataFrame(fold_scores)
        fold_feature_importance = pd.concat(fold_importance_frames, ignore_index=True)
        feature_importance = self._aggregate_feature_importance(fold_feature_importance)
        feature_stability = self._feature_stability(fold_feature_importance)
        interaction_importance = _aggregate_interaction_importance(fold_interaction_frames)
        binned_diagnostics = self._binned_diagnostics(model_X, residuals, oof_predictions)
        residual_model_score = {
            "mean_train_r2": float(fold_scores_frame["train_r2"].mean()),
            "mean_test_r2": float(fold_scores_frame["test_r2"].mean()),
            "std_test_r2": float(fold_scores_frame["test_r2"].std(ddof=0)),
        }
        return ResidualSignalFinderResult(
            residuals=residuals,
            feature_importance=feature_importance,
            feature_stability=feature_stability,
            binned_diagnostics=binned_diagnostics,
            residual_model_score=residual_model_score,
            fold_scores=fold_scores_frame,
            fold_feature_importance=fold_feature_importance,
            metadata=self._metadata(sample_weight is not None, warnings=interaction_warnings),
            models=models,
            interaction_importance=interaction_importance,
        )

    def _fit_holdout(
        self,
        *,
        X: pd.DataFrame,
        residuals: pd.Series,
        sample_weight: np.ndarray[Any, Any] | None,
        split_col: pd.Series,
        sample_weight_used: bool,
    ) -> ResidualSignalFinderResult:
        allowed_labels = {"train", "validation", "holdout"}
        split_labels = split_col.astype(str)
        labels = set(split_labels)
        invalid_labels = sorted(labels - allowed_labels)
        if invalid_labels:
            raise ValueError(
                "split_col labels must be one of: train, validation, holdout; "
                f"invalid labels: {invalid_labels}"
            )
        train_mask = split_labels == "train"
        if not train_mask.any():
            raise ValueError("split_col must contain at least one train label")
        score_labels = [
            label for label in ("validation", "holdout") if (split_labels == label).any()
        ]
        if not score_labels:
            raise ValueError("split_col must contain validation or holdout labels")

        feature_names = list(X.columns)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        train_weight = sample_weight[train_idx] if sample_weight is not None else None
        model = self._make_model()
        fit_kwargs = {"sample_weight": train_weight} if train_weight is not None else {}
        model.fit(X.iloc[train_idx], residuals.iloc[train_idx], **fit_kwargs)
        train_pred = model.predict(X.iloc[train_idx])
        train_r2 = float(
            r2_score(residuals.iloc[train_idx], train_pred, sample_weight=train_weight)
        )

        fold_scores: list[dict[str, Any]] = []
        predictions = pd.Series(np.nan, index=X.index, name="predicted_residual")
        for fold, label in enumerate(score_labels):
            test_mask = split_labels == label
            test_idx = np.flatnonzero(test_mask.to_numpy())
            test_weight = sample_weight[test_idx] if sample_weight is not None else None
            test_pred = model.predict(X.iloc[test_idx])
            predictions.iloc[test_idx] = test_pred
            test_r2 = float(
                r2_score(residuals.iloc[test_idx], test_pred, sample_weight=test_weight)
            )
            fold_scores.append(
                {
                    "fold": fold,
                    "split": label,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "validation_score": test_r2,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )

        fold_scores_frame = pd.DataFrame(fold_scores)
        fold_feature_importance = self._feature_importance(model, feature_names, 0)
        feature_importance = self._aggregate_feature_importance(fold_feature_importance)
        feature_stability = self._feature_stability(fold_feature_importance)
        shap_module, interaction_warnings = _load_shap_for_interactions(self.max_depth)
        interaction_importance = _aggregate_interaction_importance(
            [
                _fold_interaction_importance(
                    shap_module=shap_module,
                    model=model,
                    X_eval=X.loc[split_labels.isin(["validation", "holdout"])],
                    feature_names=feature_names,
                    fold=0,
                    random_state=self.random_state,
                )
            ]
        )
        eval_mask = split_labels.isin(["validation", "holdout"])
        binned_diagnostics = self._binned_diagnostics(
            X.loc[eval_mask],
            residuals.loc[eval_mask],
            predictions.loc[eval_mask],
        )
        residual_model_score = {
            "train_r2": train_r2,
            "mean_test_r2": float(fold_scores_frame["test_r2"].mean()),
            "std_test_r2": float(fold_scores_frame["test_r2"].std(ddof=0)),
        }
        for row in fold_scores:
            residual_model_score[f"{row['split']}_r2"] = float(row["test_r2"])

        return ResidualSignalFinderResult(
            residuals=residuals,
            feature_importance=feature_importance,
            feature_stability=feature_stability,
            binned_diagnostics=binned_diagnostics,
            residual_model_score=residual_model_score,
            fold_scores=fold_scores_frame,
            fold_feature_importance=fold_feature_importance,
            metadata=self._metadata(sample_weight_used, warnings=interaction_warnings),
            models=[model],
            interaction_importance=interaction_importance,
        )

    def _fit_bootstrap(
        self,
        *,
        X: pd.DataFrame,
        residuals: pd.Series,
        sample_weight: np.ndarray[Any, Any] | None,
        sample_weight_used: bool,
    ) -> ResidualSignalFinderResult:
        rng = np.random.default_rng(self.random_state)
        row_count = len(X)
        train_size = max(1, int(round(row_count * self.sample_fraction)))
        feature_names = list(X.columns)
        fold_scores: list[dict[str, Any]] = []
        fold_importance_frames: list[pd.DataFrame] = []
        fold_interaction_frames: list[pd.DataFrame] = []
        models: list[Any] = []
        prediction_sum = pd.Series(0.0, index=X.index, name="predicted_residual")
        prediction_count = pd.Series(0, index=X.index, name="prediction_count")
        shap_module, interaction_warnings = _load_shap_for_interactions(self.max_depth)

        for repeat in range(self.n_repeats):
            train_idx = rng.choice(row_count, size=train_size, replace=True)
            oob_mask = np.ones(row_count, dtype=bool)
            oob_mask[np.unique(train_idx)] = False
            test_idx = np.flatnonzero(oob_mask)
            if len(test_idx) == 0:
                continue

            train_weight = sample_weight[train_idx] if sample_weight is not None else None
            test_weight = sample_weight[test_idx] if sample_weight is not None else None
            model = self._make_model()
            fit_kwargs = {"sample_weight": train_weight} if train_weight is not None else {}
            model.fit(X.iloc[train_idx], residuals.iloc[train_idx], **fit_kwargs)
            train_pred = model.predict(X.iloc[train_idx])
            test_pred = model.predict(X.iloc[test_idx])

            train_r2 = float(
                r2_score(residuals.iloc[train_idx], train_pred, sample_weight=train_weight)
            )
            test_r2 = float(
                r2_score(residuals.iloc[test_idx], test_pred, sample_weight=test_weight)
            )
            fold_scores.append(
                {
                    "fold": repeat,
                    "repeat": repeat,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "validation_score": test_r2,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )
            fold_importance_frames.append(self._feature_importance(model, feature_names, repeat))
            interaction_frame = _fold_interaction_importance(
                shap_module=shap_module,
                model=model,
                X_eval=X.iloc[test_idx],
                feature_names=feature_names,
                fold=repeat,
                random_state=self.random_state,
            )
            if not interaction_frame.empty:
                fold_interaction_frames.append(interaction_frame)
            models.append(model)
            prediction_sum.iloc[test_idx] = prediction_sum.iloc[test_idx] + test_pred
            prediction_count.iloc[test_idx] = prediction_count.iloc[test_idx] + 1

        if not fold_scores:
            raise ValueError("No bootstrap repeats produced out-of-bag rows")

        fold_scores_frame = pd.DataFrame(fold_scores)
        fold_feature_importance = pd.concat(fold_importance_frames, ignore_index=True)
        feature_importance = self._aggregate_feature_importance(fold_feature_importance)
        feature_stability = self._feature_stability(fold_feature_importance)
        interaction_importance = _aggregate_interaction_importance(fold_interaction_frames)
        predictions = prediction_sum / prediction_count.replace(0, np.nan)
        binned_diagnostics = self._binned_diagnostics(X, residuals, predictions)
        residual_model_score = {
            "mean_train_r2": float(fold_scores_frame["train_r2"].mean()),
            "mean_test_r2": float(fold_scores_frame["test_r2"].mean()),
            "std_test_r2": float(fold_scores_frame["test_r2"].std(ddof=0)),
        }

        return ResidualSignalFinderResult(
            residuals=residuals,
            feature_importance=feature_importance,
            feature_stability=feature_stability,
            binned_diagnostics=binned_diagnostics,
            residual_model_score=residual_model_score,
            fold_scores=fold_scores_frame,
            fold_feature_importance=fold_feature_importance,
            metadata=self._metadata(sample_weight_used, warnings=interaction_warnings),
            models=models,
            interaction_importance=interaction_importance,
        )

    def _metadata(
        self,
        sample_weight_used: bool,
        warnings: Sequence[str] = (),
    ) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "n_bins": self.n_bins,
            "split_strategy": self.split_strategy,
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "sample_fraction": self.sample_fraction,
            "stratify_col": self.stratify_col,
            "group_col": self.group_col,
            "random_state": self.random_state,
            "sample_weight_used": sample_weight_used,
            "warnings": list(warnings),
        }

    def _validate_fit_inputs(
        self,
        *,
        X: pd.DataFrame,
        y_true: SeriesLike,
        y_pred: SeriesLike,
        sample_weight: SeriesLike | None,
        split_col: LabelLike | None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray[Any, Any] | None]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if X.columns.duplicated().any():
            duplicated = sorted(set(X.columns[X.columns.duplicated()].astype(str)))
            raise ValueError(f"X column names must be unique; duplicated columns: {duplicated}")

        y_true_valid = _coerce_numeric_series(y_true, name="y_true", index=X.index)
        y_pred_valid = _coerce_numeric_series(y_pred, name="y_pred", index=X.index)
        if len(X) != len(y_true_valid) or len(X) != len(y_pred_valid):
            raise ValueError("X, y_true, and y_pred must have the same length")
        if split_col is not None:
            _coerce_series(split_col, name="split_col", index=X.index)

        non_numeric = [
            str(column)
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]
        if non_numeric:
            raise ValueError(
                f"categorical columns are not supported in v1; non-numeric columns: {non_numeric}"
            )
        if X.isna().any().any() or y_true_valid.isna().any() or y_pred_valid.isna().any():
            raise ValueError("missing values are not supported in v1")

        weight_values = None
        if sample_weight is not None:
            weight_series = _coerce_numeric_series(
                sample_weight,
                name="sample_weight",
                index=X.index,
            )
            if weight_series.isna().any():
                raise ValueError("missing values are not supported in v1")
            if (weight_series < 0).any():
                raise ValueError("sample_weight cannot contain negative values")
            if np.isclose(float(weight_series.sum()), 0.0):
                raise ValueError("sample_weight cannot be all zero")
            weight_values = weight_series.to_numpy()

        return X.copy(), y_true_valid, y_pred_valid, weight_values

    def _model_frame_and_groups(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        if self.split_strategy != "group_cv":
            return X, None
        if self.group_col is None:
            raise ValueError("group_col is required when split_strategy='group_cv'")
        if self.group_col not in X.columns:
            raise ValueError(f"group_col '{self.group_col}' must be a column in X")
        groups = X[self.group_col].copy()
        if groups.nunique() < self.n_splits:
            raise ValueError("group_col must contain at least n_splits unique groups")
        return X.drop(columns=[self.group_col]), groups

    def _cv_splits(self, X: pd.DataFrame, groups: pd.Series | None) -> Any:
        if self.split_strategy == "group_cv":
            if groups is None:
                raise ValueError("group_col is required when split_strategy='group_cv'")
            splitter = GroupKFold(n_splits=self.n_splits)
            return splitter.split(X, groups=groups)

        if self.stratify_col is None:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return splitter.split(X)
        if self.stratify_col not in X.columns:
            raise ValueError(f"stratify_col '{self.stratify_col}' must be a column in X")
        strata = _make_strata(X[self.stratify_col], self.n_bins)
        splitter = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        return splitter.split(X, strata)

    def _make_model(self) -> Any:
        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
        if self.model_type == "xgboost":
            params.update(
                {
                    "learning_rate": self.learning_rate,
                    "objective": "reg:squarederror",
                    "n_jobs": 1,
                    "verbosity": 0,
                }
            )
            params.update(self.model_params or {})
            return XGBRegressor(**params)

        params.update({"n_jobs": 1})
        params.update(self.model_params or {})
        return RandomForestRegressor(**params)

    def _feature_importance(
        self,
        model: Any,
        feature_names: list[str],
        fold: int,
    ) -> pd.DataFrame:
        if self.model_type == "xgboost":
            raw_scores = model.get_booster().get_score(importance_type="gain")
            importance = _map_xgboost_importance(raw_scores, feature_names)
        else:
            importance = dict(zip(feature_names, model.feature_importances_, strict=True))

        frame = pd.DataFrame(
            {
                "fold": fold,
                "feature": feature_names,
                "importance": [float(importance.get(feature, 0.0)) for feature in feature_names],
                "importance_type": "gain",
            }
        ).sort_values(["importance", "feature"], ascending=[False, True], ignore_index=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        return frame

    @staticmethod
    def _aggregate_feature_importance(fold_feature_importance: pd.DataFrame) -> pd.DataFrame:
        importance = (
            fold_feature_importance.groupby("feature", as_index=False)["importance"]
            .mean()
            .rename(columns={"importance": "importance"})
            .sort_values(["importance", "feature"], ascending=[False, True], ignore_index=True)
        )
        importance["importance_type"] = "gain"
        return importance.loc[:, ["feature", "importance", "importance_type"]]

    @staticmethod
    def _feature_stability(fold_feature_importance: pd.DataFrame) -> pd.DataFrame:
        grouped = fold_feature_importance.groupby("feature", as_index=False)
        stability = grouped.agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_rank=("rank", "mean"),
            rank_std=("rank", "std"),
            top_1_rate=("rank", lambda value: float((value <= 1).mean())),
            top_3_rate=("rank", lambda value: float((value <= 3).mean())),
            selected_rate=("importance", lambda value: float((value > 0).mean())),
        )
        stability["std_importance"] = stability["std_importance"].fillna(0.0)
        stability["rank_std"] = stability["rank_std"].fillna(0.0)
        return stability.sort_values(
            ["mean_importance", "feature"],
            ascending=[False, True],
            ignore_index=True,
        )

    def _binned_diagnostics(
        self,
        X: pd.DataFrame,
        residuals: pd.Series,
        predictions: pd.Series,
    ) -> dict[str, pd.DataFrame]:
        diagnostics = {}
        for feature in X.columns:
            source = pd.DataFrame(
                {
                    "feature_value": X[feature],
                    "residual": residuals,
                    "predicted_residual": predictions,
                }
            ).dropna()
            source["prediction_error"] = source["residual"] - source["predicted_residual"]
            bin_count = min(self.n_bins, source["feature_value"].nunique())
            source["bin"] = pd.qcut(source["feature_value"], q=bin_count, duplicates="drop")
            diagnostics[str(feature)] = (
                source.groupby("bin", observed=True)
                .agg(
                    n_obs=("residual", "size"),
                    feature_mean=("feature_value", "mean"),
                    residual_mean=("residual", "mean"),
                    predicted_residual_mean=("predicted_residual", "mean"),
                    prediction_error_mean=("prediction_error", "mean"),
                )
                .reset_index()
            )
        return diagnostics


def _coerce_series(values: LabelLike, *, name: str, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        if len(values) != len(index):
            raise ValueError(f"{name} must have the same length as X")
        if not values.index.equals(index):
            raise ValueError(f"{name} index must match X index")
        return values.copy()

    array: np.ndarray[Any, Any] = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) != len(index):
        raise ValueError(f"{name} must have the same length as X")
    return pd.Series(array, index=index, name=name)


def _coerce_numeric_series(values: SeriesLike, *, name: str, index: pd.Index) -> pd.Series:
    series = _coerce_series(values, name=name, index=index)
    try:
        numeric = pd.to_numeric(series)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    numeric.name = series.name or name
    return numeric


def _make_strata(values: pd.Series, n_bins: int) -> pd.Series:
    unique_count = values.nunique()
    if unique_count < 2:
        raise ValueError("stratify_col must contain at least two strata")
    if pd.api.types.is_numeric_dtype(values) and unique_count > n_bins:
        return pd.Series(
            pd.qcut(values, q=n_bins, labels=False, duplicates="drop"),
            index=values.index,
            name=values.name,
        )
    return values.astype(str)


def _map_xgboost_importance(
    raw_scores: dict[str, float],
    feature_names: list[str],
) -> dict[str, float]:
    importance = {feature: 0.0 for feature in feature_names}
    for raw_feature, score in raw_scores.items():
        if raw_feature in importance:
            importance[raw_feature] = float(score)
            continue
        if raw_feature.startswith("f") and raw_feature[1:].isdigit():
            feature_index = int(raw_feature[1:])
            if feature_index < len(feature_names):
                importance[feature_names[feature_index]] = float(score)
    return importance


def _load_shap_for_interactions(max_depth: int) -> tuple[Any | None, list[str]]:
    if max_depth != 2:
        return None, []
    try:
        import shap
    except ImportError:
        return None, ["SHAP is not installed; interaction_importance skipped."]
    return shap, []


def _empty_interaction_importance() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERACTION_IMPORTANCE_COLUMNS)


def _fold_interaction_importance(
    *,
    shap_module: Any | None,
    model: Any,
    X_eval: pd.DataFrame,
    feature_names: list[str],
    fold: int,
    random_state: int | None,
) -> pd.DataFrame:
    if shap_module is None or X_eval.empty or len(feature_names) < 2:
        return _empty_interaction_importance()

    sample_size = min(len(X_eval), 200)
    X_sample = X_eval.sample(
        n=sample_size,
        random_state=(random_state or 0) + fold,
    )
    explainer = shap_module.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_sample)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]

    values = np.asarray(interaction_values)
    if values.ndim == 4:
        values = values.mean(axis=-1)
    if values.ndim != 3:
        return _empty_interaction_importance()

    rows = []
    for first_index, feature_1 in enumerate(feature_names):
        for second_index in range(first_index + 1, len(feature_names)):
            feature_2 = feature_names[second_index]
            rows.append(
                {
                    "fold": fold,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "importance": float(np.abs(values[:, first_index, second_index]).mean()),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_interaction_importance(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable_frames = [frame for frame in fold_frames if not frame.empty]
    if not usable_frames:
        return _empty_interaction_importance()

    aggregated = (
        pd.concat(usable_frames, ignore_index=True)
        .groupby(["feature_1", "feature_2"], as_index=False)["importance"]
        .mean()
        .sort_values(["importance", "feature_1", "feature_2"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    aggregated["rank"] = np.arange(1, len(aggregated) + 1)
    return aggregated.loc[:, INTERACTION_IMPORTANCE_COLUMNS]


def _unique_sorted_tuple(values: pd.Series) -> tuple[Any, ...]:
    return tuple(sorted(values.dropna().unique().tolist()))


def _validate_report_path(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"Report path must be a file, got directory: {output_path}")
    if not output_path.parent.exists():
        raise ValueError(f"Parent directory does not exist for report path: {output_path.parent}")
    return output_path


def _validate_plot_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    if not output_dir.exists():
        raise ValueError(f"Plot output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise ValueError(f"Plot output path must be a directory: {output_dir}")
    return output_dir


def _sanitize_excel_sheet_name(sheet_name: str, used_names: set[str]) -> str:
    cleaned = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(sheet_name)).strip().strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned) or "sheet"
    base = cleaned[:31]
    candidate = base
    counter = 1
    while candidate.lower() in used_names:
        suffix = f"_{counter}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        counter += 1
    used_names.add(candidate.lower())
    return candidate


def _safe_filename_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (cleaned or "feature")[:80]


def _safe_table_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    safe_frame = frame.copy()
    for column in safe_frame.columns:
        if (
            pd.api.types.is_numeric_dtype(safe_frame[column])
            or pd.api.types.is_bool_dtype(safe_frame[column])
            or pd.api.types.is_datetime64_any_dtype(safe_frame[column])
            or pd.api.types.is_timedelta64_dtype(safe_frame[column])
        ):
            continue
        safe_frame[column] = safe_frame[column].astype(str)
    return safe_frame


def _dict_frame(values: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": list(values.keys()),
            "value": [_format_report_value(value) for value in values.values()],
        }
    )


def _summary_frame(result: ResidualSignalFinderResult) -> pd.DataFrame:
    summary = {
        "n_observations": len(result.residuals),
        "n_features": len(result.feature_importance),
        "n_folds": len(result.fold_scores),
        "n_models": len(result.models),
        **{f"score_{key}": value for key, value in result.residual_model_score.items()},
    }
    return _dict_frame(summary)


def _top_feature_names(result: ResidualSignalFinderResult, top_n: int | None) -> list[str]:
    if "feature" not in result.feature_importance.columns:
        features = list(result.binned_diagnostics)
    else:
        features = result.feature_importance["feature"].astype(str).tolist()
    if top_n is None:
        return [feature for feature in features if feature in result.binned_diagnostics]
    return [feature for feature in features[:top_n] if feature in result.binned_diagnostics]


def _format_report_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, int | bool | str) or value is None:
        return str(value)
    return json.dumps(value, default=str)


def _html_table(frame: pd.DataFrame) -> str:
    return _safe_table_for_export(frame).to_html(index=False, escape=True)


def _plot_binned_diagnostics(feature: str, diagnostics: pd.DataFrame) -> Figure:
    required_columns = {"residual_mean", "predicted_residual_mean", "prediction_error_mean"}
    missing_columns = sorted(required_columns - set(diagnostics.columns))
    if missing_columns:
        raise ValueError(f"binned diagnostics missing columns: {missing_columns}")

    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    x_values = np.arange(len(diagnostics))
    x_labels = (
        diagnostics["bin"].astype(str).tolist()
        if "bin" in diagnostics.columns
        else [str(value) for value in x_values]
    )
    series_specs = [
        ("residual_mean", "Mean actual"),
        ("predicted_residual_mean", "Mean predicted"),
        ("prediction_error_mean", "Mean residual"),
    ]
    for column, label in series_specs:
        axes.plot(
            x_values,
            pd.to_numeric(diagnostics[column]),
            marker="o",
            linewidth=1.6,
            label=label,
        )

    axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes.set_title(f"Residual Signal Diagnostics: {feature}")
    axes.set_xlabel("Feature bin")
    axes.set_ylabel("Mean value")
    axes.set_xticks(x_values)
    axes.set_xticklabels(x_labels, rotation=45, ha="right")
    axes.legend()
    axes.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


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
