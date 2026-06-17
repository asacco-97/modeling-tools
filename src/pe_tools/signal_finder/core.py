from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor

ModelType = Literal["xgboost", "random_forest"]
SplitStrategy = Literal["cv", "holdout", "group_cv", "bootstrap"]
VALID_MODEL_TYPES = {"xgboost", "random_forest"}
VALID_SPLIT_STRATEGIES = {"cv", "holdout", "group_cv", "bootstrap"}
SeriesLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[float]
LabelLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[Any]


@dataclass(frozen=True)
class ResidualSignalResult:
    """Residualized target/signal pair and their correlation."""

    target_residual: pd.Series
    signal_residual: pd.Series
    correlation: float
    n_obs: int


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
        if self.split_strategy != "cv":
            raise NotImplementedError("Only split_strategy='cv' is implemented.")

        X_valid, y_true_valid, y_pred_valid, weight_values = self._validate_fit_inputs(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            split_col=split_col,
        )
        residuals = pd.Series(y_true_valid - y_pred_valid, index=X_valid.index, name="residual")
        feature_names = list(X_valid.columns)
        fold_scores: list[dict[str, Any]] = []
        fold_importance_frames: list[pd.DataFrame] = []
        models: list[Any] = []
        oof_predictions = pd.Series(np.nan, index=X_valid.index, name="predicted_residual")

        for fold, (train_idx, test_idx) in enumerate(self._cv_splits(X_valid, residuals)):
            X_train = X_valid.iloc[train_idx]
            X_test = X_valid.iloc[test_idx]
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
                }
            )
            fold_importance_frames.append(self._feature_importance(model, feature_names, fold))
            models.append(model)

        fold_scores_frame = pd.DataFrame(fold_scores)
        fold_feature_importance = pd.concat(fold_importance_frames, ignore_index=True)
        feature_importance = self._aggregate_feature_importance(fold_feature_importance)
        feature_stability = self._feature_stability(fold_feature_importance)
        binned_diagnostics = self._binned_diagnostics(X_valid, residuals, oof_predictions)
        residual_model_score = {
            "mean_train_r2": float(fold_scores_frame["train_r2"].mean()),
            "mean_test_r2": float(fold_scores_frame["test_r2"].mean()),
            "std_test_r2": float(fold_scores_frame["test_r2"].std(ddof=0)),
        }
        metadata = {
            "model_type": self.model_type,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "n_bins": self.n_bins,
            "split_strategy": self.split_strategy,
            "n_splits": self.n_splits,
            "n_repeats": self.n_repeats,
            "stratify_col": self.stratify_col,
            "group_col": self.group_col,
            "random_state": self.random_state,
            "sample_weight_used": sample_weight is not None,
        }
        return ResidualSignalFinderResult(
            residuals=residuals,
            feature_importance=feature_importance,
            feature_stability=feature_stability,
            binned_diagnostics=binned_diagnostics,
            residual_model_score=residual_model_score,
            fold_scores=fold_scores_frame,
            fold_feature_importance=fold_feature_importance,
            metadata=metadata,
            models=models,
        )

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

    def _cv_splits(
        self,
        X: pd.DataFrame,
        residuals: pd.Series,
    ) -> Any:
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
