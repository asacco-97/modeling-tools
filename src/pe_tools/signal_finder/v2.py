from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

SeriesLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[float]
LabelLike: TypeAlias = pd.Series | np.ndarray[Any, Any] | Sequence[Any] | str | None
IndexLike: TypeAlias = pd.Index | pd.Series | np.ndarray[Any, Any] | Sequence[Any]
CustomSplitLike: TypeAlias = dict[str, IndexLike] | tuple[IndexLike, IndexLike]
ModelType = Literal["xgboost", "random_forest"]
V2SplitStrategy = Literal["bootstrap", "repeated_kfold", "group_kfold"]
NullStrategy = Literal["permuted_features", "random_noise", "shuffled_residuals"]
FeatureType = Literal["continuous", "categorical"]

VALID_MODEL_TYPES = {"xgboost", "random_forest"}
VALID_V2_SPLIT_STRATEGIES = {"bootstrap", "repeated_kfold", "group_kfold"}
VALID_NULL_STRATEGIES = {"permuted_features", "random_noise", "shuffled_residuals"}
OOF_WARNING = (
    "Residual diagnostics are most reliable when base_pred contains out-of-fold or "
    "out-of-sample predictions. In-sample predictions can hide residual signal or create "
    "misleading artifacts."
)


@dataclass(frozen=True)
class _BinSpec:
    feature_type: FeatureType
    labels: list[str]
    edges: list[float | None]
    categories: list[str]
    row_labels: pd.Series | None = None


@dataclass(frozen=True)
class _SplitSpec:
    split_id: int
    train_idx: np.ndarray[Any, Any]
    validation_idx: np.ndarray[Any, Any]
    holdout_idx: np.ndarray[Any, Any] | None = None


@dataclass
class ResidualSignalFinderV2:
    """Bootstrap-based residual signal finder using univariate lift and stability.

    V2 keeps the original residual definition, ``residual = y - base_pred``, but ranks
    candidate features primarily by out-of-sample univariate residual lift, relationship
    stability, and a null/shadow-feature baseline.
    """

    screening_enabled: bool = True
    screening_model_type: ModelType = "xgboost"
    screening_top_k: int = 50
    screening_cv_folds: int = 5
    screening_n_repeats: int = 3
    screening_metric: Literal["permutation_importance", "oof_r2"] = "permutation_importance"
    univariate_model_type: ModelType = "xgboost"
    n_bootstraps: int = 100
    test_size: float = 0.2
    split_strategy: V2SplitStrategy = "bootstrap"
    n_splits: int = 5
    r2_epsilon: float = 0.001
    n_bins: int = 10
    null_strategy: NullStrategy = "permuted_features"
    n_null_features: int = 20
    random_state: int = 42
    max_categories: int = 20
    min_category_count: int = 30
    min_bin_count: int = 20
    classification_scatter_max_bin_size: int = 100
    use_sample_weight: bool = False
    strong_r2_threshold: float = 0.02
    model_params: dict[str, Any] | None = None

    summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    bootstrap_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    effect_curves_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    null_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    screening_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    residual_summary_: dict[str, float] = field(init=False, default_factory=dict)
    warnings_: list[str] = field(init=False, default_factory=list)

    _X: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    _y: pd.Series = field(init=False, default_factory=pd.Series, repr=False)
    _base_pred: pd.Series = field(init=False, default_factory=pd.Series, repr=False)
    _residuals: pd.Series = field(init=False, default_factory=pd.Series, repr=False)
    _sample_weight: pd.Series | None = field(init=False, default=None, repr=False)
    _bin_specs: dict[str, _BinSpec] = field(init=False, default_factory=dict, repr=False)
    _candidate_features: list[str] = field(init=False, default_factory=list, repr=False)
    _is_classification: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.screening_model_type not in VALID_MODEL_TYPES:
            raise ValueError("screening_model_type must be one of: xgboost, random_forest")
        if self.univariate_model_type not in VALID_MODEL_TYPES:
            raise ValueError("univariate_model_type must be one of: xgboost, random_forest")
        if self.screening_top_k < 1:
            raise ValueError("screening_top_k must be at least 1")
        if self.screening_cv_folds < 2:
            raise ValueError("screening_cv_folds must be at least 2")
        if self.screening_n_repeats < 1:
            raise ValueError("screening_n_repeats must be at least 1")
        if self.screening_metric not in {"permutation_importance", "oof_r2"}:
            raise ValueError("screening_metric must be permutation_importance or oof_r2")
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be at least 1")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if self.split_strategy not in VALID_V2_SPLIT_STRATEGIES:
            raise ValueError(
                "split_strategy must be one of: bootstrap, repeated_kfold, group_kfold"
            )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if self.null_strategy not in VALID_NULL_STRATEGIES:
            raise ValueError(
                "null_strategy must be one of: permuted_features, random_noise, shuffled_residuals"
            )
        if self.n_null_features < 0:
            raise ValueError("n_null_features cannot be negative")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least 2")
        if self.min_category_count < 1:
            raise ValueError("min_category_count must be at least 1")
        if self.min_bin_count < 1:
            raise ValueError("min_bin_count must be at least 1")
        if self.classification_scatter_max_bin_size < 1:
            raise ValueError("classification_scatter_max_bin_size must be at least 1")
        if self.strong_r2_threshold <= 0:
            raise ValueError("strong_r2_threshold must be positive")

    def fit(
        self,
        X: pd.DataFrame,
        y: SeriesLike,
        base_pred: SeriesLike,
        *,
        original_model_features: Sequence[str] | None = None,
        sample_weight: SeriesLike | None = None,
        splits: Sequence[CustomSplitLike] | None = None,
        group_col: LabelLike = None,
        segment_cols: Sequence[str] | None = None,
    ) -> ResidualSignalFinderV2:
        """Fit V2 residual signal diagnostics and store result tables on the instance."""
        features = _validate_features(X)
        actuals = _coerce_numeric_series(y, "y", features.index)
        predictions = _coerce_numeric_series(base_pred, "base_pred", features.index)
        weights = (
            _coerce_weight(sample_weight, features.index) if sample_weight is not None else None
        )
        group_values = _coerce_labels(group_col, features, "group_col")

        if len(actuals) != len(features) or len(predictions) != len(features):
            raise ValueError("X, y, and base_pred must have the same length")
        residuals = (actuals - predictions).rename("residual")
        if not np.isfinite(residuals.to_numpy()).all():
            raise ValueError("Residuals must be finite after y - base_pred")

        self._X = features
        self._y = actuals.rename("actual")
        self._base_pred = predictions.rename("base_pred")
        self._residuals = residuals
        self._sample_weight = weights
        self._is_classification = _is_binary_target(actuals)
        self._bin_specs = {
            feature: _make_bin_spec(features[feature], self, weights) for feature in features
        }
        self.warnings_ = [OOF_WARNING]
        self.residual_summary_ = _residual_summary(residuals)
        _add_data_warnings(self.warnings_, features, residuals, self)

        if segment_cols:
            missing_segments = [column for column in segment_cols if column not in features.columns]
            if missing_segments:
                raise ValueError(f"segment_cols not found in X: {missing_segments}")

        split_specs = self._make_splits(features, group_values, splits)
        if not split_specs:
            raise ValueError("No valid train/validation splits were created")

        candidate_features = list(features.columns)
        if self.screening_enabled:
            custom_screening_splits = split_specs if splits is not None else None
            self.screening_results_ = self._screen_features(
                features,
                residuals,
                weights,
                custom_screening_splits,
            )
            candidate_features = _screening_candidates(
                self.screening_results_,
                self.screening_top_k,
            )
        else:
            self.screening_results_ = pd.DataFrame()
        self._candidate_features = candidate_features

        null_features = self._make_null_features(features[candidate_features], residuals)
        real_rows: list[dict[str, Any]] = []
        curve_frames: list[pd.DataFrame] = []
        null_rows: list[dict[str, Any]] = []
        null_curve_frames: list[pd.DataFrame] = []

        for split_spec in split_specs:
            for feature in candidate_features:
                row, curves = self._evaluate_feature(
                    feature_name=feature,
                    values=features[feature],
                    residuals=residuals,
                    actuals=actuals,
                    base_pred=predictions,
                    train_idx=split_spec.train_idx,
                    evaluation_idx=split_spec.validation_idx,
                    split_id=split_spec.split_id,
                    split_role="validation",
                    sample_weight=weights,
                    is_null=False,
                )
                real_rows.append(row)
                curve_frames.append(curves)
                if split_spec.holdout_idx is not None:
                    row, curves = self._evaluate_feature(
                        feature_name=feature,
                        values=features[feature],
                        residuals=residuals,
                        actuals=actuals,
                        base_pred=predictions,
                        train_idx=split_spec.train_idx,
                        evaluation_idx=split_spec.holdout_idx,
                        split_id=split_spec.split_id,
                        split_role="holdout",
                        sample_weight=weights,
                        is_null=False,
                    )
                    real_rows.append(row)
                    curve_frames.append(curves)

            for null_feature, source_feature, values, null_residuals in null_features:
                row, curves = self._evaluate_feature(
                    feature_name=null_feature,
                    values=values,
                    residuals=null_residuals,
                    actuals=actuals,
                    base_pred=predictions,
                    train_idx=split_spec.train_idx,
                    evaluation_idx=split_spec.validation_idx,
                    split_id=split_spec.split_id,
                    split_role="validation",
                    sample_weight=weights,
                    is_null=True,
                    source_feature=source_feature,
                )
                null_rows.append(row)
                null_curve_frames.append(curves)
                if split_spec.holdout_idx is not None:
                    row, curves = self._evaluate_feature(
                        feature_name=null_feature,
                        values=values,
                        residuals=null_residuals,
                        actuals=actuals,
                        base_pred=predictions,
                        train_idx=split_spec.train_idx,
                        evaluation_idx=split_spec.holdout_idx,
                        split_id=split_spec.split_id,
                        split_role="holdout",
                        sample_weight=weights,
                        is_null=True,
                        source_feature=source_feature,
                    )
                    null_rows.append(row)
                    null_curve_frames.append(curves)

        self.bootstrap_results_ = _rank_bootstrap_results(pd.DataFrame(real_rows))
        null_effect_curves = (
            pd.concat(null_curve_frames, ignore_index=True) if null_curve_frames else pd.DataFrame()
        )
        self.null_results_ = _rank_null_results(pd.DataFrame(null_rows), null_effect_curves)
        self.effect_curves_ = (
            pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        )
        self.bootstrap_results_ = _add_null_comparison(
            self.bootstrap_results_,
            self.null_results_,
        )
        self.summary_ = self._build_summary()
        _add_signal_warning(self.warnings_, self.summary_)
        return self

    def get_summary(self) -> pd.DataFrame:
        """Return the feature-level V2 residual signal summary."""
        _require_fitted(self.summary_, "summary_")
        return self.summary_.copy()

    def get_feature_summary(self, feature_name: str) -> pd.Series:
        """Return the summary row for one feature."""
        summary = self.get_summary()
        matches = summary.loc[summary["feature"] == feature_name]
        if matches.empty:
            raise ValueError(f"Unknown feature: {feature_name}")
        return matches.iloc[0].copy()

    def plot_feature_diagnostics(self, feature_name: str) -> Figure:
        """Create a multi-panel diagnostic figure for one evaluated feature."""
        _require_fitted(self.summary_, "summary_")
        self._require_feature(feature_name)

        import matplotlib.pyplot as plt

        curve = _feature_curve_summary(self.effect_curves_, feature_name)
        bins = curve["bin_label"].astype(str).tolist()
        x = np.arange(len(bins))
        figure = plt.figure(figsize=(17, 19))
        grid = figure.add_gridspec(4, 2, height_ratios=[1.0, 1.2, 1.0, 1.0])
        actual_axis = figure.add_subplot(grid[0, :])
        effect_axis = figure.add_subplot(grid[1, :], sharex=actual_axis)
        stability_axis = figure.add_subplot(grid[2, 0])
        null_axis = figure.add_subplot(grid[2, 1])
        scatter_axis = figure.add_subplot(grid[3, :])

        actual_axis.plot(x, curve["actual_mean"], marker="o", label="actual")
        actual_axis.fill_between(
            x,
            curve["actual_ci_low"].to_numpy(dtype=float),
            curve["actual_ci_high"].to_numpy(dtype=float),
            alpha=0.18,
            label="actual 95% CI",
        )
        actual_axis.plot(x, curve["base_prediction_mean"], marker="o", label="base prediction")
        actual_axis.fill_between(
            x,
            curve["base_prediction_ci_low"].to_numpy(dtype=float),
            curve["base_prediction_ci_high"].to_numpy(dtype=float),
            alpha=0.18,
            label="base prediction 95% CI",
        )
        actual_axis.set_title("Actual vs. Base Prediction by Feature Bin (Bootstrap CI)")
        actual_axis.set_ylabel("Mean actual / prediction")
        actual_axis.legend(loc="best")

        self._plot_effect_curve_stability_on_axis(feature_name, effect_axis)
        self._plot_effect_curve_correlation_on_axis(feature_name, stability_axis)
        self._plot_null_comparison_on_axis(feature_name, null_axis)
        self._plot_residual_scatter_on_axis(feature_name, scatter_axis)

        actual_axis.set_xticks(x)
        actual_axis.set_xticklabels([])
        effect_axis.set_xticks(x)
        effect_axis.set_xticklabels(bins, rotation=35, ha="right")
        effect_axis.tick_params(axis="x", labelsize=9)
        figure.suptitle(f"Residual Signal Diagnostics: {feature_name}", y=1.01)
        figure.tight_layout()
        return figure

    def plot_top_features(self, n: int = 10) -> dict[str, Figure]:
        """Return diagnostic figures for the top ranked V2 features."""
        if n < 1:
            raise ValueError("n must be at least 1")
        return {
            feature: self.plot_feature_diagnostics(feature)
            for feature in self.summary_.head(n)["feature"].astype(str)
        }

    def plot_effect_curve(self, feature_name: str) -> Figure:
        """Plot the average centered residual curve for one feature."""
        _require_fitted(self.summary_, "summary_")
        self._require_feature(feature_name)

        import matplotlib.pyplot as plt

        curve = _feature_curve_summary(self.effect_curves_, feature_name)
        x = np.arange(len(curve))
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(x, curve["centered_mean_residual"], marker="o", color="black")
        axis.fill_between(
            x,
            curve["centered_residual_p05"],
            curve["centered_residual_p95"],
            color="black",
            alpha=0.15,
            label="5p to 95p bootstrap band",
        )
        axis.axhline(0.0, linestyle=":", color="black")
        axis.set_xticks(x)
        axis.set_xticklabels(curve["bin_label"].astype(str), rotation=45, ha="right")
        axis.set_title(f"Residual Effect Curve: {feature_name}")
        axis.set_ylabel("Centered mean residual")
        axis.legend(loc="best")
        figure.tight_layout()
        return figure

    def plot_null_comparison(self, feature_name: str) -> Figure:
        """Plot real feature bootstrap scores against the null score distribution."""
        _require_fitted(self.summary_, "summary_")
        self._require_feature(feature_name)

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 5))
        self._plot_null_comparison_on_axis(feature_name, axis)
        figure.tight_layout()
        return figure

    def plot_rank_stability(self, n: int = 20) -> Figure:
        """Plot mean and median residual-signal rank for top features."""
        if n < 1:
            raise ValueError("n must be at least 1")
        _require_fitted(self.summary_, "summary_")

        import matplotlib.pyplot as plt

        table = self.summary_.head(n).copy()
        x = np.arange(len(table))
        figure, axis = plt.subplots(figsize=(12, 5))
        width = 0.38
        axis.bar(
            x - width / 2,
            table["mean_residual_signal_rank"],
            width=width,
            color="#4c78a8",
            label="mean rank",
        )
        axis.bar(
            x + width / 2,
            table["median_residual_signal_rank"],
            width=width,
            color="#f58518",
            label="median rank",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(table["feature"].astype(str), rotation=45, ha="right")
        axis.set_ylabel("Rank within bootstrap")
        axis.set_title("Residual Signal Rank Stability")
        axis.legend()
        figure.tight_layout()
        return figure

    def plot_residual_signal_map(self) -> Figure:
        """Plot signal strength against credibility/actionability."""
        _require_fitted(self.summary_, "summary_")

        import matplotlib.pyplot as plt

        table = self.summary_.copy()
        stability = table["median_effect_curve_spearman_stability"].fillna(0.0).clip(-1.0, 1.0)
        size = 60 + 240 * table["pct_positive_feature_residual_spearman"].fillna(0.0)
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.scatter(
            table["mean_oof_residual_r2"],
            stability,
            s=size,
            alpha=0.65,
            color="#4c78a8",
        )
        for row in table.head(10).itertuples(index=False):
            axis.annotate(
                str(row.feature),
                (row.mean_oof_residual_r2, row.median_effect_curve_spearman_stability),
            )
        axis.axvline(0.0, linestyle=":", color="black")
        axis.set_xlabel("Mean OOF residual R²")
        axis.set_ylabel("Median effect-curve Spearman stability")
        axis.set_title("Residual Signal Map")
        figure.tight_layout()
        return figure

    def _screen_features(
        self,
        features: pd.DataFrame,
        residuals: pd.Series,
        sample_weight: pd.Series | None,
        custom_splits: Sequence[_SplitSpec] | None,
    ) -> pd.DataFrame:
        encoded = _encoded_feature_frame(features, self)
        rows: list[dict[str, Any]] = []
        repeats = self.screening_n_repeats
        rng = np.random.default_rng(self.random_state)
        screening_splits: list[tuple[int, int, np.ndarray[Any, Any], np.ndarray[Any, Any]]]
        if custom_splits is not None:
            screening_splits = [
                (0, split.split_id, split.train_idx, split.validation_idx)
                for split in custom_splits
            ]
        else:
            screening_splits = []
            fold_count = min(self.screening_cv_folds, len(features))
            for repeat in range(repeats):
                splitter = KFold(
                    n_splits=fold_count,
                    shuffle=True,
                    random_state=self.random_state + repeat,
                )
                for fold, (train_idx, validation_idx) in enumerate(splitter.split(encoded)):
                    screening_splits.append((repeat, fold, train_idx, validation_idx))

        for repeat, fold, train_idx, validation_idx in screening_splits:
            model = self._make_model(
                self.screening_model_type,
                seed=self.random_state + repeat + fold,
            )
            train_X = encoded.iloc[train_idx]
            validation_X = encoded.iloc[validation_idx]
            train_y = residuals.iloc[train_idx]
            validation_y = residuals.iloc[validation_idx]
            weights_train = _weights_for_fit(sample_weight, train_idx, self.use_sample_weight)
            model.fit(train_X, train_y, sample_weight=weights_train)
            baseline_pred = np.asarray(model.predict(validation_X), dtype=float)
            baseline_r2 = _oof_r2(
                validation_y.to_numpy(),
                baseline_pred,
                _weighted_mean(train_y.to_numpy(), weights_train),
                _weights_for_score(sample_weight, validation_idx, self.use_sample_weight),
            )
            importances: list[tuple[str, float]] = []
            for feature in encoded.columns:
                permuted = validation_X.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                permuted_pred = np.asarray(model.predict(permuted), dtype=float)
                permuted_r2 = _oof_r2(
                    validation_y.to_numpy(),
                    permuted_pred,
                    _weighted_mean(train_y.to_numpy(), weights_train),
                    _weights_for_score(sample_weight, validation_idx, self.use_sample_weight),
                )
                importances.append((feature, baseline_r2 - permuted_r2))

            ranked = sorted(importances, key=lambda item: item[1], reverse=True)
            ranks = {feature: rank for rank, (feature, _score) in enumerate(ranked, start=1)}
            for feature, importance in importances:
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "feature": feature,
                        "screening_oof_r2": baseline_r2,
                        "permutation_importance": importance,
                        "rank": ranks[feature],
                        "is_top_5": ranks[feature] <= 5,
                        "is_top_10": ranks[feature] <= 10,
                    }
                )

        raw = pd.DataFrame(rows)
        if raw.empty:
            return raw
        summaries = []
        for feature, group in raw.groupby("feature", sort=False):
            summaries.append(
                {
                    "feature": feature,
                    "screening_mean_perm_importance": group["permutation_importance"].mean(),
                    "screening_median_perm_importance": group["permutation_importance"].median(),
                    "screening_importance_std": group["permutation_importance"].std(ddof=0),
                    "screening_median_rank": group["rank"].median(),
                    "screening_top_5_rate": group["is_top_5"].mean(),
                    "screening_top_10_rate": group["is_top_10"].mean(),
                }
            )
        return pd.DataFrame(summaries).sort_values(
            by=[
                "screening_median_rank",
                "screening_top_5_rate",
                "screening_mean_perm_importance",
            ],
            ascending=[True, False, False],
            ignore_index=True,
        )

    def _make_splits(
        self,
        features: pd.DataFrame,
        group_values: pd.Series | None,
        custom_splits: Sequence[CustomSplitLike] | None,
    ) -> list[_SplitSpec]:
        if custom_splits is not None:
            return _normalize_custom_splits(custom_splits, features.index)

        n_obs = len(features)
        positions = np.arange(n_obs)
        if self.split_strategy == "group_kfold":
            if group_values is None:
                raise ValueError("group_col is required when split_strategy='group_kfold'")
            unique_groups = pd.Series(group_values).nunique(dropna=False)
            if unique_groups < self.n_splits:
                raise ValueError("group_col must contain at least n_splits unique groups")
            splitter = GroupKFold(n_splits=self.n_splits)
            return [
                _SplitSpec(
                    split_id=split_id,
                    train_idx=np.asarray(train_idx),
                    validation_idx=np.asarray(validation_idx),
                )
                for split_id, (train_idx, validation_idx) in enumerate(
                    splitter.split(features, groups=group_values)
                )
            ]

        if self.split_strategy == "repeated_kfold":
            splits: list[_SplitSpec] = []
            repeat_count = int(np.ceil(self.n_bootstraps / self.n_splits))
            for repeat in range(repeat_count):
                splitter = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state + repeat,
                )
                for train_idx, validation_idx in splitter.split(features):
                    splits.append(
                        _SplitSpec(
                            split_id=len(splits),
                            train_idx=np.asarray(train_idx),
                            validation_idx=np.asarray(validation_idx),
                        )
                    )
                    if len(splits) >= self.n_bootstraps:
                        return splits
            return splits

        rng = np.random.default_rng(self.random_state)
        train_size = max(1, min(n_obs - 1, int(round((1.0 - self.test_size) * n_obs))))
        splits = []
        for _bootstrap_id in range(self.n_bootstraps):
            train_idx = np.sort(rng.choice(positions, size=train_size, replace=False))
            validation_idx = np.setdiff1d(positions, train_idx)
            if len(validation_idx) > 0:
                splits.append(
                    _SplitSpec(
                        split_id=len(splits),
                        train_idx=train_idx,
                        validation_idx=validation_idx,
                    )
                )
        return splits

    def _make_null_features(
        self,
        features: pd.DataFrame,
        residuals: pd.Series,
    ) -> list[tuple[str, str, pd.Series, pd.Series]]:
        if self.n_null_features == 0 or features.empty:
            return []
        rng = np.random.default_rng(self.random_state + 100_003)
        source_features = rng.choice(
            features.columns.to_numpy(),
            size=min(self.n_null_features, len(features.columns)),
            replace=False,
        )
        null_features = []
        for null_id, source_feature in enumerate(source_features):
            name = f"__null_{null_id}_{source_feature}"
            if self.null_strategy == "random_noise":
                values = pd.Series(rng.normal(size=len(features)), index=features.index, name=name)
                null_residuals = residuals
            elif self.null_strategy == "shuffled_residuals":
                values = features[str(source_feature)].rename(name)
                null_residuals = pd.Series(
                    rng.permutation(residuals.to_numpy()),
                    index=residuals.index,
                    name="shuffled_residual",
                )
            else:
                values = pd.Series(
                    rng.permutation(features[str(source_feature)].to_numpy()),
                    index=features.index,
                    name=name,
                )
                null_residuals = residuals
            null_features.append((name, str(source_feature), values, null_residuals))
        return null_features

    def _evaluate_feature(
        self,
        *,
        feature_name: str,
        values: pd.Series,
        residuals: pd.Series,
        actuals: pd.Series,
        base_pred: pd.Series,
        train_idx: np.ndarray[Any, Any],
        evaluation_idx: np.ndarray[Any, Any],
        split_id: int,
        split_role: Literal["validation", "holdout"],
        sample_weight: pd.Series | None,
        is_null: bool,
        source_feature: str | None = None,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        feature_type = _infer_feature_type(values)
        bin_spec = self._bin_specs.get(source_feature or feature_name)
        if bin_spec is None or is_null:
            bin_spec = _make_bin_spec(values, self, sample_weight)

        model_values = _model_values(values, feature_type, bin_spec, train_idx)
        train_X = pd.DataFrame({"feature": model_values[train_idx]})
        validation_X = pd.DataFrame({"feature": model_values[evaluation_idx]})
        train_y = residuals.iloc[train_idx]
        validation_y = residuals.iloc[evaluation_idx]
        weights_train = _weights_for_fit(sample_weight, train_idx, self.use_sample_weight)
        weights_validation = _weights_for_score(
            sample_weight,
            evaluation_idx,
            self.use_sample_weight,
        )

        model = self._make_model(
            self.univariate_model_type,
            seed=self.random_state + split_id,
        )
        model.fit(train_X, train_y, sample_weight=weights_train)
        predicted_residual = np.asarray(model.predict(validation_X), dtype=float)
        train_mean = _weighted_mean(train_y.to_numpy(), weights_train)
        oof_r2 = _oof_r2(
            validation_y.to_numpy(),
            predicted_residual,
            train_mean,
            weights_validation,
        )
        spearman = _spearman(values.iloc[evaluation_idx], validation_y, feature_type)
        curves = _effect_curve_frame(
            feature_name=feature_name,
            values=values,
            bin_spec=bin_spec,
            residuals=residuals,
            actuals=actuals,
            base_pred=base_pred,
            predicted_residual=predicted_residual,
            validation_idx=evaluation_idx,
            bootstrap_id=split_id,
            split_id=split_id,
            split_role=split_role,
            sample_weight=sample_weight if self.use_sample_weight else None,
            min_bin_count=self.min_bin_count,
        )
        return (
            {
                "bootstrap_id": split_id,
                "split_id": split_id,
                "split_role": split_role,
                "feature": feature_name,
                "source_feature": source_feature,
                "is_null": is_null,
                "oof_r2": oof_r2,
                "spearman": spearman,
            },
            curves,
        )

    def _make_model(self, model_type: ModelType, seed: int) -> Any:
        params: dict[str, Any]
        if model_type == "xgboost":
            params = {
                "n_estimators": 120,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 1.0,
                "min_child_weight": 10,
                "reg_lambda": 2.0,
                "objective": "reg:squarederror",
                "random_state": seed,
                "n_jobs": 1,
                "verbosity": 0,
            }
            params.update(self.model_params or {})
            return XGBRegressor(**params)

        params = {
            "n_estimators": 160,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "max_features": 1.0,
            "bootstrap": True,
            "random_state": seed,
            "n_jobs": 1,
        }
        params.update(self.model_params or {})
        return RandomForestRegressor(**params)

    def _build_summary(self) -> pd.DataFrame:
        if self.bootstrap_results_.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        validation_results = self.bootstrap_results_.loc[
            self.bootstrap_results_["split_role"].eq("validation")
        ]
        validation_curves = self.effect_curves_.loc[
            self.effect_curves_["split_role"].eq("validation")
        ]
        for feature, group in validation_results.groupby("feature", sort=False):
            feature_values = self._X[feature]
            curve_stats = _effect_curve_stats(validation_curves, feature)
            spearman_values = group["spearman"].dropna()
            percent_positive = (
                float((spearman_values > 0.0).mean()) if len(spearman_values) else np.nan
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_type": _infer_feature_type(feature_values),
                    "n_observations": int(len(feature_values)),
                    "mean_oof_residual_r2": float(group["oof_r2"].mean()),
                    "median_oof_residual_r2": float(group["oof_r2"].median()),
                    "p05_oof_residual_r2": _quantile(group["oof_r2"], 0.05),
                    "p95_oof_residual_r2": _quantile(group["oof_r2"], 0.95),
                    "mean_residual_signal_rank": float(group["rank"].mean()),
                    "median_residual_signal_rank": float(group["rank"].median()),
                    "mean_feature_residual_spearman": float(spearman_values.mean())
                    if len(spearman_values)
                    else np.nan,
                    "median_feature_residual_spearman": float(spearman_values.median())
                    if len(spearman_values)
                    else np.nan,
                    "pct_positive_feature_residual_spearman": percent_positive,
                    "mean_effect_curve_spearman_stability": curve_stats[
                        "mean_effect_curve_spearman_stability"
                    ],
                    "median_effect_curve_spearman_stability": curve_stats[
                        "median_effect_curve_spearman_stability"
                    ],
                    "std_effect_curve_spearman_stability": curve_stats[
                        "std_effect_curve_spearman_stability"
                    ],
                }
            )

        return pd.DataFrame(rows).sort_values(
            by=[
                "mean_oof_residual_r2",
                "median_residual_signal_rank",
                "median_effect_curve_spearman_stability",
            ],
            ascending=[False, True, False],
            ignore_index=True,
        )

    def _require_feature(self, feature_name: str) -> None:
        if feature_name not in set(self.summary_["feature"].astype(str)):
            raise ValueError(f"Unknown feature: {feature_name}")

    def _full_bin_frame(self, feature_name: str) -> pd.DataFrame:
        values = self._X[feature_name]
        labels = _assign_bins(values, self._bin_specs[feature_name])
        return pd.DataFrame(
            {
                "bin_label": pd.Categorical(
                    labels,
                    categories=self._bin_specs[feature_name].labels,
                    ordered=True,
                ),
                "residual": self._residuals.to_numpy(),
                "abs_residual": np.abs(self._residuals.to_numpy()),
            }
        )

    def _missingness_frame(self, feature_name: str) -> pd.DataFrame:
        missing = self._X[feature_name].isna()
        if not missing.any():
            return pd.DataFrame()
        rows = []
        for status, mask in {"missing": missing, "observed": ~missing}.items():
            rows.append(
                {
                    "status": status,
                    "n_obs": int(mask.sum()),
                    "mean_residual": float(self._residuals.loc[mask].mean()),
                    "mean_abs_residual": float(np.abs(self._residuals.loc[mask]).mean()),
                }
            )
        return pd.DataFrame(rows)

    def _plot_null_comparison_on_axis(self, feature_name: str, axis: Any) -> None:
        real_scores = self.bootstrap_results_.loc[
            self.bootstrap_results_["feature"] == feature_name,
            "oof_r2",
        ]
        null_scores = self.null_results_["oof_r2"] if not self.null_results_.empty else pd.Series()
        if not null_scores.empty:
            axis.hist(null_scores, bins=15, alpha=0.45, label="null", color="lightgray")
        axis.hist(real_scores, bins=10, alpha=0.65, label=feature_name, color="#4c78a8")
        axis.axvline(real_scores.mean(), linestyle="-", color="#4c78a8", label="real mean")
        if not null_scores.empty:
            axis.axvline(null_scores.quantile(0.95), linestyle=":", color="black", label="null p95")
        axis.set_title("Null Baseline Comparison: Real Feature vs Shadow Features")
        axis.set_xlabel("OOF residual R²")
        axis.legend()

    def _plot_effect_curve_stability_on_axis(self, feature_name: str, axis: Any) -> None:
        curves = self.effect_curves_.loc[self.effect_curves_["feature"] == feature_name]
        if curves.empty:
            axis.text(0.5, 0.5, "No effect curves available", ha="center", va="center")
            axis.axis("off")
            return

        curve_matrix = _effect_curve_matrix(self.effect_curves_, feature_name)
        curve_summary = _feature_curve_summary(self.effect_curves_, feature_name)
        x = np.arange(len(curve_summary))
        for bootstrap_id, row in curve_matrix.iterrows():
            axis.plot(
                x,
                row.reindex(curve_summary["bin_id"]).to_numpy(dtype=float),
                color="darkgray",
                linewidth=0.9,
                alpha=0.75,
                label="bootstrap runs" if bootstrap_id == curve_matrix.index[0] else None,
            )
        axis.plot(
            x,
            curve_summary["centered_mean_residual"],
            color="black",
            linewidth=2.2,
            marker="o",
            label="mean error",
        )
        axis.fill_between(
            x,
            curve_summary["centered_residual_p05"].to_numpy(dtype=float),
            curve_summary["centered_residual_p95"].to_numpy(dtype=float),
            color="black",
            alpha=0.14,
            label="5p to 95p bootstrap band",
        )
        axis.axhline(0.0, linestyle=":", color="black", linewidth=1.0)
        axis.set_title(
            "Prediction Error by Feature Bin Across Bootstrap Runs "
            "(Actual - Base Prediction)"
        )
        axis.set_ylabel("Centered mean residual")
        axis.legend(loc="best")

    def _plot_effect_curve_correlation_on_axis(self, feature_name: str, axis: Any) -> None:
        correlations = _effect_curve_pairwise_spearman(self.effect_curves_, feature_name)
        if correlations.empty:
            axis.text(
                0.5,
                0.5,
                "Not enough non-constant bootstrap curves",
                ha="center",
                va="center",
            )
            axis.axis("off")
            return
        axis.hist(correlations, bins=15, color="#4c78a8", alpha=0.75)
        mean_value = float(correlations.mean())
        median_value = float(correlations.median())
        axis.axvline(
            mean_value,
            color="black",
            linewidth=2.0,
            label=f"mean Spearman: {mean_value:.2f}",
        )
        axis.axvline(
            median_value,
            color="darkgray",
            linestyle="--",
            linewidth=2.0,
            label=f"median Spearman: {median_value:.2f}",
        )
        axis.set_title(f"{feature_name}: Spearman Rank Correlation Between Effect Curves")
        axis.set_xlabel("Pairwise bootstrap curve Spearman correlation")
        axis.set_ylabel("Bootstrap-pair count")
        axis.legend(loc="best")

    def _plot_residual_scatter_on_axis(self, feature_name: str, axis: Any) -> None:
        values = self._X[feature_name]
        feature_type = _infer_feature_type(values)
        if self._is_classification:
            scatter_frame = self._classification_residual_scatter_frame(feature_name)
            axis.scatter(
                scatter_frame["x_value"],
                scatter_frame["mean_residual"],
                s=scatter_frame["point_size"],
                alpha=0.7,
                color="#4c78a8",
            )
            axis.set_ylabel("Mean residual per small bin")
            axis.set_title("Classification Residual Scatter by Feature Bins")
        else:
            scatter_frame = self._observation_residual_scatter_frame(feature_name)
            axis.scatter(
                scatter_frame["x_value"],
                scatter_frame["residual"],
                s=14,
                alpha=0.35,
                color="#4c78a8",
                edgecolors="none",
            )
            axis.set_ylabel("Residual")
            axis.set_title("Observation-Level Residual Scatter")

        axis.axhline(0.0, linestyle=":", color="black")
        axis.set_xlabel(feature_name)
        if self._is_classification or feature_type == "categorical":
            _set_discrete_axis_labels(axis, scatter_frame["x_value"], scatter_frame["x_label"])

    def _observation_residual_scatter_frame(self, feature_name: str) -> pd.DataFrame:
        values = self._X[feature_name]
        feature_type = _infer_feature_type(values)
        if feature_type == "categorical":
            labels = _assign_bins(values, self._bin_specs[feature_name])
            categories = list(dict.fromkeys(labels.astype(str).tolist()))
            mapping = {label: index for index, label in enumerate(categories)}
            x_value = labels.map(mapping).to_numpy(dtype=float)
            x_label = labels.astype(str).to_numpy()
        else:
            x_value = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            x_label = values.astype(str).to_numpy()
        return pd.DataFrame(
            {
                "x_value": x_value,
                "x_label": x_label,
                "residual": self._residuals.to_numpy(dtype=float),
            }
        ).dropna(subset=["x_value", "residual"])

    def _classification_residual_scatter_frame(self, feature_name: str) -> pd.DataFrame:
        values = self._X[feature_name]
        feature_type = _infer_feature_type(values)
        residuals = self._residuals.to_numpy(dtype=float)
        if feature_type == "categorical":
            labels = _assign_bins(values, self._bin_specs[feature_name])
            categories = list(dict.fromkeys(labels.astype(str).tolist()))
            mapping = {label: index for index, label in enumerate(categories)}
            frame = pd.DataFrame(
                {
                    "x_value": labels.map(mapping).to_numpy(dtype=float),
                    "x_label": labels.astype(str).to_numpy(),
                    "residual": residuals,
                }
            )
            grouped = frame.groupby(["x_value", "x_label"], sort=True)
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            frame = pd.DataFrame(
                {
                    "feature_value": numeric.to_numpy(dtype=float),
                    "residual": residuals,
                    "actual": self._y.to_numpy(dtype=float),
                }
            ).dropna(subset=["feature_value", "residual"])
            bin_size = self._classification_scatter_bin_size()
            frame = frame.sort_values("feature_value").reset_index(drop=True)
            frame["bin_id"] = np.arange(len(frame)) // bin_size
            grouped = frame.groupby("bin_id", sort=True)

        rows = []
        for key, group in grouped:
            if feature_type == "categorical":
                x_value = float(group["x_value"].iloc[0])
                x_label = str(group["x_label"].iloc[0])
            else:
                x_value = float(key)
                x_label = _balanced_bin_label(
                    int(key) + 1,
                    float(group["feature_value"].min()),
                    float(group["feature_value"].max()),
                )
            rows.append(
                {
                    "x_value": x_value,
                    "x_label": x_label,
                    "mean_residual": float(group["residual"].mean()),
                    "n_observations": int(len(group)),
                    "point_size": 25.0 + min(float(len(group)), 200.0),
                }
            )
        return pd.DataFrame(rows)

    def _classification_scatter_bin_size(self) -> int:
        positive_rate = float(self._y.mean())
        if not 0.0 < positive_rate < 1.0:
            return self.classification_scatter_max_bin_size
        min_size_for_events = int(np.ceil(3.0 / positive_rate))
        return int(
            np.clip(
                max(20, min_size_for_events),
                1,
                self.classification_scatter_max_bin_size,
            )
        )


def _validate_features(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if X.empty:
        raise ValueError("X must contain at least one feature")
    if X.columns.duplicated().any():
        raise ValueError("X cannot contain duplicated column names")
    columns = [str(column) for column in X.columns]
    if any(not column or column.strip() == "" for column in columns):
        raise ValueError("Feature names must be non-empty strings")
    features = X.copy()
    features.columns = columns
    return features


def _coerce_numeric_series(values: SeriesLike, name: str, index: pd.Index) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if len(series) != len(index):
        raise ValueError(f"{name} must have the same length as X")
    if isinstance(values, pd.Series) and not values.index.equals(index):
        raise ValueError(f"{name} index must align with X")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} must be numeric and cannot contain missing values")
    return pd.Series(numeric.to_numpy(dtype=float), index=index, name=name)


def _coerce_weight(values: SeriesLike, index: pd.Index) -> pd.Series:
    weight = _coerce_numeric_series(values, "sample_weight", index)
    if (weight < 0).any():
        raise ValueError("sample_weight cannot contain negative values")
    if float(weight.sum()) == 0.0:
        raise ValueError("sample_weight cannot be all zero")
    return weight


def _coerce_labels(values: LabelLike, X: pd.DataFrame, name: str) -> pd.Series | None:
    if values is None:
        return None
    if isinstance(values, str):
        if values not in X.columns:
            raise ValueError(f"{name} column not found in X: {values}")
        return X[values].copy()
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if len(series) != len(X):
        raise ValueError(f"{name} must have the same length as X")
    if isinstance(values, pd.Series) and not values.index.equals(X.index):
        raise ValueError(f"{name} index must align with X")
    return pd.Series(series.to_numpy(), index=X.index, name=name)


def _normalize_custom_splits(
    splits: Sequence[CustomSplitLike],
    index: pd.Index,
) -> list[_SplitSpec]:
    if not splits:
        raise ValueError("splits must contain at least one split")
    normalized = []
    for split_id, split in enumerate(splits):
        train, validation, holdout = _extract_custom_split_parts(split, split_id)
        train_idx = _normalize_split_index(train, index, f"splits[{split_id}]['train']")
        validation_idx = _normalize_split_index(
            validation,
            index,
            f"splits[{split_id}]['validation']",
        )
        holdout_idx = (
            _normalize_split_index(holdout, index, f"splits[{split_id}]['holdout']")
            if holdout is not None
            else None
        )
        if len(train_idx) == 0:
            raise ValueError(f"splits[{split_id}] train indices cannot be empty")
        if len(validation_idx) == 0:
            raise ValueError(f"splits[{split_id}] validation indices cannot be empty")
        if holdout_idx is not None and len(holdout_idx) == 0:
            raise ValueError(f"splits[{split_id}] holdout indices cannot be empty")
        _validate_split_disjoint(split_id, train_idx, validation_idx, holdout_idx)
        normalized.append(
            _SplitSpec(
                split_id=split_id,
                train_idx=train_idx,
                validation_idx=validation_idx,
                holdout_idx=holdout_idx,
            )
        )
    return normalized


def _extract_custom_split_parts(
    split: CustomSplitLike,
    split_id: int,
) -> tuple[IndexLike, IndexLike, IndexLike | None]:
    if isinstance(split, dict):
        missing = {"train", "validation"} - set(split)
        if missing:
            raise ValueError(f"splits[{split_id}] is missing required keys: {sorted(missing)}")
        return split["train"], split["validation"], split.get("holdout")
    if isinstance(split, tuple) and len(split) == 2:
        return split[0], split[1], None
    raise ValueError(
        "Each split must be a dict with train/validation keys or a "
        "(train_idx, validation_idx) tuple"
    )


def _normalize_split_index(values: IndexLike, index: pd.Index, name: str) -> np.ndarray[Any, Any]:
    if isinstance(values, pd.Series) and pd.api.types.is_bool_dtype(values):
        if len(values) != len(index):
            raise ValueError(f"{name} boolean mask must have the same length as X")
        if values.index.equals(index):
            return np.flatnonzero(values.to_numpy(dtype=bool))
        return np.flatnonzero(values.reset_index(drop=True).to_numpy(dtype=bool))

    array = np.asarray(values)
    if array.dtype == bool:
        if len(array) != len(index):
            raise ValueError(f"{name} boolean mask must have the same length as X")
        return np.flatnonzero(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if len(array) == 0:
        return np.asarray([], dtype=int)
    if isinstance(values, pd.Index | pd.Series):
        positions = index.get_indexer(array)
        if (positions < 0).any():
            missing = array[positions < 0][:5].tolist()
            raise ValueError(f"{name} contains labels not present in X.index: {missing}")
        return np.asarray(positions, dtype=int)

    if np.issubdtype(array.dtype, np.integer):
        positions = array.astype(int)
        if (positions < 0).any() or (positions >= len(index)).any():
            raise ValueError(f"{name} positional indices must be within the X row range")
        return positions

    positions = index.get_indexer(array)
    if (positions < 0).any():
        missing = array[positions < 0][:5].tolist()
        raise ValueError(f"{name} contains labels not present in X.index: {missing}")
    return np.asarray(positions, dtype=int)


def _validate_split_disjoint(
    split_id: int,
    train_idx: np.ndarray[Any, Any],
    validation_idx: np.ndarray[Any, Any],
    holdout_idx: np.ndarray[Any, Any] | None,
) -> None:
    seen = {
        "train": set(train_idx.tolist()),
        "validation": set(validation_idx.tolist()),
        "holdout": set(holdout_idx.tolist()) if holdout_idx is not None else set(),
    }
    if len(seen["train"]) != len(train_idx):
        raise ValueError(f"splits[{split_id}] train indices contain duplicates")
    if len(seen["validation"]) != len(validation_idx):
        raise ValueError(f"splits[{split_id}] validation indices contain duplicates")
    if holdout_idx is not None and len(seen["holdout"]) != len(holdout_idx):
        raise ValueError(f"splits[{split_id}] holdout indices contain duplicates")
    for left, right in (("train", "validation"), ("train", "holdout"), ("validation", "holdout")):
        if seen[left] & seen[right]:
            raise ValueError(f"splits[{split_id}] {left} and {right} indices must be disjoint")


def _infer_feature_type(values: pd.Series) -> FeatureType:
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return "continuous"
    return "categorical"


def _is_binary_target(values: pd.Series) -> bool:
    unique_values = set(pd.to_numeric(values, errors="coerce").dropna().unique().tolist())
    return bool(unique_values) and unique_values.issubset({0.0, 1.0})


def _make_bin_spec(
    values: pd.Series,
    finder: ResidualSignalFinderV2,
    sample_weight: pd.Series | None = None,
) -> _BinSpec:
    feature_type = _infer_feature_type(values)
    if feature_type == "categorical":
        category_labels = _categorical_labels(values, finder)
        return _BinSpec(
            feature_type="categorical",
            labels=category_labels,
            edges=[None] * len(category_labels),
            categories=category_labels,
        )

    numeric = pd.to_numeric(values, errors="coerce")
    non_missing = numeric.dropna().sort_values(kind="mergesort")
    if non_missing.nunique() <= 1:
        label = str(non_missing.iloc[0]) if not non_missing.empty else "__MISSING__"
        row_labels = pd.Series(label, index=values.index, name=values.name, dtype="object")
        singleton_labels = [label]
        if numeric.isna().any() and "__MISSING__" not in singleton_labels:
            singleton_labels.append("__MISSING__")
            row_labels.loc[numeric.isna()] = "__MISSING__"
        return _BinSpec(
            "continuous",
            singleton_labels,
            [None] * len(singleton_labels),
            [],
            row_labels,
        )

    sorted_weights = None
    if sample_weight is not None:
        sorted_weights = sample_weight.reindex(non_missing.index).to_numpy(dtype=float)
    bin_positions = _balanced_bin_positions(
        n_obs=len(non_missing),
        n_bins=finder.n_bins,
        sample_weight=sorted_weights,
    )
    row_labels = pd.Series(index=values.index, dtype="object", name=values.name)
    continuous_labels: list[str] = []
    edges_list: list[float | None] = []
    for bin_id, positions in enumerate(bin_positions, start=1):
        bin_values = non_missing.iloc[positions]
        left = float(bin_values.min())
        right = float(bin_values.max())
        label = _balanced_bin_label(bin_id, left, right)
        continuous_labels.append(label)
        edges_list.append(right)
        row_labels.loc[bin_values.index] = label
    if numeric.isna().any():
        continuous_labels.append("__MISSING__")
        edges_list.append(None)
        row_labels.loc[numeric.isna()] = "__MISSING__"
    return _BinSpec("continuous", continuous_labels, edges_list, [], row_labels)


def _balanced_bin_positions(
    *,
    n_obs: int,
    n_bins: int,
    sample_weight: np.ndarray[Any, Any] | None,
) -> list[np.ndarray[Any, Any]]:
    bin_count = max(1, min(n_bins, n_obs))
    positions = np.arange(n_obs)
    if sample_weight is None:
        return [np.asarray(chunk) for chunk in np.array_split(positions, bin_count) if len(chunk)]

    clean_weight = np.asarray(sample_weight, dtype=float)
    if (
        len(clean_weight) != n_obs
        or not np.isfinite(clean_weight).all()
        or clean_weight.sum() <= 0.0
    ):
        return [np.asarray(chunk) for chunk in np.array_split(positions, bin_count) if len(chunk)]

    cumulative_weight = np.cumsum(clean_weight)
    total_weight = float(cumulative_weight[-1])
    boundaries = [0]
    for bin_id in range(1, bin_count):
        target = total_weight * bin_id / bin_count
        position = int(np.searchsorted(cumulative_weight, target, side="right"))
        min_position = boundaries[-1] + 1
        max_position = n_obs - (bin_count - bin_id)
        boundaries.append(int(np.clip(position, min_position, max_position)))
    boundaries.append(n_obs)
    return [
        positions[left:right]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        if right > left
    ]


def _balanced_bin_label(bin_id: int, left: float, right: float) -> str:
    left_label = _format_bin_value(left)
    right_label = _format_bin_value(right)
    if left == right:
        return f"Bin {bin_id}: {left_label}"
    return f"Bin {bin_id}: {left_label} to {right_label}"


def _format_bin_value(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    sign = "-" if value < 0 else ""
    absolute = abs(float(value))
    if absolute >= 1_000_000_000:
        return f"{sign}{absolute / 1_000_000_000:.2f}B".rstrip("0").rstrip(".")
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M".rstrip("0").rstrip(".")
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.2f}K".rstrip("0").rstrip(".")
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if absolute >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _set_discrete_axis_labels(
    axis: Any,
    x_values: pd.Series,
    x_labels: pd.Series,
    max_labels: int = 20,
) -> None:
    ticks = (
        pd.DataFrame({"x_value": x_values, "x_label": x_labels})
        .drop_duplicates()
        .sort_values("x_value")
    )
    if len(ticks) > max_labels:
        step = int(np.ceil(len(ticks) / max_labels))
        ticks = pd.concat([ticks.iloc[::step], ticks.tail(1)]).drop_duplicates("x_value")
    axis.set_xticks(ticks["x_value"])
    axis.set_xticklabels(ticks["x_label"], rotation=35, ha="right")
    axis.tick_params(axis="x", labelsize=8)


def _categorical_labels(values: pd.Series, finder: ResidualSignalFinderV2) -> list[str]:
    labels = values.astype("object").where(values.notna(), "__MISSING__").astype(str)
    counts = labels.value_counts(dropna=False)
    keepers = [
        label
        for label, count in counts.items()
        if count >= finder.min_category_count and label != "__MISSING__"
    ][: finder.max_categories]
    final_labels = list(keepers)
    if ((~labels.isin(keepers)) & labels.ne("__MISSING__")).any():
        final_labels.append("__OTHER__")
    if labels.eq("__MISSING__").any():
        final_labels.append("__MISSING__")
    return final_labels or ["__OTHER__"]


def _assign_bins(values: pd.Series, spec: _BinSpec) -> pd.Series:
    if spec.feature_type == "categorical":
        labels = values.astype("object").where(values.notna(), "__MISSING__").astype(str)
        keepers = set(spec.labels)
        mapped = labels.where(labels.isin(keepers), "__OTHER__")
        if "__OTHER__" not in keepers:
            mapped = mapped.where(mapped.ne("__OTHER__"), spec.labels[0])
        return pd.Series(mapped.to_numpy(), index=values.index, name=values.name)

    numeric = pd.to_numeric(values, errors="coerce")
    if spec.row_labels is not None:
        assigned = spec.row_labels.reindex(values.index)
        if assigned.notna().all():
            return pd.Series(assigned.astype(str).to_numpy(), index=values.index, name=values.name)

    if len(spec.labels) == 1 and spec.labels[0] != "__MISSING__":
        assigned = pd.Series(spec.labels[0], index=values.index, name=values.name)
    else:
        finite_edges = sorted(edge for edge in spec.edges if edge is not None)
        if finite_edges:
            bins = [-np.inf, *finite_edges]
            labels = spec.labels[: len(finite_edges)]
            assigned = pd.cut(
                numeric,
                bins=bins,
                labels=labels,
                include_lowest=True,
            ).astype("object")
        else:
            assigned = pd.Series(spec.labels[0], index=values.index, name=values.name)
    if "__MISSING__" in spec.labels:
        assigned = pd.Series(assigned, index=values.index).where(numeric.notna(), "__MISSING__")
    return pd.Series(assigned.astype(str).to_numpy(), index=values.index, name=values.name)


def _model_values(
    values: pd.Series,
    feature_type: FeatureType,
    bin_spec: _BinSpec,
    train_idx: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if feature_type == "continuous":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        train_values = numeric[train_idx]
        fill_value = float(np.nanmedian(train_values)) if not np.isnan(train_values).all() else 0.0
        return np.where(np.isnan(numeric), fill_value, numeric)

    labels = _assign_bins(values, bin_spec)
    mapping = {label: float(index) for index, label in enumerate(bin_spec.labels)}
    return labels.map(mapping).fillna(0.0).to_numpy(dtype=float)


def _encoded_feature_frame(X: pd.DataFrame, finder: ResidualSignalFinderV2) -> pd.DataFrame:
    encoded = {}
    for feature in X.columns:
        spec = _make_bin_spec(X[feature], finder)
        encoded[feature] = _model_values(
            X[feature],
            _infer_feature_type(X[feature]),
            spec,
            np.arange(len(X)),
        )
    return pd.DataFrame(encoded, index=X.index)


def _weights_for_fit(
    sample_weight: pd.Series | None,
    idx: np.ndarray[Any, Any],
    use_sample_weight: bool,
) -> np.ndarray[Any, Any] | None:
    if sample_weight is None or not use_sample_weight:
        return None
    return sample_weight.iloc[idx].to_numpy(dtype=float)


def _weights_for_score(
    sample_weight: pd.Series | None,
    idx: np.ndarray[Any, Any],
    use_sample_weight: bool,
) -> np.ndarray[Any, Any] | None:
    return _weights_for_fit(sample_weight, idx, use_sample_weight)


def _weighted_mean(values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any] | None) -> float:
    if weights is None:
        return float(np.mean(values))
    weight_sum = float(np.sum(weights))
    if weight_sum == 0.0:
        return float(np.mean(values))
    return float(np.sum(weights * values) / weight_sum)


def _oof_r2(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    null_prediction: float,
    sample_weight: np.ndarray[Any, Any] | None,
) -> float:
    if sample_weight is None:
        sse_model = float(np.sum(np.square(y_true - y_pred)))
        sse_null = float(np.sum(np.square(y_true - null_prediction)))
    else:
        sse_model = float(np.sum(sample_weight * np.square(y_true - y_pred)))
        sse_null = float(np.sum(sample_weight * np.square(y_true - null_prediction)))
    if sse_null <= 0.0:
        return 0.0
    return 1.0 - (sse_model / sse_null)


def _spearman(values: pd.Series, residuals: pd.Series, feature_type: FeatureType) -> float:
    if feature_type != "continuous":
        return np.nan
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & residuals.notna()
    if valid.sum() < 3 or numeric.loc[valid].nunique() <= 1:
        return np.nan
    value = numeric.loc[valid].corr(residuals.loc[valid], method="spearman")
    return float(value) if pd.notna(value) else np.nan


def _effect_curve_frame(
    *,
    feature_name: str,
    values: pd.Series,
    bin_spec: _BinSpec,
    residuals: pd.Series,
    actuals: pd.Series,
    base_pred: pd.Series,
    predicted_residual: np.ndarray[Any, Any],
    validation_idx: np.ndarray[Any, Any],
    bootstrap_id: int,
    split_id: int,
    split_role: Literal["validation", "holdout"],
    sample_weight: pd.Series | None,
    min_bin_count: int,
) -> pd.DataFrame:
    labels = _assign_bins(values, bin_spec).iloc[validation_idx]
    residual_validation = residuals.iloc[validation_idx]
    actual_validation = actuals.iloc[validation_idx]
    base_validation = base_pred.iloc[validation_idx]
    weights = (
        sample_weight.iloc[validation_idx].to_numpy(dtype=float)
        if sample_weight is not None
        else None
    )
    validation_mean = _weighted_mean(residual_validation.to_numpy(dtype=float), weights)
    rows = []
    for bin_id, label in enumerate(bin_spec.labels):
        mask = labels.astype(str).eq(label).to_numpy()
        bin_weights = weights[mask] if weights is not None else None
        bin_residual = residual_validation.to_numpy(dtype=float)[mask]
        bin_actual = actual_validation.to_numpy(dtype=float)[mask]
        bin_base = base_validation.to_numpy(dtype=float)[mask]
        bin_predicted_residual = predicted_residual[mask]
        n_obs = int(mask.sum())
        if n_obs == 0:
            continue
        mean_residual = _weighted_mean(bin_residual, bin_weights) if n_obs else np.nan
        predicted_mean = _weighted_mean(bin_predicted_residual, bin_weights) if n_obs else np.nan
        base_mean = _weighted_mean(bin_base, bin_weights) if n_obs else np.nan
        rows.append(
            {
                "feature": feature_name,
                "bootstrap_id": bootstrap_id,
                "split_id": split_id,
                "split_role": split_role,
                "bin_id": bin_id,
                "bin_label": label,
                "bin_left": _bin_left(bin_spec, bin_id),
                "bin_right": _bin_right(bin_spec, bin_id),
                "n_obs": n_obs,
                "reliable": n_obs >= min_bin_count,
                "actual_mean": _weighted_mean(bin_actual, bin_weights) if n_obs else np.nan,
                "base_prediction_mean": base_mean,
                "corrected_prediction_mean": base_mean + predicted_mean if n_obs else np.nan,
                "mean_residual": mean_residual,
                "centered_mean_residual": mean_residual - validation_mean if n_obs else np.nan,
                "predicted_residual": predicted_mean,
                "mean_abs_residual": _weighted_mean(np.abs(bin_residual), bin_weights)
                if n_obs
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _bin_left(spec: _BinSpec, bin_id: int) -> float | None:
    finite_edges = [edge for edge in spec.edges if edge is not None]
    if spec.feature_type == "continuous" and bin_id < len(finite_edges) - 1:
        return finite_edges[bin_id]
    return None


def _bin_right(spec: _BinSpec, bin_id: int) -> float | None:
    finite_edges = [edge for edge in spec.edges if edge is not None]
    if spec.feature_type == "continuous" and bin_id < len(finite_edges) - 1:
        return finite_edges[bin_id + 1]
    return None


def _rank_bootstrap_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    ranked = results.copy()
    ranked["rank"] = ranked.groupby(["bootstrap_id", "split_role"])["oof_r2"].rank(
        method="min",
        ascending=False,
    )
    ranked["is_top_5"] = ranked["rank"] <= 5
    ranked["is_top_10"] = ranked["rank"] <= 10
    return ranked


def _rank_null_results(results: pd.DataFrame, null_effect_curves: pd.DataFrame) -> pd.DataFrame:
    stability_columns = [
        "mean_effect_curve_spearman_stability",
        "median_effect_curve_spearman_stability",
        "std_effect_curve_spearman_stability",
    ]
    if results.empty:
        return pd.DataFrame(
            columns=[
                "bootstrap_id",
                "split_id",
                "split_role",
                "null_feature",
                "source_feature",
                "oof_r2",
                "rank",
                *stability_columns,
            ]
        )
    ranked = results.copy()
    ranked["rank"] = ranked.groupby(["bootstrap_id", "split_role"])["oof_r2"].rank(
        method="min",
        ascending=False,
    )
    ranked = ranked.rename(columns={"feature": "null_feature"})
    stability = _null_curve_stability(null_effect_curves)
    if not stability.empty:
        ranked = ranked.merge(stability, on="null_feature", how="left")
    for column in stability_columns:
        if column not in ranked.columns:
            ranked[column] = np.nan
    return ranked[
        [
            "bootstrap_id",
            "split_id",
            "split_role",
            "null_feature",
            "source_feature",
            "oof_r2",
            "rank",
            *stability_columns,
        ]
    ]


def _null_curve_stability(null_effect_curves: pd.DataFrame) -> pd.DataFrame:
    if null_effect_curves.empty:
        return pd.DataFrame()
    rows = []
    for null_feature in null_effect_curves["feature"].drop_duplicates().astype(str):
        stats = _effect_curve_stats(null_effect_curves, null_feature)
        rows.append({"null_feature": null_feature, **stats})
    return pd.DataFrame(rows)


def _add_null_comparison(results: pd.DataFrame, null_results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    compared = results.copy()
    if null_results.empty:
        compared["null_95_score"] = np.nan
        compared["beats_null_95"] = False
        return compared
    null_95 = null_results.groupby(["bootstrap_id", "split_role"])["oof_r2"].quantile(0.95)
    null_lookup = pd.MultiIndex.from_frame(compared[["bootstrap_id", "split_role"]])
    compared["null_95_score"] = null_95.reindex(null_lookup).to_numpy()
    compared["beats_null_95"] = compared["oof_r2"] > compared["null_95_score"]
    return compared


def _residual_summary(residuals: pd.Series) -> dict[str, float]:
    return {
        "mean_residual": float(residuals.mean()),
        "median_residual": float(residuals.median()),
        "std_residual": float(residuals.std(ddof=0)),
        "mae_residual": float(np.abs(residuals).mean()),
        "rmse_residual": float(np.sqrt(np.mean(np.square(residuals)))),
        "residual_skew": float(residuals.skew()),
        "residual_kurtosis": float(residuals.kurtosis()),
    }


def _add_data_warnings(
    warning_list: list[str],
    features: pd.DataFrame,
    residuals: pd.Series,
    finder: ResidualSignalFinderV2,
) -> None:
    if len(features) < 500:
        warning_list.append("n_obs < 500; residual signal stability may be noisy.")
    if finder.n_bootstraps < 50:
        warning_list.append("n_bootstraps < 50; stability summaries are directional, not final.")
    if residuals.std(ddof=0) == 0.0:
        warning_list.append("Residuals have zero variance; no residual signal can be measured.")
    for feature in features.columns:
        if features[feature].isna().mean() > 0.50:
            warning_list.append(
                f"{feature} has missing_rate > 0.50; review sparse/missingness signal."
            )


def _add_signal_warning(warning_list: list[str], summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    median_mean_r2 = float(summary["mean_oof_residual_r2"].median())
    if median_mean_r2 <= 0.0:
        warning_list.append(
            "Most features do not produce positive residual lift. Residuals may be mostly noise, "
            "candidate features may not contain signal, or the split procedure may be unstable."
        )


def _screening_candidates(screening_results: pd.DataFrame, top_k: int) -> list[str]:
    if screening_results.empty:
        return []
    return screening_results.head(top_k)["feature"].astype(str).tolist()


def _effect_curve_stats(
    effect_curves: pd.DataFrame,
    feature: str,
) -> dict[str, float]:
    correlations = _effect_curve_pairwise_spearman(effect_curves, feature)
    if correlations.empty:
        mean_stability = np.nan
        median_stability = np.nan
        std_stability = np.nan
    else:
        mean_stability = float(correlations.mean())
        median_stability = float(correlations.median())
        std_stability = float(correlations.std(ddof=0))
    return {
        "mean_effect_curve_spearman_stability": mean_stability,
        "median_effect_curve_spearman_stability": median_stability,
        "std_effect_curve_spearman_stability": std_stability,
    }


def _effect_curve_matrix(effect_curves: pd.DataFrame, feature: str) -> pd.DataFrame:
    curves = effect_curves.loc[effect_curves["feature"] == feature].copy()
    if "split_role" in curves.columns:
        curves = curves.loc[curves["split_role"].eq("validation")]
    if curves.empty:
        return pd.DataFrame()
    return curves.pivot(index="bootstrap_id", columns="bin_id", values="centered_mean_residual")


def _effect_curve_pairwise_spearman(effect_curves: pd.DataFrame, feature: str) -> pd.Series:
    pivot = _effect_curve_matrix(effect_curves, feature)
    if pivot.empty:
        return pd.Series(dtype=float)
    vectors = [row.to_numpy(dtype=float) for _idx, row in pivot.iterrows()]
    correlations = []
    for left, right in combinations(vectors, 2):
        corr = _spearman_array(left, right)
        if np.isfinite(corr):
            correlations.append(corr)
    return pd.Series(correlations, dtype=float, name="pairwise_spearman")


def _spearman_array(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return np.nan
    left_valid = pd.Series(left[valid])
    right_valid = pd.Series(right[valid])
    if left_valid.nunique() <= 1 or right_valid.nunique() <= 1:
        return np.nan
    value = left_valid.corr(right_valid, method="spearman")
    return float(value) if pd.notna(value) else np.nan


def _feature_curve_summary(effect_curves: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    feature_curves = effect_curves.loc[effect_curves["feature"] == feature_name]
    if "split_role" in feature_curves.columns:
        feature_curves = feature_curves.loc[feature_curves["split_role"].eq("validation")]
    if feature_curves.empty:
        raise ValueError(f"No effect curves are available for {feature_name}")
    rows = []
    for bin_id, group in feature_curves.groupby("bin_id", sort=True):
        rows.append(
            {
                "bin_id": bin_id,
                "bin_label": group["bin_label"].iloc[0],
                "n_obs": group["n_obs"].mean(),
                "actual_mean": group["actual_mean"].mean(),
                "actual_ci_low": group["actual_mean"].quantile(0.025),
                "actual_ci_high": group["actual_mean"].quantile(0.975),
                "base_prediction_mean": group["base_prediction_mean"].mean(),
                "base_prediction_ci_low": group["base_prediction_mean"].quantile(0.025),
                "base_prediction_ci_high": group["base_prediction_mean"].quantile(0.975),
                "corrected_prediction_mean": group["corrected_prediction_mean"].mean(),
                "residual_error_mean": group["mean_residual"].mean(),
                "residual_error_ci_low": group["mean_residual"].quantile(0.025),
                "residual_error_ci_high": group["mean_residual"].quantile(0.975),
                "centered_mean_residual": group["centered_mean_residual"].mean(),
                "centered_residual_p05": group["centered_mean_residual"].quantile(0.05),
                "centered_residual_p10": group["centered_mean_residual"].quantile(0.10),
                "centered_residual_p90": group["centered_mean_residual"].quantile(0.90),
                "centered_residual_p95": group["centered_mean_residual"].quantile(0.95),
                "predicted_residual": group["predicted_residual"].mean(),
                "predicted_residual_p10": group["predicted_residual"].quantile(0.10),
                "predicted_residual_p90": group["predicted_residual"].quantile(0.90),
            }
        )
    return pd.DataFrame(rows)


def _quantile(values: pd.Series, q: float) -> float:
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def _require_fitted(table: pd.DataFrame, attribute: str) -> None:
    if table.empty:
        raise ValueError(f"{attribute} is empty; call fit before requesting diagnostics")
