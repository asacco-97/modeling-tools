from __future__ import annotations

import base64
import html as _html
import io
import warnings
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
    random_state: int = 42
    max_categories: int = 20
    min_category_count: int = 30
    min_bin_count: int = 20
    classification_scatter_max_bin_size: int = 100
    use_sample_weight: bool = False
    strong_r2_threshold: float = 0.02
    model_params: dict[str, Any] | None = None
    subsample_max_train_size: int | None = None
    subsample_positive_class_target_perc: float = 0.30

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
        if self.subsample_max_train_size is not None and self.subsample_max_train_size < 1:
            raise ValueError("subsample_max_train_size must be at least 1")
        if not 0.0 < self.subsample_positive_class_target_perc < 1.0:
            raise ValueError("subsample_positive_class_target_perc must be between 0 and 1")

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
        runtime_warnings = _warn_large_dataset(len(features), len(features.columns), self)
        self.warnings_.extend(runtime_warnings)

        segment_values: pd.Series | None = None
        if segment_cols:
            missing_segments = [column for column in segment_cols if column not in features.columns]
            if missing_segments:
                raise ValueError(f"segment_cols not found in X: {missing_segments}")
            segment_values = _make_composite_segment(features, list(segment_cols))

        split_specs = self._make_splits(features, group_values, splits, segment_values)
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

        filtered_ec = _drop_degenerate_bins(self.effect_curves_, feature_name)
        curve = _feature_curve_summary(filtered_ec, feature_name)
        bins = curve["bin_label"].astype(str).tolist()
        x = np.arange(len(bins))
        figure = plt.figure(figsize=(17, 18))
        grid = figure.add_gridspec(
            4, 2,
            height_ratios=[1.3, 1.3, 0.35, 1.0],
            hspace=0.55,
            wspace=0.35,
        )
        actual_axis = figure.add_subplot(grid[0, :])
        boxplot_axis = figure.add_subplot(grid[1, :], sharex=actual_axis)
        count_axis = figure.add_subplot(grid[2, :], sharex=actual_axis)
        partial_axis = figure.add_subplot(grid[3, 0])
        stability_axis = figure.add_subplot(grid[3, 1])

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

        self._plot_residual_boxplot_on_axis(feature_name, boxplot_axis, bins, filtered_ec, curve)
        self._plot_bin_counts_on_axis(feature_name, count_axis, bins, x)
        self._plot_partial_residual_on_axis(feature_name, partial_axis, curve, filtered_ec)
        self._plot_bootstrap_r2_distribution_on_axis(feature_name, stability_axis)

        actual_axis.set_xticks(x)
        actual_axis.set_xticklabels([])
        boxplot_axis.set_xticks(x)
        boxplot_axis.set_xticklabels([])
        count_axis.set_xticks(x)
        count_axis.set_xticklabels(bins)
        count_axis.tick_params(axis="x", labelsize=9)
        plt.setp(count_axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
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

    def to_html(
        self,
        path: str | None = None,
        title: str | None = None,
        top_n: int = 10,
    ) -> str:
        """Render a self-contained HTML report and optionally write it to disk.

        Args:
            path: Optional file path to write the HTML. If None, returns the string only.
            title: Report title. Defaults to "Residual Signal Report".
            top_n: Number of top features to include in the ranking table and detail cards.

        Returns:
            The full HTML string.
        """
        from datetime import datetime

        import matplotlib.pyplot as plt

        _require_fitted(self.summary_, "summary_")
        effective_title = title or "Residual Signal Report"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        summary = self.summary_.copy()
        res_summary = getattr(self, "residual_summary_", {})
        exec_items: list[tuple[str, str]] = [
            ("Observations", f"{res_summary.get('n_obs', '—'):,}" if "n_obs" in res_summary else "—"),
            ("Residual mean", f"{res_summary.get('mean', float('nan')):.4f}" if "mean" in res_summary else "—"),
            ("Residual std", f"{res_summary.get('std', float('nan')):.4f}" if "std" in res_summary else "—"),
            ("Features screened", str(len(summary))),
            ("Bootstrap runs", str(getattr(self, "n_bootstraps", "—"))),
        ]

        ranking_cols = [
            c for c in [
                "feature",
                "feature_type",
                "mean_oof_residual_r2",
                "null_beat_rate",
                "median_effect_curve_spearman_stability",
                "mean_oof_abs_residual_r2",
                "pct_positive_feature_residual_spearman",
            ]
            if c in summary.columns
        ]
        top_summary = summary.head(top_n)[ranking_cols].copy()
        for col in [
            "mean_oof_residual_r2",
            "null_beat_rate",
            "median_effect_curve_spearman_stability",
            "mean_oof_abs_residual_r2",
            "pct_positive_feature_residual_spearman",
        ]:
            if col in top_summary.columns and col != "null_beat_rate":
                top_summary[col] = top_summary[col].map(
                    lambda v: f"{v:.3f}" if pd.notna(v) else ""
                )
            elif col in top_summary.columns:
                top_summary[col] = top_summary[col].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else ""
                )

        feature_figures: list[tuple[str, str | None]] = []
        top_features = list(summary.head(top_n)["feature"]) if "feature" in summary.columns else []
        for feature_name in top_features:
            try:
                fig = self.plot_feature_diagnostics(feature_name)
                img_b64 = _figure_to_base64(fig)
                plt.close(fig)
            except Exception:
                img_b64 = None
            feature_figures.append((feature_name, img_b64))

        warning_messages = list(getattr(self, "warnings_", []))

        html_str = _build_html_report(
            title=effective_title,
            generated_at=generated_at,
            exec_items=exec_items,
            ranking_df=top_summary,
            feature_figures=feature_figures,
            top_n=top_n,
            warning_messages=warning_messages,
        )

        if path is not None:
            import pathlib
            pathlib.Path(path).write_text(html_str, encoding="utf-8")

        return html_str

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
        segment_values: pd.Series | None = None,
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
            if segment_values is not None:
                train_idx, validation_idx = _stratified_bootstrap_split(
                    positions, segment_values, train_size, rng
                )
            else:
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
        """Create one permuted null per candidate feature."""
        if features.empty:
            return []
        rng = np.random.default_rng(self.random_state + 100_003)
        candidate_features = list(features.columns)
        null_features_list = []
        for source_feature in candidate_features:
            name = f"__null_{source_feature}"
            if self.null_strategy == "random_noise":
                values = pd.Series(rng.normal(size=len(features)), index=features.index, name=name)
                null_residuals = residuals
            elif self.null_strategy == "shuffled_residuals":
                values = features[source_feature].rename(name)
                null_residuals = pd.Series(
                    rng.permutation(residuals.to_numpy()),
                    index=residuals.index,
                    name="shuffled_residual",
                )
            else:
                values = pd.Series(
                    rng.permutation(features[source_feature].to_numpy()),
                    index=features.index,
                    name=name,
                )
                null_residuals = residuals
            null_features_list.append((name, source_feature, values, null_residuals))
        return null_features_list

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

        # Optionally subsample training rows for large datasets
        effective_train_idx = train_idx
        if (
            self.subsample_max_train_size is not None
            and len(train_idx) > self.subsample_max_train_size
        ):
            subsample_rng = np.random.default_rng(self.random_state + split_id + 777_777)
            effective_train_idx = _subsample_train_idx(
                train_idx,
                actuals,
                self.subsample_max_train_size,
                self.subsample_positive_class_target_perc,
                subsample_rng,
            )

        model_values = _model_values(values, feature_type, bin_spec, effective_train_idx)
        train_X = pd.DataFrame({"feature": model_values[effective_train_idx]})
        validation_X = pd.DataFrame({"feature": model_values[evaluation_idx]})
        train_y = residuals.iloc[effective_train_idx]
        validation_y = residuals.iloc[evaluation_idx]
        weights_train = _weights_for_fit(sample_weight, effective_train_idx, self.use_sample_weight)
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

        # Variance signal: model on |residuals|
        abs_train_y = pd.Series(
            np.abs(train_y.to_numpy()), index=train_y.index, name="abs_residual"
        )
        abs_validation_y = np.abs(validation_y.to_numpy())
        abs_model = self._make_model(
            self.univariate_model_type, seed=self.random_state + split_id + 999_983
        )
        abs_model.fit(train_X, abs_train_y, sample_weight=weights_train)
        abs_predicted = np.asarray(abs_model.predict(validation_X), dtype=float)
        abs_train_mean = _weighted_mean(abs_train_y.to_numpy(), weights_train)
        abs_oof_r2 = _oof_r2(abs_validation_y, abs_predicted, abs_train_mean, weights_validation)

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
                "abs_oof_r2": abs_oof_r2,
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
            abs_r2_values = group["abs_oof_r2"].dropna() if "abs_oof_r2" in group.columns else pd.Series(dtype=float)
            beats_null_values = group["beats_null"].dropna() if "beats_null" in group.columns else pd.Series(dtype=float)
            oof_r2_vals = group["oof_r2"].dropna()
            prob_gt0 = float((oof_r2_vals > 0.0).mean()) if len(oof_r2_vals) else np.nan
            prob_abs_gt0 = float((abs_r2_values > 0.0).mean()) if len(abs_r2_values) else np.nan
            feature_str = str(feature)
            bin_spec = self._bin_specs.get(feature_str)
            bin_stats: dict[str, Any] = (
                _bin_count_stats(self._X[feature_str], bin_spec)
                if bin_spec is not None
                else {"min_bin_n": np.nan, "max_bin_share": np.nan, "sparse_bin_warning": False}
            )
            try:
                curve_for_shape = _feature_curve_summary(validation_curves, feature_str)
                shape = _classify_residual_shape(
                    curve_for_shape,
                    stability=curve_stats["median_effect_curve_spearman_stability"],
                )
            except (ValueError, KeyError):
                shape = "unstable"
            nbr_for_rec = float(beats_null_values.mean()) if len(beats_null_values) else np.nan
            category, recommendation = _action_recommendation(
                mean_oof_r2=float(group["oof_r2"].mean()),
                prob_signal_gt_zero=prob_gt0,
                null_beat_rate=nbr_for_rec,
                stability=curve_stats["median_effect_curve_spearman_stability"],
                sparse_bin_warning=bool(bin_stats["sparse_bin_warning"]),
                shape_class=shape,
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
                    "mean_oof_abs_residual_r2": float(abs_r2_values.mean()) if len(abs_r2_values) else np.nan,
                    "median_oof_abs_residual_r2": float(abs_r2_values.median()) if len(abs_r2_values) else np.nan,
                    "p05_oof_abs_residual_r2": _quantile(abs_r2_values, 0.05),
                    "p95_oof_abs_residual_r2": _quantile(abs_r2_values, 0.95),
                    "null_beat_rate": float(beats_null_values.mean()) if len(beats_null_values) else np.nan,
                    "prob_residual_signal_gt_zero": prob_gt0,
                    "prob_abs_residual_signal_gt_zero": prob_abs_gt0,
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
                    "min_bin_n": bin_stats["min_bin_n"],
                    "max_bin_share": bin_stats["max_bin_share"],
                    "sparse_bin_warning": bin_stats["sparse_bin_warning"],
                    "residual_shape_class": shape,
                    "action_category": category,
                    "action_recommendation": recommendation,
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
                "bin_label": labels.astype(str),
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

    def _plot_residual_boxplot_on_axis(
        self,
        feature_name: str,
        axis: Any,
        retained_bin_labels: list[str],
        filtered_ec: pd.DataFrame,
        curve: pd.DataFrame,
    ) -> None:
        # Use bootstrap validation mean_residual per bin — consistent with Actual vs Pred panel
        feature_curves = filtered_ec.loc[filtered_ec["feature"].eq(feature_name)]
        if "split_role" in feature_curves.columns:
            feature_curves = feature_curves.loc[feature_curves["split_role"].eq("validation")]
        label_to_id = {str(row["bin_label"]): row["bin_id"] for _, row in curve.iterrows()}
        data_by_bin = []
        for label in retained_bin_labels:
            bin_id = label_to_id.get(label)
            if bin_id is not None:
                vals = feature_curves.loc[
                    feature_curves["bin_id"].eq(bin_id), "mean_residual"
                ].dropna().to_numpy(dtype=float)
            else:
                vals = np.array([], dtype=float)
            data_by_bin.append(vals)
        x = np.arange(len(retained_bin_labels))
        non_empty = [arr for arr in data_by_bin if len(arr) > 0]
        if not non_empty:
            axis.text(0.5, 0.5, "No data available", ha="center", va="center")
            axis.set_title("Residual Distribution by Feature Bin")
            return
        axis.boxplot(
            data_by_bin,
            positions=x,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#e05c00", "linewidth": 2.0},
            boxprops={"facecolor": "#d0e4f5", "alpha": 0.75},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        means = [float(arr.mean()) if len(arr) else np.nan for arr in data_by_bin]
        axis.scatter(x, means, color="#2a6a94", zorder=5, s=30, marker="D", label="mean")
        axis.axhline(0.0, linestyle=":", color="black", linewidth=1.0)
        axis.set_title("Residual Distribution by Feature Bin (Bootstrap Validation)")
        axis.set_ylabel("Mean residual per bootstrap run")
        axis.legend(loc="best", fontsize=8)

    def _plot_bin_counts_on_axis(
        self,
        feature_name: str,
        axis: Any,
        bin_labels: list[str],
        x: np.ndarray[Any, Any],
    ) -> None:
        bin_frame = self._full_bin_frame(feature_name)
        bin_str = bin_frame["bin_label"].astype(str)
        counts = [int(bin_str.eq(label).sum()) for label in bin_labels]
        axis.bar(x, counts, color="#8cb8d4", alpha=0.75, width=0.7)
        axis.set_ylabel("n", fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
        axis.set_title("Sample Count per Bin", fontsize=9)
        if len(bin_labels) <= 8:
            for xi, n in zip(x, counts):
                axis.text(float(xi), n * 0.5, str(n), ha="center", va="center", fontsize=7)

    def _plot_partial_residual_on_axis(
        self,
        feature_name: str,
        axis: Any,
        curve: pd.DataFrame,
        filtered_ec: pd.DataFrame,
    ) -> None:
        import matplotlib.pyplot as plt

        feature_type = _infer_feature_type(self._X[feature_name])
        residuals = self._residuals

        # Smooth line uses bootstrap validation means from the effect curve (same source as
        # Actual vs Pred panel) so all bin-level panels are directionally consistent.
        smooth_y = curve["residual_error_mean"].to_numpy(dtype=float)
        smooth_ci_low = curve["residual_error_ci_low"].to_numpy(dtype=float)
        smooth_ci_high = curve["residual_error_ci_high"].to_numpy(dtype=float)

        if feature_type == "continuous":
            numeric = pd.to_numeric(self._X[feature_name], errors="coerce")
            valid_mask = numeric.notna() & residuals.notna()
            # Scatter: full-dataset raw observations for shape context
            x_all = numeric.loc[valid_mask].to_numpy(dtype=float)
            y_all = residuals.loc[valid_mask].to_numpy(dtype=float)
            max_scatter = 2000
            if len(x_all) > max_scatter:
                rng_idx = np.random.default_rng(42).choice(len(x_all), size=max_scatter, replace=False)
                x_scatter = x_all[rng_idx]
                y_scatter = y_all[rng_idx]
            else:
                x_scatter, y_scatter = x_all, y_all
            axis.scatter(x_scatter, y_scatter, s=8, alpha=0.15, color="#4c78a8", edgecolors="none")
            # Smooth: compute bin medians for x positions, use bootstrap validation y means
            bin_spec = self._bin_specs.get(feature_name)
            if bin_spec is not None:
                assigned = _assign_bins(self._X[feature_name], bin_spec).astype(str)
                bin_labels = curve["bin_label"].astype(str).tolist()
                x_mids: list[float] = []
                for label in bin_labels:
                    mask = assigned.eq(label) & valid_mask
                    x_mids.append(float(numeric.loc[mask].median()) if mask.any() else np.nan)
                valid_smooth = [
                    (xm, ym, yl, yh)
                    for xm, ym, yl, yh in zip(x_mids, smooth_y, smooth_ci_low, smooth_ci_high)
                    if np.isfinite(xm) and np.isfinite(ym)
                ]
                if len(valid_smooth) >= 2:
                    xs, ys, yls, yhs = zip(*sorted(valid_smooth, key=lambda t: t[0]))
                    xs_arr = np.array(xs)
                    axis.fill_between(xs_arr, list(yls), list(yhs), color="#e05c00", alpha=0.20)
                    axis.plot(
                        xs_arr, list(ys),
                        color="#e05c00", linewidth=2.0, marker="o", markersize=4,
                        label="bin mean (validation)",
                    )
                    axis.legend(loc="best", fontsize=8)
            axis.set_xlabel(feature_name)
            axis.set_ylabel("Residual (y − base prediction)")
        else:
            x_cat = np.arange(len(curve))
            axis.bar(x_cat, smooth_y, color="#4c78a8", alpha=0.75)
            axis.fill_between(
                x_cat, smooth_ci_low, smooth_ci_high, color="#4c78a8", alpha=0.18
            )
            axis.set_xticks(x_cat)
            axis.set_xticklabels(curve["bin_label"].astype(str).tolist())
            plt.setp(
                axis.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=8,
            )
            axis.set_xlabel(feature_name)
            axis.set_ylabel("Mean Residual (y − base prediction)")
        axis.axhline(0.0, linestyle=":", color="black", linewidth=1.0)
        axis.set_title(
            "Partial Residual Plot\n(positive = underprediction, negative = overprediction)",
            fontsize=9,
        )

    def _plot_bootstrap_r2_distribution_on_axis(self, feature_name: str, axis: Any) -> None:
        validation_mask = self.bootstrap_results_["split_role"].eq("validation")
        feature_mask = self.bootstrap_results_["feature"].eq(feature_name)
        oof_r2_vals = self.bootstrap_results_.loc[
            validation_mask & feature_mask, "oof_r2"
        ].dropna()
        if oof_r2_vals.empty:
            axis.text(0.5, 0.5, "No bootstrap data", ha="center", va="center")
            axis.axis("off")
            return
        axis.hist(oof_r2_vals, bins=15, color="#4c78a8", alpha=0.75)
        mean_r2 = float(oof_r2_vals.mean())
        axis.axvline(mean_r2, color="black", linewidth=2.0, label=f"mean: {mean_r2:.4f}")
        axis.axvline(0.0, linestyle=":", color="red", linewidth=1.2, label="zero")
        prob_gt0 = float((oof_r2_vals > 0.0).mean())
        axis.text(
            0.97,
            0.95,
            f"P(R²>0): {prob_gt0:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#2a6a94",
        )
        axis.set_xlabel("Bootstrap OOF residual R²")
        axis.set_ylabel("Bootstrap run count")
        axis.set_title("Bootstrap Residual R² Distribution")
        axis.legend(loc="upper left", fontsize=8)

    def _plot_null_comparison_on_axis(self, feature_name: str, axis: Any) -> None:
        validation_mask = self.bootstrap_results_["split_role"].eq("validation")
        feature_mask = self.bootstrap_results_["feature"].eq(feature_name)
        real_rows = self.bootstrap_results_.loc[validation_mask & feature_mask]
        real_scores = real_rows["oof_r2"]

        beats_col = "beats_null" if "beats_null" in real_rows.columns else None
        beat_rate = float(real_rows[beats_col].mean()) if beats_col and len(real_rows) else np.nan

        if np.isfinite(beat_rate):
            axis.text(
                0.5, 0.82,
                f"{beat_rate:.0%}",
                transform=axis.transAxes,
                ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="#4a7fa5",
            )
            axis.text(
                0.5, 0.70,
                "of bootstrap runs beat the null",
                transform=axis.transAxes,
                ha="center", va="center",
                fontsize=9, color="#555555",
            )

        null_scores: pd.Series = pd.Series(dtype=float)
        if not self.null_results_.empty and "source_feature" in self.null_results_.columns:
            null_scores = self.null_results_.loc[
                self.null_results_["source_feature"].eq(feature_name), "oof_r2"
            ]

        y_real = np.zeros(len(real_scores)) - 0.06
        axis.scatter(real_scores, y_real, alpha=0.50, color="#4a7fa5", s=18, zorder=3, label=feature_name)
        if not null_scores.empty:
            y_null = np.zeros(len(null_scores)) - 0.14
            axis.scatter(null_scores, y_null, alpha=0.45, color="#aaaaaa", s=18, zorder=2, label="null")
            axis.axvline(float(null_scores.median()), linestyle="--", color="#888", linewidth=1.2, label="null p50")
            axis.axvline(float(null_scores.quantile(0.95)), linestyle=":", color="#333", linewidth=1.2, label="null p95")
        if len(real_scores):
            axis.axvline(float(real_scores.mean()), linestyle="-", color="#4a7fa5", linewidth=1.5, label="real mean")
        axis.set_yticks([])
        axis.set_title(f"Null Baseline Comparison: {feature_name}")
        axis.set_xlabel("OOF residual R²")
        axis.legend(loc="lower right", fontsize=8)

    def _plot_effect_curve_stability_on_axis(
        self,
        feature_name: str,
        axis: Any,
        effect_curves: pd.DataFrame | None = None,
    ) -> None:
        ec = effect_curves if effect_curves is not None else self.effect_curves_
        curves = ec.loc[ec["feature"] == feature_name]
        if curves.empty:
            axis.text(0.5, 0.5, "No effect curves available", ha="center", va="center")
            axis.axis("off")
            return

        curve_summary = _feature_curve_summary(ec, feature_name)
        x = np.arange(len(curve_summary))
        reliable = curve_summary["pct_reliable"].to_numpy(dtype=float) >= 0.50

        # Continuous CI band; mask values for unreliable bins so they don't appear
        p025 = curve_summary["centered_residual_p025"].to_numpy(dtype=float).copy()
        p975 = curve_summary["centered_residual_p975"].to_numpy(dtype=float).copy()
        p025[~reliable] = np.nan
        p975[~reliable] = np.nan
        axis.fill_between(x, p025, p975, color="#4a7fa5", alpha=0.22, label="95% CI")

        axis.plot(
            x,
            curve_summary["centered_mean_residual"].to_numpy(dtype=float),
            color="#4a7fa5",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="mean",
        )
        axis.axhline(0.0, linestyle=":", color="black", linewidth=1.0)
        axis.set_title("Prediction Error by Feature Bin (Bootstrap 95% CI)")
        axis.set_ylabel("Centered mean residual")
        axis.legend(loc="best")

    def _plot_effect_curve_correlation_on_axis(self, feature_name: str, axis: Any) -> None:
        correlations = _effect_curve_pairwise_spearman(self.effect_curves_, feature_name)
        axis.set_title(
            f"{feature_name}: Spearman Rank Correlation Between Effect Curves\n"
            "(consistency of residual effect across bootstrap samples)",
            fontsize=10,
        )
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
        axis.axvline(
            mean_value,
            color="black",
            linewidth=2.0,
            label=f"mean Spearman: {mean_value:.2f}",
        )
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
        return left_label
    return f"{left_label}–{right_label}"


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
    import matplotlib.pyplot as plt

    axis.set_xticks(ticks["x_value"])
    axis.set_xticklabels(ticks["x_label"])
    axis.tick_params(axis="x", labelsize=8)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


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
    """Match each real feature to its specific permuted null by source_feature + split."""
    if results.empty:
        return results
    compared = results.copy()
    if null_results.empty or "source_feature" not in null_results.columns:
        compared["null_oof_r2"] = np.nan
        compared["beats_null"] = np.nan
        return compared
    null_lookup = (
        null_results[["source_feature", "split_id", "split_role", "oof_r2"]]
        .rename(columns={"source_feature": "feature", "oof_r2": "null_oof_r2"})
    )
    compared = compared.merge(null_lookup, on=["feature", "split_id", "split_role"], how="left")
    has_null = compared["null_oof_r2"].notna()
    compared["beats_null"] = np.where(
        has_null, compared["oof_r2"] > compared["null_oof_r2"], np.nan
    )
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
    feature_curves = effect_curves.loc[effect_curves["feature"] == feature].copy()
    if "split_role" in feature_curves.columns:
        feature_curves = feature_curves.loc[feature_curves["split_role"].eq("validation")]
    if feature_curves.empty:
        return pd.Series(dtype=float)
    # Keep only bins that are reliable in at least 50% of bootstrap runs
    if "reliable" in feature_curves.columns:
        bin_reliability = feature_curves.groupby("bin_id")["reliable"].mean()
        reliable_bin_ids = set(bin_reliability.index[bin_reliability >= 0.5])
        feature_curves = feature_curves[feature_curves["bin_id"].isin(reliable_bin_ids)]
    if feature_curves.empty or feature_curves["bin_id"].nunique() < 3:
        return pd.Series(dtype=float)
    pivot = feature_curves.pivot(
        index="bootstrap_id", columns="bin_id", values="centered_mean_residual"
    )
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


def _drop_degenerate_bins(effect_curves: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """Remove point-mass bins (bin_left == bin_right) for a feature from effect_curves.

    When a continuous feature has many repeated values the quantile-based binning
    can create bins whose left and right boundary are identical — these add no
    shape information to the effect curve.  Removing them keeps both panels
    aligned and avoids a flat segment that spans a single value.
    """
    if "bin_left" not in effect_curves.columns or "bin_right" not in effect_curves.columns:
        return effect_curves
    feature_mask = effect_curves["feature"] == feature_name
    feature_ec = effect_curves.loc[feature_mask]
    if feature_ec.empty:
        return effect_curves
    # A bin is degenerate when bin_left == bin_right (both non-NaN) across >50% of runs
    bin_degenerate = feature_ec.groupby("bin_id")[["bin_left", "bin_right"]].apply(
        lambda g: (
            g["bin_left"].notna() & g["bin_right"].notna() & (g["bin_left"] == g["bin_right"])
        ).mean()
        > 0.50
    )
    degenerate_ids = set(bin_degenerate.index[bin_degenerate])
    if not degenerate_ids:
        return effect_curves
    drop_mask = feature_mask & effect_curves["bin_id"].isin(degenerate_ids)
    return effect_curves.loc[~drop_mask].reset_index(drop=True)


def _bin_count_stats(
    feature_values: pd.Series, bin_spec: _BinSpec
) -> dict[str, Any]:
    labels = _assign_bins(feature_values, bin_spec).astype(str)
    counts = labels.value_counts()
    non_missing = counts.drop("__MISSING__", errors="ignore")
    if non_missing.empty:
        return {"min_bin_n": 0, "max_bin_share": 1.0, "sparse_bin_warning": True}
    total = int(non_missing.sum())
    min_n = int(non_missing.min())
    max_share = float(non_missing.max() / total) if total > 0 else 1.0
    return {
        "min_bin_n": min_n,
        "max_bin_share": max_share,
        "sparse_bin_warning": min_n < 30 or max_share > 0.50,
    }


def _classify_residual_shape(
    curve: pd.DataFrame,
    stability: float,
    stability_threshold: float = 0.40,
    flat_threshold: float = 0.001,
) -> str:
    if not np.isfinite(stability) or stability < stability_threshold:
        return "unstable"
    centered = curve["centered_mean_residual"].to_numpy(dtype=float)
    finite = centered[np.isfinite(centered)]
    if len(finite) < 3:
        return "unstable"
    overall_range = float(np.max(finite) - np.min(finite))
    if overall_range < flat_threshold:
        return "flat_or_noisy"
    diffs = np.diff(finite)
    pct_positive = float(np.mean(diffs > 0))
    pct_negative = float(np.mean(diffs < 0))
    n = len(finite)
    mid_start = n // 4
    mid_end = 3 * n // 4
    midpoint = float(
        np.mean(finite[mid_start : mid_end + 1]) if mid_start < mid_end else finite[n // 2]
    )
    edge_mean = (finite[0] + finite[-1]) / 2.0
    mid_vs_edge = midpoint - edge_mean
    if pct_positive >= 0.75:
        return "monotonic_increasing"
    if pct_negative >= 0.75:
        return "monotonic_decreasing"
    if mid_vs_edge < -overall_range * 0.30:
        return "u_shaped"
    if mid_vs_edge > overall_range * 0.30:
        return "inverted_u_shaped"
    return "nonlinear_or_threshold"


def _action_recommendation(
    *,
    mean_oof_r2: float,
    prob_signal_gt_zero: float,
    null_beat_rate: float,
    stability: float,
    sparse_bin_warning: bool,
    shape_class: str,
    strong_r2_threshold: float = 0.01,
    moderate_r2_threshold: float = 0.005,
    prob_signal_threshold: float = 0.90,
    moderate_prob_threshold: float = 0.75,
    stability_threshold: float = 0.60,
    moderate_stability_threshold: float = 0.40,
    null_beat_threshold: float = 0.80,
) -> tuple[str, str]:
    def _s(v: float) -> float:
        return float(v) if np.isfinite(v) else 0.0

    r2 = _s(mean_oof_r2)
    prob = _s(prob_signal_gt_zero)
    nbr = _s(null_beat_rate)
    stab = _s(stability)
    strong = (
        r2 >= strong_r2_threshold
        and prob >= prob_signal_threshold
        and nbr >= null_beat_threshold
        and stab >= stability_threshold
    )
    moderate = not strong and (
        r2 >= moderate_r2_threshold
        and prob >= moderate_prob_threshold
        and stab >= moderate_stability_threshold
    )
    if strong:
        category = "strong"
        base_text = (
            "Strong candidate for re-specification. The feature shows stable residual signal "
            "and should be tested with a more flexible form."
        )
    elif moderate:
        category = "moderate"
        base_text = (
            "Moderate residual signal. Investigate this feature, but validate improvement "
            "before prioritizing production changes."
        )
    else:
        category = "weak"
        base_text = (
            "Weak or unstable residual signal. Do not prioritize this feature unless there "
            "is strong business rationale."
        )
    sparse_note = (
        " Residual pattern may be driven by sparse bins — validate with larger samples "
        "or grouped bins before acting."
        if sparse_bin_warning
        else ""
    )
    return category, base_text + sparse_note


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
                "pct_reliable": float(group["reliable"].mean()) if "reliable" in group.columns else 1.0,
                "centered_mean_residual": group["centered_mean_residual"].mean(),
                "centered_residual_p025": group["centered_mean_residual"].quantile(0.025),
                "centered_residual_p05": group["centered_mean_residual"].quantile(0.05),
                "centered_residual_p10": group["centered_mean_residual"].quantile(0.10),
                "centered_residual_p90": group["centered_mean_residual"].quantile(0.90),
                "centered_residual_p95": group["centered_mean_residual"].quantile(0.95),
                "centered_residual_p975": group["centered_mean_residual"].quantile(0.975),
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


def _warn_large_dataset(
    n_obs: int,
    n_features: int,
    finder: ResidualSignalFinderV2,
) -> list[str]:
    messages = []
    if n_obs > 100_000:
        suggested_bootstraps = max(30, min(finder.n_bootstraps, 5_000_000 // n_obs))
        if suggested_bootstraps < finder.n_bootstraps:
            messages.append(
                f"Large dataset detected (n={n_obs:,}). With {n_obs:,} observations each "
                f"validation fold has ~{int(n_obs * finder.test_size):,} rows, providing "
                f"stable OOF estimates. Consider reducing n_bootstraps from "
                f"{finder.n_bootstraps} to {suggested_bootstraps} to reduce runtime."
            )
        if finder.subsample_max_train_size is None:
            messages.append(
                f"Large dataset (n={n_obs:,}): set subsample_max_train_size to cap per-split "
                f"training size for faster univariate fits without losing stability."
            )
    if n_features > 30:
        messages.append(
            f"Many features ({n_features}): screening_enabled=True with screening_top_k="
            f"{finder.screening_top_k} is recommended to reduce the candidate set before "
            f"the univariate loop."
        )
    for msg in messages:
        warnings.warn(msg, UserWarning, stacklevel=4)
    return messages


def _subsample_train_idx(
    train_idx: np.ndarray[Any, Any],
    actuals: pd.Series,
    max_size: int,
    positive_class_target_perc: float,
    rng: np.random.Generator,
) -> np.ndarray[Any, Any]:
    """Subsample training indices, upsampling non-zeros for zero-inflated targets."""
    if len(train_idx) <= max_size:
        return train_idx
    train_y = actuals.iloc[train_idx].to_numpy(dtype=float)
    zero_rate = float(np.mean(train_y == 0.0))
    if zero_rate > 0.80:
        zero_positions = train_idx[train_y == 0.0]
        nonzero_positions = train_idx[train_y != 0.0]
        n_nonzero_target = max(1, int(round(max_size * positive_class_target_perc)))
        n_zero_target = max(0, max_size - n_nonzero_target)
        sampled_zeros = rng.choice(
            zero_positions, size=min(n_zero_target, len(zero_positions)), replace=False
        )
        replace_nonzero = len(nonzero_positions) < n_nonzero_target
        sampled_nonzeros = rng.choice(
            nonzero_positions, size=n_nonzero_target, replace=replace_nonzero
        )
        return np.sort(np.concatenate([sampled_zeros, sampled_nonzeros]))
    return np.sort(rng.choice(train_idx, size=max_size, replace=False))


def _make_composite_segment(features: pd.DataFrame, segment_cols: list[str]) -> pd.Series:
    """Combine multiple segment columns into a single composite key."""
    if len(segment_cols) == 1:
        return features[segment_cols[0]].astype(str)
    return features[segment_cols].astype(str).agg("__".join, axis=1)


def _stratified_bootstrap_split(
    positions: np.ndarray[Any, Any],
    segment_values: pd.Series,
    train_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Bootstrap split that preserves segment proportions."""
    seg_array = segment_values.to_numpy()
    unique_segments = np.unique(seg_array)
    train_parts: list[np.ndarray[Any, Any]] = []
    val_parts: list[np.ndarray[Any, Any]] = []
    for segment in unique_segments:
        seg_positions = positions[seg_array == segment]
        seg_n = len(seg_positions)
        seg_train_size = max(1, int(round(train_size * seg_n / len(positions))))
        seg_train = rng.choice(seg_positions, size=min(seg_train_size, seg_n), replace=False)
        seg_val = np.setdiff1d(seg_positions, seg_train)
        train_parts.append(seg_train)
        if len(seg_val) > 0:
            val_parts.append(seg_val)
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def _figure_to_base64(fig: Any) -> str:
    """Serialize a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_html_report(
    title: str,
    generated_at: str,
    exec_items: list[tuple[str, str]],
    ranking_df: pd.DataFrame,
    feature_figures: list[tuple[str, str | None]],
    top_n: int,
    warning_messages: list[str],
) -> str:
    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 14px; line-height: 1.65; color: #222; background: #fff;
      max-width: 1100px; margin: 0 auto; padding: 44px 36px;
    }
    .report-header { border-top: 3px solid #4a7fa5; padding-top: 22px; margin-bottom: 40px; }
    h1 { font-size: 22px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }
    .meta { font-size: 12px; color: #999; }
    h2 {
      font-size: 12px; font-weight: 700; color: #4a7fa5;
      text-transform: uppercase; letter-spacing: 0.07em;
      margin-top: 44px; margin-bottom: 16px;
      padding-bottom: 7px; border-bottom: 1px solid #e4e4e4;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      text-align: left; font-weight: 600; color: #666;
      padding: 9px 13px; border-bottom: 2px solid #e4e4e4; white-space: nowrap;
    }
    td { padding: 8px 13px; border-bottom: 1px solid #f2f2f2; vertical-align: top; }
    tr:nth-child(even) td { background: #f9fafb; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 3px;
      font-size: 11px; font-weight: 700; letter-spacing: 0.02em;
    }
    .badge-strong { background: #ddeef7; color: #2a6a94; }
    .badge-moderate { background: #e6f4e6; color: #3a7a3a; }
    .badge-weak { background: #f0f0f0; color: #999; }
    .feature-card { margin-top: 36px; padding-top: 24px; border-top: 1px solid #e8e8e8; }
    .feature-card h3 { font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 12px; }
    .feature-card img { max-width: 100%; height: auto; }
    .warnings {
      background: #fffbf0; border-left: 3px solid #f0a500;
      padding: 14px 18px; margin-top: 36px; font-size: 13px; color: #555;
    }
    .warnings ul { margin: 8px 0 0 18px; }
    .warnings li { margin-bottom: 4px; }
    """

    exec_rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td class='num'>{_html.escape(v)}</td></tr>"
        for k, v in exec_items
    )

    col_labels = {
        "feature": "Feature",
        "feature_type": "Type",
        "mean_oof_residual_r2": "Mean OOF R²",
        "null_beat_rate": "Null Beat Rate",
        "median_effect_curve_spearman_stability": "Curve Stability",
        "mean_oof_abs_residual_r2": "Variance Signal R²",
        "pct_positive_feature_residual_spearman": "% Positive Spearman",
    }
    header_cells = "".join(
        f"<th>{col_labels.get(c, c)}</th>" for c in ranking_df.columns
    )
    ranking_rows = []
    for _, row in ranking_df.iterrows():
        cells = []
        for col in ranking_df.columns:
            val = str(row[col]) if pd.notna(row[col]) else ""
            if col == "null_beat_rate" and val:
                try:
                    rate = float(val)
                    if rate > 0.75:
                        badge_cls = "badge-strong"
                    elif rate >= 0.50:
                        badge_cls = "badge-moderate"
                    else:
                        badge_cls = "badge-weak"
                    val = f"<span class='badge {badge_cls}'>{rate:.0%}</span>"
                    cells.append(f"<td>{val}</td>")
                    continue
                except ValueError:
                    pass
            numeric_cols = {
                "mean_oof_residual_r2", "median_effect_curve_spearman_stability",
                "mean_oof_abs_residual_r2", "pct_positive_feature_residual_spearman",
            }
            td_class = " class='num'" if col in numeric_cols else ""
            cells.append(f"<td{td_class}>{_html.escape(val)}</td>")
        ranking_rows.append(f"<tr>{''.join(cells)}</tr>")

    feature_cards_html = ""
    for feature_name, img_b64 in feature_figures:
        img_tag = (
            f"<img src='data:image/png;base64,{img_b64}' alt='{_html.escape(feature_name)}'>"
            if img_b64
            else "<p><em>Figure unavailable.</em></p>"
        )
        feature_cards_html += (
            f"<div class='feature-card'>"
            f"<h3>{_html.escape(feature_name)}</h3>"
            f"{img_tag}"
            f"</div>"
        )

    warnings_html = ""
    if warning_messages:
        items = "".join(f"<li>{_html.escape(w)}</li>" for w in warning_messages)
        warnings_html = f"<div class='warnings'><strong>Warnings</strong><ul>{items}</ul></div>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <div class="report-header">
    <h1>{_html.escape(title)}</h1>
    <p class="meta">Generated {_html.escape(generated_at)}</p>
  </div>

  <h2>Residual Health</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{exec_rows}</tbody>
  </table>

  <h2>Feature Rankings — Top {top_n}</h2>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(ranking_rows)}</tbody>
  </table>

  <h2>Feature Diagnostics</h2>
  {feature_cards_html}

  {warnings_html}
</body>
</html>"""
