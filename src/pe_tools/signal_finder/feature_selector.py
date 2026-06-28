from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from pe_tools.signal_finder.v2 import (
    CustomSplitLike,
    SeriesLike,
    _normalize_custom_splits,
    _oof_r2,
    _SplitSpec,
)

_FEATURE_COLOR = "#2C7FB8"
_NULL_COLOR = "#9E9E9E"
_ZERO_LINE_COLOR = "#424242"

_VALID_SPLIT_STRATEGIES = {"bootstrap", "stratified_bootstrap", "repeated_kfold", "group_kfold"}
_VALID_MODEL_TYPES = {"xgboost", "random_forest"}
_INTERACTION_R2_EPSILON = 0.001


def _require_fitted(selector: FeatureSelector, attr: str) -> None:
    if not selector._fitted:
        raise RuntimeError(f"FeatureSelector must be fitted before calling {attr}")


def _is_binary_y(y: pd.Series) -> bool:
    unique = set(pd.to_numeric(y, errors="coerce").dropna().unique().tolist())
    return bool(unique) and unique.issubset({0.0, 1.0})


def _is_categorical_feature(values: pd.Series, explicit_cats: set[str]) -> bool:
    if str(values.name) in explicit_cats:
        return True
    return isinstance(values.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(
        values
    )


def _encode_feature(
    values: pd.Series,
    is_categorical: bool,
    train_idx: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    if is_categorical:
        str_vals = values.astype(str)
        train_cats = str_vals.iloc[train_idx].value_counts().index.tolist()
        mapping = {cat: float(i) for i, cat in enumerate(train_cats)}
        return str_vals.map(mapping).fillna(-1.0).to_numpy(dtype=float)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    train_vals = numeric[train_idx]
    fill = float(np.nanmedian(train_vals)) if not np.isnan(train_vals).all() else 0.0
    return np.where(np.isnan(numeric), fill, numeric)


def _gini_safe(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return 2.0 * float(roc_auc_score(y_true, y_pred)) - 1.0
    except Exception:
        return float("nan")


def _spearman_safe(x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    if len(x) < 3:
        return float("nan")
    try:
        r, _ = spearmanr(x, y)
        return float(r) if np.isfinite(r) else float("nan")
    except Exception:
        return float("nan")


@dataclass
class FeatureSelector:
    """Bootstrap-based feature selector that works on raw (X, y).

    Evaluates each feature independently across bootstrap or k-fold splits,
    comparing lift against a permuted-null baseline. Optional multivariate
    refinement prunes features whose marginal importance is weak and computes
    pairwise Spearman correlations among selected features.

    After `.fit()`, call `.find_interactions()` to detect pairwise interaction
    effects among selected features.
    """

    n_bootstraps: int = 50
    test_size: float = 0.20
    split_strategy: str = "bootstrap"
    n_splits: int = 5
    model_type: str = "xgboost"
    max_depth: int = 1
    random_state: int = 42
    null_beat_rate_threshold: float = 0.80
    positive_score_rate_threshold: float = 0.80
    refinement_enabled: bool = False
    n_refinement_bootstraps: int = 20
    correlation_threshold: float = 0.20
    model_params: dict[str, Any] | None = None
    verbose: bool = False

    summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    bootstrap_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    selected_features_: list[str] = field(init=False, default_factory=list)
    refinement_summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    feature_correlation_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    interaction_summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    interaction_bootstrap_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    _X: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    _y: pd.Series = field(init=False, default_factory=pd.Series, repr=False)
    _is_binary: bool = field(init=False, default=False, repr=False)
    _categorical_features: set[str] = field(init=False, default_factory=set, repr=False)
    _candidate_features: list[str] = field(init=False, default_factory=list, repr=False)
    _fitted: bool = field(init=False, default=False, repr=False)
    _interactions_fitted: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.split_strategy not in _VALID_SPLIT_STRATEGIES:
            raise ValueError(f"split_strategy must be one of: {sorted(_VALID_SPLIT_STRATEGIES)}")
        if self.model_type not in _VALID_MODEL_TYPES:
            raise ValueError("model_type must be one of: xgboost, random_forest")
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be at least 1")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.n_refinement_bootstraps < 1:
            raise ValueError("n_refinement_bootstraps must be at least 1")

    def fit(
        self,
        X: pd.DataFrame,
        y: SeriesLike,
        exclude_features: list[str] | None = None,
        group_col: Any = None,
    ) -> FeatureSelector:
        """Fit univariate bootstrap evaluation for all non-excluded features.

        Args:
            X: Feature matrix.
            y: Target vector. Binary 0/1 is auto-detected; anything else is treated as regression.
            exclude_features: Column names to skip entirely.
            group_col: Group labels when split_strategy='group_kfold'.

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) != len(X):
            raise ValueError("y must have the same length as X")
        if not np.isfinite(y_arr).all():
            raise ValueError("y must be finite")

        y_series = pd.Series(y_arr, index=X.index, name="y")
        self._y = y_series
        self._is_binary = _is_binary_y(y_series)

        excluded = set(exclude_features or [])
        features = [f for f in X.columns if f not in excluded]
        self._X = X[features].copy()
        self._categorical_features = {
            f for f in features if _is_categorical_feature(X[f], set())
        }
        self._candidate_features = features

        group_series: pd.Series | None = None
        if group_col is not None:
            group_series = pd.Series(np.asarray(group_col), index=X.index, name="group")

        split_specs = self._make_splits(X, y_series, group_series, None)
        if not split_specs:
            raise ValueError("No valid train/validation splits were created")

        rng = np.random.default_rng(self.random_state + 999_983)
        rows: list[dict[str, Any]] = []

        for feat_idx, feat in enumerate(features):
            if self.verbose:
                print(f"[FeatureSelector] {feat_idx + 1}/{len(features)}: {feat}")

            is_cat = feat in self._categorical_features

            for split_spec in split_specs:
                tr_idx = split_spec.train_idx
                vl_idx = split_spec.validation_idx

                train_y = y_series.iloc[tr_idx].to_numpy(dtype=float)
                val_y = y_series.iloc[vl_idx].to_numpy(dtype=float)
                train_mean = float(np.mean(train_y))

                enc = _encode_feature(self._X[feat], is_cat, tr_idx)
                X_tr = pd.DataFrame({"feature": enc[tr_idx]})
                X_vl = pd.DataFrame({"feature": enc[vl_idx]})

                null_enc_train = rng.permutation(enc[tr_idx])
                null_enc_val = rng.permutation(enc[vl_idx])
                X_tr_null = pd.DataFrame({"feature": null_enc_train})
                X_vl_null = pd.DataFrame({"feature": null_enc_val})

                seed = self.random_state + split_spec.split_id * 997 + feat_idx * 13
                model = self._build_model(depth=self.max_depth, seed=seed)
                model.fit(X_tr, train_y)
                pred = np.asarray(model.predict(X_vl), dtype=float)
                r2 = _oof_r2(val_y, pred, train_mean, None)

                null_model = self._build_model(depth=self.max_depth, seed=seed + 1)
                null_model.fit(X_tr_null, train_y)
                null_pred = np.asarray(null_model.predict(X_vl_null), dtype=float)
                null_r2 = _oof_r2(val_y, null_pred, train_mean, None)

                if self._is_binary:
                    robust_metric = _gini_safe(val_y, pred)
                    null_robust_metric = _gini_safe(val_y, null_pred)
                else:
                    robust_metric = _spearman_safe(enc[vl_idx], val_y)
                    null_robust_metric = _spearman_safe(null_enc_val, val_y)

                all_finite = all(
                    np.isfinite(v) for v in [r2, null_r2, robust_metric, null_robust_metric]
                )
                beats_null = (
                    float(int(r2 > null_r2 and robust_metric > null_robust_metric))
                    if all_finite
                    else float("nan")
                )

                rows.append({
                    "feature": feat,
                    "split_id": split_spec.split_id,
                    "r2": r2,
                    "robust_metric": robust_metric,
                    "null_r2": null_r2,
                    "null_robust_metric": null_robust_metric,
                    "beats_null": beats_null,
                })

        self.bootstrap_results_ = pd.DataFrame(rows)
        self.summary_ = self._build_univariate_summary()
        self.selected_features_ = self.summary_.loc[
            self.summary_["selected"], "feature"
        ].tolist()

        if self.refinement_enabled and len(self.selected_features_) >= 2:
            self._run_refinement(y_series)

        self._fitted = True
        return self

    def find_interactions(
        self,
        top_n: int = 10,
        candidate_features: list[str] | None = None,
    ) -> FeatureSelector:
        """Detect pairwise interaction effects among selected (or specified) features.

        Args:
            top_n: Maximum number of features to consider; takes the first top_n
                from selected_features_ (or candidate_features if provided).
            candidate_features: Explicit feature list; overrides selected_features_.

        Returns:
            self
        """
        _require_fitted(self, "find_interactions()")

        features_to_use = (
            candidate_features if candidate_features is not None else self.selected_features_
        )[:top_n]
        if len(features_to_use) < 2:
            raise ValueError(
                "Need at least 2 candidate features for interaction detection. "
                "Ensure fit() produced selected_features_ or pass candidate_features."
            )

        split_specs = self._make_splits(self._X[features_to_use], self._y, None, None)
        if not split_specs:
            raise ValueError("No valid splits for interaction detection")

        pairs = list(combinations(features_to_use, 2))
        rng = np.random.default_rng(self.random_state)
        rows: list[dict[str, Any]] = []

        for split_spec in split_specs:
            tr_idx = split_spec.train_idx
            vl_idx = split_spec.validation_idx
            train_y = self._y.iloc[tr_idx].to_numpy(dtype=float)
            val_y = self._y.iloc[vl_idx].to_numpy(dtype=float)
            null_mean = float(np.mean(train_y))

            for feat_a, feat_b in pairs:
                is_cat_a = feat_a in self._categorical_features
                is_cat_b = feat_b in self._categorical_features
                enc_a = _encode_feature(self._X[feat_a], is_cat_a, tr_idx)
                enc_b = _encode_feature(self._X[feat_b], is_cat_b, tr_idx)

                X_tr = pd.DataFrame({"feature_a": enc_a[tr_idx], "feature_b": enc_b[tr_idx]})
                X_vl = pd.DataFrame({"feature_a": enc_a[vl_idx], "feature_b": enc_b[vl_idx]})

                X_tr_null = X_tr.copy()
                X_vl_null = X_vl.copy()
                X_tr_null["feature_b"] = rng.permutation(enc_b[tr_idx])
                X_vl_null["feature_b"] = rng.permutation(enc_b[vl_idx])

                base_seed = self.random_state + split_spec.split_id * 997
                d1_r2 = self._fit_score(
                    X_tr, train_y, X_vl, val_y, null_mean, depth=1, seed=base_seed
                )
                d2_r2 = self._fit_score(
                    X_tr, train_y, X_vl, val_y, null_mean, depth=2, seed=base_seed + 1
                )
                d1_null = self._fit_score(
                    X_tr_null, train_y, X_vl_null, val_y, null_mean, depth=1, seed=base_seed + 2
                )
                d2_null = self._fit_score(
                    X_tr_null, train_y, X_vl_null, val_y, null_mean, depth=2, seed=base_seed + 3
                )

                lift = d2_r2 - d1_r2
                null_lift = d2_null - d1_null
                rows.append({
                    "feature_1": feat_a,
                    "feature_2": feat_b,
                    "bootstrap_id": split_spec.split_id,
                    "depth1_r2": d1_r2,
                    "depth2_r2": d2_r2,
                    "interaction_lift": lift,
                    "null_lift": null_lift,
                    "beats_null": bool(lift > null_lift),
                })

        self.interaction_bootstrap_results_ = pd.DataFrame(rows)
        self.interaction_summary_ = self._build_interaction_summary()
        self._interactions_fitted = True
        return self

    # --- Plotting ---

    def plot_selected_features(self, top_n: int = 20) -> Figure:
        """Two-panel summary: metric bar chart (left) and null beat rate histogram (right)."""
        _require_fitted(self, "plot_selected_features()")

        import matplotlib.pyplot as plt

        df = self.summary_.head(top_n).copy()
        colors = ["#2E7D32" if s else "#9E9E9E" for s in df["selected"]]
        y_pos = np.arange(len(df))[::-1]

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, max(5, len(df) * 0.45)))

        ax_left.barh(
            y_pos,
            df["mean_robust_metric"],
            xerr=df["std_robust_metric"].fillna(0.0),
            color=colors,
            alpha=0.85,
            height=0.7,
            error_kw={"elinewidth": 1.0, "capsize": 3},
        )
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(df["feature"].tolist(), fontsize=9)
        metric_label = "Gini (2·AUC − 1)" if self._is_binary else "Spearman correlation"
        ax_left.set_xlabel(metric_label)
        ax_left.set_title("Mean Robust Metric  (green = selected, gray = not selected)")
        ax_left.axvline(0.0, color=_ZERO_LINE_COLOR, linewidth=1.0, linestyle=":")
        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)

        nbr_vals = self.summary_["null_beat_rate"].dropna()
        bins = min(20, max(5, len(nbr_vals)))
        ax_right.hist(nbr_vals, bins=bins, color=_FEATURE_COLOR, alpha=0.70)
        ax_right.axvline(
            self.null_beat_rate_threshold,
            color="#C62828",
            linestyle="--",
            linewidth=1.5,
            label=f"threshold = {self.null_beat_rate_threshold:.0%}",
        )
        ax_right.set_xlabel("Null beat rate")
        ax_right.set_ylabel("Feature count")
        ax_right.set_title("Null Beat Rate Distribution Across Features")
        ax_right.legend(fontsize=9)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    def plot_feature_correlations(self) -> Figure:
        """Heatmap of Spearman correlations among selected features (masked at threshold).

        Raises:
            RuntimeError: If refinement was not run or fit() was not called.
        """
        if not self.refinement_enabled or self.feature_correlation_.empty:
            raise RuntimeError(
                "refinement_enabled must be True and fit() must be called first"
            )

        import matplotlib.pyplot as plt

        corr = self.feature_correlation_.astype(float)
        n = len(corr)
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))

        values = corr.to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(values))) if not np.isnan(values).all() else 1.0
        im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ")

        for i in range(n):
            for j in range(n):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        ax.set_xticks(range(n))
        ax.set_xticklabels(corr.columns.tolist(), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(corr.index.tolist(), fontsize=8)
        ax.set_title(f"Feature Correlation (NaN where |ρ| < {self.correlation_threshold})")
        fig.tight_layout()
        return fig

    def plot_interactions(self, top_n: int = 5) -> dict[str, Figure]:
        """Return one diagnostic figure per top-N ranked interaction pair.

        Raises:
            RuntimeError: If find_interactions() has not been called.
        """
        if not self._interactions_fitted or self.interaction_summary_.empty:
            raise RuntimeError("Call find_interactions() before plot_interactions()")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        figures: dict[str, Figure] = {}
        for _, row in self.interaction_summary_.head(top_n).iterrows():
            feat_a = str(row["feature_1"])
            feat_b = str(row["feature_2"])
            key = f"{feat_a}__x__{feat_b}"
            pair_boot = self.interaction_bootstrap_results_.loc[
                self.interaction_bootstrap_results_["feature_1"].eq(feat_a)
                & self.interaction_bootstrap_results_["feature_2"].eq(feat_b)
            ]
            figures[key] = self._plot_pair(feat_a, feat_b, row, pair_boot)
        return figures

    def plot_interaction_surface(
        self,
        feature_a: str,
        feature_b: str,
        mode: str = "actionable",
        surface_smoother: str = "random_forest",
        winsorize_quantiles: tuple[float, float] = (0.01, 0.99),
        materiality_threshold: float | None = None,
        show_raw_points: bool = True,
        show_zero_contour: bool = True,
        max_scatter_points: int = 10000,
        min_support: int = 100,
        random_state: int | None = None,
    ) -> Figure:
        """Return an actionable interaction surface plot for a single feature pair."""
        if not self._interactions_fitted:
            raise RuntimeError("Call find_interactions() before plot_interaction_surface()")
        valid_modes = {"actionable", "smooth_contour", "sliced_curves", "heatmap", "hexbin"}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes!r}")
        for feat in (feature_a, feature_b):
            if feat not in self._candidate_features:
                raise ValueError(
                    f"{feat!r} was not in features passed to fit(); "
                    f"available: {self._candidate_features}"
                )

        rng_seed = random_state if random_state is not None else self.random_state
        summary_row = self._lookup_pair_summary(feature_a, feature_b)
        pair_type = self._infer_pair_plot_type(feature_a, feature_b)
        is_cat_a = feature_a in self._categorical_features
        is_cat_b = feature_b in self._categorical_features

        effective_mode = mode
        if mode == "actionable":
            effective_mode = {
                "continuous_continuous": "smooth_contour",
                "categorical_continuous": "sliced_curves",
                "categorical_categorical": "heatmap",
            }[pair_type]

        val_idx, net = self._one_shot_net_predict(feature_a, feature_b, is_cat_a, is_cat_b)

        if materiality_threshold is None:
            q25, q75 = np.nanpercentile(np.abs(net), [25, 75])
            materiality_threshold = max(float(q75 - q25) * 0.1, 1e-6)

        import matplotlib.pyplot as plt

        fig, (ax_main, ax_rec) = plt.subplots(
            1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]}
        )

        _nan = float("nan")
        lift = float(summary_row["mean_interaction_lift"]) if summary_row is not None else _nan
        nbr = float(summary_row["interaction_null_beat_rate"]) if summary_row is not None else _nan
        fig.suptitle(
            f"Actionable Interaction Surface: {feature_a} × {feature_b}   "
            f"(mean lift={lift:.4f}, null beat rate={nbr:.0%})",
            fontsize=12,
        )

        shape: str = "unclear"
        warnings: list[str] = []
        xx: np.ndarray[Any, Any] | None = None
        yy: np.ndarray[Any, Any] | None = None
        zz: np.ndarray[Any, Any] | None = None

        if effective_mode in ("smooth_contour", "hexbin"):
            shape, warnings, xx, yy, zz = self._plot_smooth_contour(
                feature_a, feature_b, val_idx, net,
                winsorize_quantiles, show_raw_points, show_zero_contour,
                materiality_threshold, max_scatter_points, rng_seed,
                effective_mode == "hexbin", ax_main,
            )
        elif effective_mode == "sliced_curves":
            cat_feat = feature_a if is_cat_a else feature_b
            cont_feat = feature_b if is_cat_a else feature_a
            shape, warnings = self._plot_sliced_interaction_curves(
                cont_feat, cat_feat, val_idx, net, min_support, ax_main
            )
        else:
            shape, warnings = self._plot_support_heatmap(
                feature_a, feature_b, val_idx, net, min_support, ax_main
            )

        self._render_recommendation_panel(
            feature_a, feature_b, pair_type, shape, warnings,
            materiality_threshold, summary_row, xx, yy, zz, ax_rec,
        )
        ax_rec.set_title("Suggested respecification", fontsize=10, pad=8)
        fig.tight_layout()
        return fig

    # --- Private: build summaries ---

    def _build_univariate_summary(self) -> pd.DataFrame:
        metric_name = "gini" if self._is_binary else "spearman"
        rows: list[dict[str, Any]] = []

        for feat, group in self.bootstrap_results_.groupby("feature", sort=False):
            is_cat = feat in self._categorical_features
            dtype_str = "categorical" if is_cat else "continuous"

            r2_vals = group["r2"].dropna()
            robust_vals = group["robust_metric"].dropna()
            valid_beats = group["beats_null"].dropna()
            null_beat_rate = float(valid_beats.mean()) if len(valid_beats) else float("nan")
            positive_score_rate = float(
                ((group["r2"] > 0) & (group["robust_metric"] > 0)).mean()
            )

            selected = bool(
                np.isfinite(null_beat_rate)
                and null_beat_rate >= self.null_beat_rate_threshold
                and positive_score_rate >= self.positive_score_rate_threshold
            )

            rows.append({
                "feature": feat,
                "dtype": dtype_str,
                "mean_r2": float(r2_vals.mean()) if len(r2_vals) else float("nan"),
                "std_r2": float(r2_vals.std(ddof=0)) if len(r2_vals) else float("nan"),
                "mean_robust_metric": (
                    float(robust_vals.mean()) if len(robust_vals) else float("nan")
                ),
                "std_robust_metric": (
                    float(robust_vals.std(ddof=0)) if len(robust_vals) else float("nan")
                ),
                "null_beat_rate": null_beat_rate,
                "positive_score_rate": positive_score_rate,
                "metric_name": metric_name,
                "selected": selected,
            })

        df = pd.DataFrame(rows).sort_values(
            by=["null_beat_rate", "mean_robust_metric"],
            ascending=False,
            ignore_index=True,
        )
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def _build_interaction_summary(self) -> pd.DataFrame:
        empty_cols = [
            "feature_1", "feature_2", "mean_interaction_lift", "median_interaction_lift",
            "positive_lift_rate", "interaction_null_beat_rate", "mean_depth2_r2",
            "mean_depth1_r2", "rank",
        ]
        if self.interaction_bootstrap_results_.empty:
            return pd.DataFrame(columns=empty_cols)

        rows: list[dict[str, Any]] = []
        for (feat_a, feat_b), group in self.interaction_bootstrap_results_.groupby(
            ["feature_1", "feature_2"], sort=False
        ):
            lift = group["interaction_lift"].dropna()
            rows.append({
                "feature_1": feat_a,
                "feature_2": feat_b,
                "mean_interaction_lift": float(lift.mean()),
                "median_interaction_lift": float(lift.median()),
                "positive_lift_rate": (
                    float((lift > _INTERACTION_R2_EPSILON).mean()) if len(lift) else float("nan")
                ),
                "interaction_null_beat_rate": (
                    float(group["beats_null"].mean()) if len(group) else float("nan")
                ),
                "mean_depth2_r2": float(group["depth2_r2"].mean()),
                "mean_depth1_r2": float(group["depth1_r2"].mean()),
            })

        summary = pd.DataFrame(rows).sort_values(
            "mean_interaction_lift", ascending=False, ignore_index=True
        )
        summary["rank"] = np.arange(1, len(summary) + 1)
        return summary

    # --- Private: refinement ---

    def _run_refinement(self, y: pd.Series) -> None:
        n_obs = len(self._X)
        positions = np.arange(n_obs)
        rng = np.random.default_rng(self.random_state + 1_234_567)
        train_size = max(1, min(n_obs - 1, int(round((1.0 - self.test_size) * n_obs))))
        y_arr = y.to_numpy(dtype=float)

        imp_rows: list[dict[str, Any]] = []
        for boot_id in range(self.n_refinement_bootstraps):
            tr_idx = np.sort(rng.choice(positions, size=train_size, replace=False))
            vl_idx = np.setdiff1d(positions, tr_idx)
            if len(vl_idx) == 0:
                continue

            enc_arrays: dict[str, np.ndarray[Any, Any]] = {}
            for feat in self.selected_features_:
                is_cat = feat in self._categorical_features
                enc_arrays[feat] = _encode_feature(self._X[feat], is_cat, tr_idx)

            X_tr = pd.DataFrame({f: enc_arrays[f][tr_idx] for f in self.selected_features_})
            X_vl = pd.DataFrame({f: enc_arrays[f][vl_idx] for f in self.selected_features_})
            train_y = y_arr[tr_idx]
            val_y = y_arr[vl_idx]
            train_mean = float(np.mean(train_y))

            model = self._build_model(depth=3, seed=self.random_state + boot_id)
            model.fit(X_tr, train_y)
            baseline_pred = np.asarray(model.predict(X_vl), dtype=float)
            baseline_r2 = _oof_r2(val_y, baseline_pred, train_mean, None)

            for feat in self.selected_features_:
                X_vl_perm = X_vl.copy()
                X_vl_perm[feat] = rng.permutation(X_vl_perm[feat].to_numpy())
                perm_pred = np.asarray(model.predict(X_vl_perm), dtype=float)
                perm_r2 = _oof_r2(val_y, perm_pred, train_mean, None)
                imp_rows.append({
                    "feature": feat,
                    "marginal_importance": baseline_r2 - perm_r2,
                })

        if not imp_rows:
            return

        raw = pd.DataFrame(imp_rows)
        refinement_rows: list[dict[str, Any]] = []
        pruned: set[str] = set()

        for feat in self.selected_features_:
            feat_imp = raw.loc[raw["feature"] == feat, "marginal_importance"]
            mean_imp = float(feat_imp.mean()) if len(feat_imp) else float("nan")
            std_imp = float(feat_imp.std(ddof=0)) if len(feat_imp) else float("nan")
            pct_non_positive = float((feat_imp <= 0).mean()) if len(feat_imp) else 1.0
            kept = pct_non_positive <= 0.50
            if not kept:
                pruned.add(str(feat))
            refinement_rows.append({
                "feature": feat,
                "mean_marginal_importance": mean_imp,
                "std_marginal_importance": std_imp,
                "kept": kept,
            })

        self.refinement_summary_ = pd.DataFrame(refinement_rows)
        self.selected_features_ = [f for f in self.selected_features_ if f not in pruned]

        if len(self.selected_features_) >= 2:
            self.feature_correlation_ = self._compute_feature_correlation()

    def _compute_feature_correlation(self) -> pd.DataFrame:
        feats = self.selected_features_
        all_idx = np.arange(len(self._X))
        enc_vals: dict[str, np.ndarray[Any, Any]] = {}
        for f in feats:
            is_cat = f in self._categorical_features
            enc_vals[f] = _encode_feature(self._X[f], is_cat, all_idx)

        corr_matrix = pd.DataFrame(np.nan, index=feats, columns=feats, dtype=float)
        for f1 in feats:
            corr_matrix.loc[f1, f1] = 1.0
            for f2 in feats:
                if f1 >= f2:
                    continue
                r, _ = spearmanr(enc_vals[f1], enc_vals[f2])
                val = float(r) if np.isfinite(r) else float("nan")
                corr_matrix.loc[f1, f2] = val
                corr_matrix.loc[f2, f1] = val

        masked = corr_matrix.copy()
        for f1 in feats:
            for f2 in feats:
                if f1 != f2:
                    val = float(masked.loc[f1, f2])
                    if np.isfinite(val) and abs(val) < self.correlation_threshold:
                        masked.loc[f1, f2] = float("nan")
        return masked

    # --- Private: splits ---

    def _make_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        group_values: pd.Series | None,
        custom_splits: Sequence[CustomSplitLike] | None,
    ) -> list[_SplitSpec]:
        if custom_splits is not None:
            return _normalize_custom_splits(custom_splits, X.index)

        n_obs = len(X)
        positions = np.arange(n_obs)

        if self.split_strategy == "group_kfold":
            if group_values is None:
                raise ValueError("group_col is required when split_strategy='group_kfold'")
            if pd.Series(group_values).nunique(dropna=False) < self.n_splits:
                raise ValueError("group_col must contain at least n_splits unique groups")
            splitter = GroupKFold(n_splits=self.n_splits)
            return [
                _SplitSpec(split_id=i, train_idx=np.asarray(tr), validation_idx=np.asarray(vl))
                for i, (tr, vl) in enumerate(splitter.split(X, groups=group_values))
            ]

        if self.split_strategy == "repeated_kfold":
            splits: list[_SplitSpec] = []
            repeat_count = int(np.ceil(self.n_bootstraps / self.n_splits))
            for repeat in range(repeat_count):
                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state + repeat,
                )
                for tr, vl in kf.split(X):
                    splits.append(
                        _SplitSpec(
                            split_id=len(splits),
                            train_idx=np.asarray(tr),
                            validation_idx=np.asarray(vl),
                        )
                    )
                    if len(splits) >= self.n_bootstraps:
                        return splits
            return splits

        rng = np.random.default_rng(self.random_state)
        train_size = max(1, min(n_obs - 1, int(round((1.0 - self.test_size) * n_obs))))
        splits = []

        if self.split_strategy == "stratified_bootstrap":
            y_array = np.asarray(y, dtype=float) if y is not None else np.zeros(n_obs)
            classes, class_counts = np.unique(y_array, return_counts=True)
            for _ in range(self.n_bootstraps):
                train_idx_parts: list[np.ndarray[Any, Any]] = []
                for cls, cnt in zip(classes, class_counts, strict=True):
                    cls_positions = positions[y_array == cls]
                    n_cls = max(1, int(round(train_size * cnt / n_obs)))
                    n_cls = min(n_cls, len(cls_positions))
                    train_idx_parts.append(rng.choice(cls_positions, size=n_cls, replace=False))
                tr = np.sort(np.concatenate(train_idx_parts))
                vl = np.setdiff1d(positions, tr)
                if len(vl) > 0:
                    splits.append(
                        _SplitSpec(split_id=len(splits), train_idx=tr, validation_idx=vl)
                    )
            return splits

        for _ in range(self.n_bootstraps):
            tr = np.sort(rng.choice(positions, size=train_size, replace=False))
            vl = np.setdiff1d(positions, tr)
            if len(vl) > 0:
                splits.append(_SplitSpec(split_id=len(splits), train_idx=tr, validation_idx=vl))
        return splits

    # --- Private: model helpers ---

    def _build_model(self, depth: int, seed: int) -> Any:
        if self.model_type == "xgboost":
            params: dict[str, Any] = {
                "n_estimators": 120,
                "max_depth": depth,
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
            "max_depth": depth,
            "min_samples_leaf": 10,
            "max_features": 1.0,
            "bootstrap": True,
            "random_state": seed,
            "n_jobs": 1,
        }
        params.update(self.model_params or {})
        return RandomForestRegressor(**params)

    def _fit_score(
        self,
        X_train: pd.DataFrame,
        train_y: np.ndarray[Any, Any],
        X_val: pd.DataFrame,
        val_y: np.ndarray[Any, Any],
        null_mean: float,
        depth: int,
        seed: int,
    ) -> float:
        model = self._build_model(depth=depth, seed=seed)
        model.fit(X_train, train_y)
        pred = np.asarray(model.predict(X_val), dtype=float)
        return _oof_r2(val_y, pred, null_mean, None)

    # --- Private: interaction plotting ---

    def _plot_pair(
        self,
        feat_a: str,
        feat_b: str,
        summary_row: pd.Series,
        pair_boot: pd.DataFrame,
    ) -> Figure:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        lift = float(summary_row["mean_interaction_lift"])
        nbr = float(summary_row["interaction_null_beat_rate"])
        fig.suptitle(
            f"Interaction: {feat_a} × {feat_b}   "
            f"(mean lift={lift:.4f}, null beat rate={nbr:.0%})",
            fontsize=12,
        )
        is_cat_a = feat_a in self._categorical_features
        is_cat_b = feat_b in self._categorical_features

        if not is_cat_a and not is_cat_b:
            val_idx, net = self._one_shot_net_predict(feat_a, feat_b, is_cat_a, is_cat_b)
            self._plot_scatter_interaction(feat_a, feat_b, val_idx, net, axes[0])
            self._plot_target_scatter(feat_a, feat_b, val_idx, axes[1])
        elif is_cat_a != is_cat_b:
            cat_feat = feat_a if is_cat_a else feat_b
            cont_feat = feat_b if is_cat_a else feat_a
            self._plot_conditional_curves(cont_feat, cat_feat, axes[0])
            self._plot_target_conditional_curves(cont_feat, cat_feat, axes[1])
        else:
            val_idx, net = self._one_shot_net_predict(feat_a, feat_b, is_cat_a, is_cat_b)
            self._plot_interaction_bubbles(feat_a, feat_b, val_idx, net, axes[0])
            self._plot_target_bubbles(feat_a, feat_b, val_idx, axes[1])

        self._plot_lift_dist(pair_boot, axes[2])
        fig.tight_layout()
        return fig

    def _one_shot_net_predict(
        self,
        feat_a: str,
        feat_b: str,
        is_cat_a: bool,
        is_cat_b: bool,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        n = len(self._X)
        rng = np.random.default_rng(self.random_state)
        train_size = max(1, int(round(0.8 * n)))
        train_idx = np.sort(rng.choice(np.arange(n), size=train_size, replace=False))
        val_idx = np.setdiff1d(np.arange(n), train_idx)

        enc_a = _encode_feature(self._X[feat_a], is_cat_a, train_idx)
        enc_b = _encode_feature(self._X[feat_b], is_cat_b, train_idx)
        X_tr = pd.DataFrame({"feature_a": enc_a[train_idx], "feature_b": enc_b[train_idx]})
        X_vl = pd.DataFrame({"feature_a": enc_a[val_idx], "feature_b": enc_b[val_idx]})
        train_y = self._y.iloc[train_idx].to_numpy(dtype=float)

        m1 = self._build_model(depth=1, seed=self.random_state)
        m2 = self._build_model(depth=2, seed=self.random_state)
        m1.fit(X_tr, train_y)
        m2.fit(X_tr, train_y)
        net = (
            np.asarray(m2.predict(X_vl), dtype=float)
            - np.asarray(m1.predict(X_vl), dtype=float)
        )
        return val_idx, net

    def _infer_pair_plot_type(self, feat_a: str, feat_b: str) -> str:
        _LOW_CARDINALITY = 10

        def _low_card(feat: str) -> bool:
            if feat in self._categorical_features:
                return True
            return int(self._X[feat].nunique()) <= _LOW_CARDINALITY

        a_cat = _low_card(feat_a)
        b_cat = _low_card(feat_b)
        if not a_cat and not b_cat:
            return "continuous_continuous"
        if a_cat != b_cat:
            return "categorical_continuous"
        return "categorical_categorical"

    def _lookup_pair_summary(self, feat_a: str, feat_b: str) -> pd.Series | None:
        df = self.interaction_summary_
        row = df[
            ((df["feature_1"] == feat_a) & (df["feature_2"] == feat_b))
            | ((df["feature_1"] == feat_b) & (df["feature_2"] == feat_a))
        ]
        return row.iloc[0] if not row.empty else None

    def _plot_smooth_contour(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        winsorize_quantiles: tuple[float, float],
        show_raw_points: bool,
        show_zero_contour: bool,
        materiality_threshold: float,
        max_scatter_points: int,
        random_state: int,
        hexbin_mode: bool,
        ax: Any,
    ) -> tuple[str, list[str], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        import matplotlib.pyplot as plt

        lo_q, hi_q = winsorize_quantiles
        vals_a = pd.to_numeric(self._X[feat_a], errors="coerce").to_numpy(dtype=float)[val_idx]
        vals_b = pd.to_numeric(self._X[feat_b], errors="coerce").to_numpy(dtype=float)[val_idx]

        valid = np.isfinite(vals_a) & np.isfinite(vals_b) & np.isfinite(net)
        vals_a, vals_b, net_v = vals_a[valid], vals_b[valid], net[valid]

        _empty: np.ndarray[Any, Any] = np.array([])
        if len(vals_a) < 10:
            ax.text(0.5, 0.5, "Insufficient data for surface", ha="center", va="center")
            ax.axis("off")
            return "unclear", ["Insufficient validation data"], _empty, _empty, _empty

        a_lo = float(np.nanpercentile(vals_a, lo_q * 100))
        a_hi = float(np.nanpercentile(vals_a, hi_q * 100))
        b_lo = float(np.nanpercentile(vals_b, lo_q * 100))
        b_hi = float(np.nanpercentile(vals_b, hi_q * 100))
        a_lo, a_hi = (a_lo - 1, a_hi + 1) if a_lo == a_hi else (a_lo, a_hi)
        b_lo, b_hi = (b_lo - 1, b_hi + 1) if b_lo == b_hi else (b_lo, b_hi)

        va = np.clip(vals_a, a_lo, a_hi)
        vb = np.clip(vals_b, b_lo, b_hi)

        min_leaf = max(30, int(0.01 * len(va)))
        smoother = RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=min_leaf,
            random_state=random_state, n_jobs=1,
        )
        smoother.fit(np.column_stack([va, vb]), net_v)

        GRID_SIZE = 40
        xx, yy = np.meshgrid(
            np.linspace(a_lo, a_hi, GRID_SIZE), np.linspace(b_lo, b_hi, GRID_SIZE)
        )
        zz = smoother.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        if hexbin_mode:
            sc = ax.hexbin(
                va, vb, C=net_v, reduce_C_function=np.mean,
                gridsize=25, mincnt=3, cmap="RdBu_r",
            )
            plt.colorbar(sc, ax=ax, label="Mean net interaction effect")
            ax.set_title(f"Hexbin: {feat_a} × {feat_b}", fontsize=10)
        else:
            vmax = float(max(abs(float(zz.min())), abs(float(zz.max())))) or 1.0
            cf = ax.contourf(
                xx, yy, zz, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.85
            )
            plt.colorbar(cf, ax=ax, label="Net interaction effect")

            if show_raw_points:
                n_pts = len(va)
                if n_pts > max_scatter_points:
                    rng = np.random.default_rng(random_state)
                    sub = rng.choice(n_pts, size=max_scatter_points, replace=False)
                    ax.scatter(va[sub], vb[sub], s=4, alpha=0.08, color="gray", rasterized=True)
                else:
                    ax.scatter(va, vb, s=4, alpha=0.08, color="gray", rasterized=True)

            if show_zero_contour:
                import contextlib
                with contextlib.suppress(Exception):
                    ax.contour(
                        xx, yy, zz, levels=[0], colors="black",
                        linestyles="--", linewidths=1.5,
                    )

            ax.set_title(
                f"Interaction Surface: {feat_a} × {feat_b}\n"
                "(smoothed net effect = depth-2 − depth-1)",
                fontsize=10,
            )

        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        if lo_q > 0 or hi_q < 1:
            ax.text(
                0.02, 0.02, f"Axes: {lo_q:.0%}–{hi_q:.0%} percentile",
                transform=ax.transAxes, fontsize=7, color="gray", va="bottom",
            )

        mat_frac = float(np.mean(np.abs(zz) > materiality_threshold))
        if mat_frac < 0.05:
            shape = "unclear"
        elif mat_frac < 0.35:
            shape = "local_threshold"
        else:
            shape = "smooth_surface"

        warnings: list[str] = []
        if shape == "local_threshold" and not hexbin_mode:
            pos_mask = zz > materiality_threshold
            if pos_mask.any():
                x_thresh = float(np.nanpercentile(xx[pos_mask], 20))
                y_thresh = float(np.nanpercentile(yy[pos_mask], 80))
                for val, axis_dir, lo, hi, feat in [
                    (x_thresh, "v", a_lo, a_hi, feat_a),
                    (y_thresh, "h", b_lo, b_hi, feat_b),
                ]:
                    if lo < val < hi:
                        if axis_dir == "v":
                            ax.axvline(
                                val, color="#333333", linestyle=":", linewidth=1.2, alpha=0.7
                            )
                            ax.text(
                                val, b_lo, f" {feat}≈{val:.3g}", va="bottom",
                                fontsize=7, color="#333333",
                            )
                        else:
                            ax.axhline(
                                val, color="#333333", linestyle=":", linewidth=1.2, alpha=0.7
                            )
                            ax.text(
                                a_lo, val, f" {feat}≈{val:.3g}", va="bottom",
                                fontsize=7, color="#333333",
                            )

        return shape, warnings, xx, yy, zz

    def _plot_sliced_interaction_curves(
        self,
        cont_feat: str,
        cat_feat: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        min_support: int,
        ax: Any,
    ) -> tuple[str, list[str]]:
        cont_raw = pd.to_numeric(self._X[cont_feat], errors="coerce").to_numpy(dtype=float)[val_idx]
        cat_raw = self._X[cat_feat].astype(str).to_numpy()[val_idx]
        valid = np.isfinite(cont_raw) & np.array([c != "" for c in cat_raw], dtype=bool)
        cont_arr, cat_arr, net_v = cont_raw[valid], cat_raw[valid], net[valid]

        cont_series = pd.Series(cont_arr)
        try:
            cont_bins = pd.qcut(cont_series, q=8, duplicates="drop")
            bin_cats = list(cont_bins.cat.categories)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data for sliced curves", ha="center", va="center")
            ax.axis("off")
            return "unclear", ["Insufficient data for sliced curves"]

        unique_cats, counts = np.unique(cat_arr, return_counts=True)
        top_levels = unique_cats[np.argsort(-counts)][:6]

        colors = ["#2C7FB8", "#E34234", "#2E7D32", "#F28C00", "#7B2D8B", "#A0522D"]
        x = np.arange(len(bin_cats))
        warnings: list[str] = []

        for i, level in enumerate(top_levels):
            lvl_mask = cat_arr == level
            count = int(lvl_mask.sum())
            alpha = 0.35 if count < min_support else 0.9
            if count < min_support:
                warnings.append(f"{cat_feat}={level!r} has low support (n={count})")

            bin_arr = cont_bins.to_numpy()
            means = [
                float(net_v[lvl_mask & (bin_arr == b)].mean())
                if int((lvl_mask & (bin_arr == b)).sum()) >= 3
                else float("nan")
                for b in bin_cats
            ]
            ax.plot(
                x, means, marker="o", markersize=5, linewidth=1.8, alpha=alpha,
                label=f"{level} (n={count})", color=colors[i % len(colors)],
            )

        ax.axhline(0.0, linestyle="--", color=_ZERO_LINE_COLOR, linewidth=1.2, label="Zero effect")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bin_cats], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{cont_feat} (quantile bins)")
        ax.set_ylabel("Mean net interaction effect")
        ax.set_title(
            f"Sliced Interaction Curves: {cont_feat} by {cat_feat}\n"
            "(net effect = depth-2 − depth-1 prediction)",
            fontsize=10,
        )
        ax.legend(title=cat_feat, fontsize=8, framealpha=0.7)
        return "category_specific_slope", warnings

    def _plot_support_heatmap(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        min_support: int,
        ax: Any,
    ) -> tuple[str, list[str]]:
        import matplotlib.pyplot as plt

        cats_a = self._X[feat_a].astype(str).value_counts().head(8).index.tolist()
        cats_b = self._X[feat_b].astype(str).value_counts().head(8).index.tolist()
        a_arr = self._X[feat_a].astype(str).to_numpy()[val_idx]
        b_arr = self._X[feat_b].astype(str).to_numpy()[val_idx]

        grid_net = np.full((len(cats_a), len(cats_b)), float("nan"))
        grid_cnt = np.zeros((len(cats_a), len(cats_b)), dtype=int)
        warnings: list[str] = []

        for i, ca in enumerate(cats_a):
            for j, cb in enumerate(cats_b):
                mask = (a_arr == ca) & (b_arr == cb)
                cnt = int(mask.sum())
                grid_cnt[i, j] = cnt
                if cnt > 0:
                    grid_net[i, j] = float(net[mask].mean())
                if 0 < cnt < min_support:
                    warnings.append(f"Low support: {feat_a}={ca!r}, {feat_b}={cb!r} (n={cnt})")

        vmax = float(np.nanmax(np.abs(grid_net))) if not np.isnan(grid_net).all() else 1.0
        im = ax.imshow(
            grid_net, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", origin="upper"
        )
        plt.colorbar(im, ax=ax, label="Mean net interaction effect")

        for i in range(len(cats_a)):
            for j in range(len(cats_b)):
                cnt = grid_cnt[i, j]
                if cnt > 0:
                    color = "red" if cnt < min_support else "black"
                    ax.text(j, i, f"n={cnt}", ha="center", va="center", fontsize=7, color=color)

        ax.set_xticks(range(len(cats_b)))
        ax.set_xticklabels(cats_b, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats_a)))
        ax.set_yticklabels(cats_a, fontsize=8)
        ax.set_xlabel(feat_b)
        ax.set_ylabel(feat_a)
        ax.set_title(
            f"Interaction Heatmap: {feat_a} × {feat_b}\n(red n = low support)", fontsize=10
        )
        return "cell_effect", warnings

    def _plot_interaction_bubbles(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        cats_a = self._X[feat_a].astype(str).value_counts().head(8).index.tolist()
        cats_b = self._X[feat_b].astype(str).value_counts().head(8).index.tolist()
        a_vals = self._X[feat_a].astype(str).iloc[val_idx].to_numpy()
        b_vals = self._X[feat_b].astype(str).iloc[val_idx].to_numpy()

        xs: list[int] = []
        ys: list[int] = []
        sizes: list[float] = []
        colors_list: list[float] = []
        for i, ca in enumerate(cats_a):
            for j, cb in enumerate(cats_b):
                mask = (a_vals == ca) & (b_vals == cb)
                count = int(mask.sum())
                if count > 0:
                    xs.append(j)
                    ys.append(i)
                    sizes.append(max(30.0, count * 5.0))
                    colors_list.append(float(net[mask].mean()))

        if not xs:
            ax.text(0.5, 0.5, "No data for bubble chart", ha="center", va="center")
            ax.axis("off")
            return

        vmax = float(np.nanmax(np.abs(colors_list))) if colors_list else 1.0
        sc = ax.scatter(
            xs, ys, c=colors_list, s=sizes, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.8
        )
        plt.colorbar(sc, ax=ax, label="Mean interaction effect (depth-2 − depth-1)")
        ax.set_xticks(range(len(cats_b)))
        ax.set_xticklabels(cats_b, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats_a)))
        ax.set_yticklabels(cats_a, fontsize=8)
        ax.set_xlabel(feat_b)
        ax.set_ylabel(feat_a)
        ax.set_title("Interaction Effect (bubble size = count)")

    def _plot_scatter_interaction(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        vals_a = pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        vals_b = pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        vmax = float(np.nanmax(np.abs(net))) if len(net) > 0 else 1.0
        sc = ax.scatter(
            vals_a, vals_b, c=net, cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, alpha=0.6, s=18, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Interaction effect (depth-2 − depth-1)")
        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        ax.set_title("Interaction Effect")

    def _plot_conditional_curves(self, cont_feat: str, cat_feat: str, ax: Any) -> None:
        y_vals = self._y.to_numpy(dtype=float)
        cont_vals = pd.to_numeric(self._X[cont_feat], errors="coerce")
        cat_vals = self._X[cat_feat].astype(str)

        try:
            cont_bins = pd.qcut(cont_vals, q=6, duplicates="drop")
            bin_cats = list(cont_bins.cat.categories)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data for conditional curves", ha="center", va="center")
            ax.axis("off")
            return

        colors = ["#2C7FB8", "#E34234", "#2E7D32", "#F28C00", "#7B2D8B"]
        x = np.arange(len(bin_cats))
        for i, cat in enumerate(cat_vals.value_counts().head(5).index):
            cat_mask = cat_vals.eq(cat)
            means = [
                float(y_vals[cat_mask & cont_bins.eq(b)].mean())
                if (cat_mask & cont_bins.eq(b)).sum() >= 5
                else float("nan")
                for b in bin_cats
            ]
            ax.plot(x, means, marker="o", label=str(cat), color=colors[i % len(colors)])

        ax.axhline(0.0, linestyle=":", color=_ZERO_LINE_COLOR, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bin_cats], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{cont_feat} (quantile bins)")
        ax.set_ylabel("Mean y")
        ax.set_title(f"Conditional Effect: {cont_feat} by {cat_feat}")
        ax.legend(title=cat_feat, fontsize=8, framealpha=0.7)

    def _plot_target_scatter(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        vals_a = pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        vals_b = pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        y_vals = self._y.iloc[val_idx].to_numpy(dtype=float)
        sc = ax.scatter(vals_a, vals_b, c=y_vals, cmap="YlOrRd", alpha=0.6, s=18, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Target value")
        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        ax.set_title("Target Distribution")

    def _plot_target_conditional_curves(self, cont_feat: str, cat_feat: str, ax: Any) -> None:
        y_vals = self._y.to_numpy(dtype=float)
        cont_vals = pd.to_numeric(self._X[cont_feat], errors="coerce")
        cat_vals = self._X[cat_feat].astype(str)
        try:
            cont_bins = pd.qcut(cont_vals, q=6, duplicates="drop")
            bin_cats = list(cont_bins.cat.categories)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data for target curves", ha="center", va="center")
            ax.axis("off")
            return

        colors = ["#2C7FB8", "#E34234", "#2E7D32", "#F28C00", "#7B2D8B"]
        x = np.arange(len(bin_cats))
        for i, cat in enumerate(cat_vals.value_counts().head(5).index):
            cat_mask = cat_vals.eq(cat)
            means = [
                float(y_vals[cat_mask & cont_bins.eq(b)].mean())
                if (cat_mask & cont_bins.eq(b)).sum() >= 5
                else float("nan")
                for b in bin_cats
            ]
            ax.plot(x, means, marker="o", label=str(cat), color=colors[i % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bin_cats], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{cont_feat} (quantile bins)")
        ax.set_ylabel("Mean target value")
        ax.set_title(f"Target: {cont_feat} by {cat_feat}")
        ax.legend(title=cat_feat, fontsize=8, framealpha=0.7)

    def _plot_target_bubbles(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        cats_a = self._X[feat_a].astype(str).value_counts().head(8).index.tolist()
        cats_b = self._X[feat_b].astype(str).value_counts().head(8).index.tolist()
        a_vals = self._X[feat_a].astype(str).iloc[val_idx].to_numpy()
        b_vals = self._X[feat_b].astype(str).iloc[val_idx].to_numpy()
        y_vals = self._y.iloc[val_idx].to_numpy(dtype=float)

        xs: list[int] = []
        ys: list[int] = []
        sizes: list[float] = []
        colors_list: list[float] = []
        for i, ca in enumerate(cats_a):
            for j, cb in enumerate(cats_b):
                mask = (a_vals == ca) & (b_vals == cb)
                count = int(mask.sum())
                if count > 0:
                    xs.append(j)
                    ys.append(i)
                    sizes.append(max(30.0, count * 5.0))
                    colors_list.append(float(y_vals[mask].mean()))

        if not xs:
            ax.text(0.5, 0.5, "No data for bubble chart", ha="center", va="center")
            ax.axis("off")
            return

        sc = ax.scatter(xs, ys, c=colors_list, s=sizes, cmap="YlOrRd", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Mean target value")
        ax.set_xticks(range(len(cats_b)))
        ax.set_xticklabels(cats_b, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats_a)))
        ax.set_yticklabels(cats_a, fontsize=8)
        ax.set_xlabel(feat_b)
        ax.set_ylabel(feat_a)
        ax.set_title("Target Distribution (bubble size = count)")

    def _plot_lift_dist(self, pair_boot: pd.DataFrame, ax: Any) -> None:
        lift = pair_boot["interaction_lift"].dropna()
        null_lift = pair_boot["null_lift"].dropna()
        all_vals = pd.concat([lift, null_lift]).dropna()
        if all_vals.empty:
            ax.text(0.5, 0.5, "No bootstrap data", ha="center", va="center")
            ax.axis("off")
            return

        lo, hi = float(all_vals.min()), float(all_vals.max())
        if lo == hi:
            lo, hi = lo - 0.01, hi + 0.01
        bins = np.linspace(lo, hi, 20)

        ax.hist(null_lift, bins=bins, color=_NULL_COLOR, alpha=0.65, label="Null lift")
        ax.hist(lift, bins=bins, color=_FEATURE_COLOR, alpha=0.70, label="Interaction lift")
        if len(lift):
            mean_lift = float(lift.mean())
            ax.axvline(
                mean_lift, color=_FEATURE_COLOR, linewidth=2.0, label=f"Mean: {mean_lift:.4f}"
            )
        ax.axvline(0.0, color=_ZERO_LINE_COLOR, linewidth=1.2, linestyle=":", label="Zero")
        if "beats_null" in pair_boot.columns:
            beat_rate = float(pair_boot["beats_null"].mean())
            ax.text(
                0.97, 0.97, f"Null beat rate: {beat_rate:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#333333",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "alpha": 0.85, "edgecolor": "#cccccc"},
            )
        ax.set_xlabel("Interaction lift (depth-2 R² − depth-1 R²)")
        ax.set_ylabel("Bootstrap run count")
        ax.set_title("Interaction Lift vs Null")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    def _suggest_candidate_terms(
        self,
        feat_a: str,
        feat_b: str,
        pair_type: str,
        shape: str,
        xx: np.ndarray[Any, Any] | None,
        yy: np.ndarray[Any, Any] | None,
        zz: np.ndarray[Any, Any] | None,
        materiality_threshold: float,
    ) -> str:
        if shape == "local_threshold" and pair_type == "continuous_continuous" and zz is not None:
            pos_mask = zz > materiality_threshold
            if pos_mask.any():
                x_thresh = float(np.nanpercentile(xx[pos_mask], 20))  # type: ignore[index]
                y_thresh = float(np.nanpercentile(yy[pos_mask], 80))  # type: ignore[index]
                return f"I({feat_a} > {x_thresh:.4g})\n  × I({feat_b} < {y_thresh:.4g})"
            return f"threshold({feat_a}) × threshold({feat_b})"
        if shape == "smooth_surface" and pair_type == "continuous_continuous":
            return f"spline({feat_a}) × spline({feat_b})\nor tensor product spline"
        if shape == "category_specific_slope":
            cat_feat = feat_a if feat_a in self._categorical_features else feat_b
            cont_feat = feat_b if feat_a in self._categorical_features else feat_a
            return (
                f"{cat_feat}_group\n  × spline({cont_feat})\nor"
                f"\n  × I({cont_feat} > threshold)"
            )
        if shape == "cell_effect":
            return f"{feat_a} × {feat_b}\n(category interaction terms)"
        return "Inspect surface;\nsignal may be weak"

    def _render_recommendation_panel(
        self,
        feat_a: str,
        feat_b: str,
        pair_type: str,
        shape: str,
        warnings: list[str],
        materiality_threshold: float,
        summary_row: pd.Series | None,
        xx: np.ndarray[Any, Any] | None,
        yy: np.ndarray[Any, Any] | None,
        zz: np.ndarray[Any, Any] | None,
        ax: Any,
    ) -> None:
        ax.axis("off")

        if summary_row is not None:
            lift = float(summary_row["mean_interaction_lift"])
            nbr = float(summary_row["interaction_null_beat_rate"])
            plr_raw = summary_row.get("positive_lift_rate", float("nan"))
            plr = float(plr_raw) if plr_raw is not None else float("nan")
            metrics = (
                f"Lift: {lift:.4f}\nNull beat rate: {nbr:.0%}\nPositive lift rate: {plr:.0%}"
            )
        else:
            metrics = ""

        shape_labels = {
            "local_threshold": "Local threshold interaction",
            "smooth_surface": "Smooth continuous surface",
            "category_specific_slope": "Category-specific slope",
            "cell_effect": "Cell-level cat × cat effect",
            "unclear": "Unclear / weak signal",
        }
        candidate = self._suggest_candidate_terms(
            feat_a, feat_b, pair_type, shape, xx, yy, zz, materiality_threshold
        )

        lines = [
            metrics, "",
            f"Shape:\n  {shape_labels.get(shape, shape)}", "",
            f"Candidate term:\n  {candidate}",
        ]
        if warnings:
            capped = warnings[:3]
            warn_str = "Warnings:\n" + "\n".join(f"  • {w}" for w in capped)
            if len(warnings) > 3:
                warn_str += f"\n  (+{len(warnings) - 3} more)"
            lines += ["", warn_str]

        ax.text(
            0.05, 0.95, "\n".join(lines).strip(),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace", linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f9f9f9", "alpha": 0.95,
                  "edgecolor": "#cccccc"},
        )
