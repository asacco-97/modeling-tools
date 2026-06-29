from __future__ import annotations

import base64
import html as _html
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
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


def _fs_figure_to_base64(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@dataclass
class FeatureSelector:
    """Bootstrap-based feature selector that works on raw (X, y).

    Evaluates each feature independently across bootstrap or k-fold splits,
    comparing lift against a permuted-null baseline. After fit(), always computes
    pairwise Spearman correlations among selected features and groups them into
    clusters.

    After .fit(), call .find_interactions() to detect pairwise interaction
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
    correlation_threshold: float = 0.20
    model_params: dict[str, Any] | None = None
    verbose: bool = False

    summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    bootstrap_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    selected_features_: list[str] = field(init=False, default_factory=list)
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

        if len(self.selected_features_) >= 2:
            self.feature_correlation_ = self._compute_feature_correlation()
        else:
            self.feature_correlation_ = pd.DataFrame()

        print(f"FeatureSelector: {len(self.selected_features_)} features selected")
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
        metric_label = "Gini (2·AUC − 1)" if self._is_binary else "Spearman ρ"
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

    def plot_feature_response(self, top_n: int = 10) -> Figure:
        """Grid of univariate response curves for the top-N selected features.

        For each feature shows mean(y) binned by feature value, with observation
        counts as a bar strip below. Continuous features use 10 quantile bins;
        categorical features show top-10 levels sorted by mean(y) descending.

        Useful for identifying non-linearities before feature engineering.
        """
        _require_fitted(self, "plot_feature_response()")

        import matplotlib.pyplot as plt

        feats = [
            f for f in self.summary_["feature"].tolist()
            if f in self.selected_features_
        ][:top_n]
        if not feats:
            raise RuntimeError("No selected features to plot.")

        ncols = min(3, len(feats))
        nrows = (len(feats) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )

        y_arr = self._y.to_numpy(dtype=float)
        y_label = "Mean target (y)"

        for idx, feat in enumerate(feats):
            ax = axes[idx // ncols][idx % ncols]
            is_cat = feat in self._categorical_features
            vals = self._X[feat]

            if is_cat:
                str_vals = vals.astype(str)
                counts = str_vals.value_counts()
                top_levels = counts.head(10).index.tolist()
                means = [
                    float(y_arr[str_vals.eq(lv)].mean()) for lv in top_levels
                ]
                cnts = [int(counts[lv]) for lv in top_levels]
                order = sorted(range(len(top_levels)), key=lambda i: means[i], reverse=True)
                labels = [top_levels[i] for i in order]
                bar_means = [means[i] for i in order]
                bar_cnts = [cnts[i] for i in order]

                x = np.arange(len(labels))
                ax.bar(x, bar_means, color=_FEATURE_COLOR, alpha=0.75)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

                ax2 = ax.twinx()
                ax2.bar(x, bar_cnts, color="#CCCCCC", alpha=0.35, zorder=0)
                ax2.set_ylabel("Count", fontsize=8, color="#888888")
                ax2.tick_params(axis="y", labelsize=7, labelcolor="#888888")

            else:
                num_vals = pd.to_numeric(vals, errors="coerce")
                try:
                    binned = pd.qcut(num_vals, q=10, duplicates="drop")
                except Exception:
                    ax.text(0.5, 0.5, "Cannot bin", ha="center", va="center")
                    ax.set_title(feat, fontsize=9)
                    continue

                bin_cats = list(binned.cat.categories)
                means = [
                    float(y_arr[binned.eq(b)].mean())
                    if int(binned.eq(b).sum()) > 0
                    else float("nan")
                    for b in bin_cats
                ]
                cnts = [int(binned.eq(b).sum()) for b in bin_cats]
                x = np.arange(len(bin_cats))

                ax.plot(x, means, color=_FEATURE_COLOR, linewidth=2, marker="o", markersize=5)
                ax.fill_between(x, means, alpha=0.12, color=_FEATURE_COLOR)

                ax2 = ax.twinx()
                ax2.bar(x, cnts, color="#CCCCCC", alpha=0.35, zorder=0)
                ax2.set_ylabel("Count", fontsize=8, color="#888888")
                ax2.tick_params(axis="y", labelsize=7, labelcolor="#888888")

                ax.set_xticks(x)
                ax.set_xticklabels(
                    [str(b) for b in bin_cats], rotation=45, ha="right", fontsize=7
                )

            ax.set_ylabel(y_label, fontsize=8)
            ax.set_title(feat, fontsize=9, fontweight="bold")
            ax.spines["top"].set_visible(False)

        # Hide unused axes
        for idx in range(len(feats), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        task_label = "binary" if self._is_binary else "regression"
        fig.suptitle(
            f"Feature Response Curves ({task_label})  —  top {len(feats)} selected features",
            fontsize=11,
            y=1.01,
        )
        fig.tight_layout()
        return fig

    def plot_feature_correlations(self) -> Figure:
        """Heatmap of Spearman correlations among selected features, ordered by cluster.

        Raises:
            RuntimeError: If fit() was not called or no features were selected.
        """
        _require_fitted(self, "plot_feature_correlations()")
        if self.feature_correlation_.empty:
            raise RuntimeError(
                "No features selected or no correlation to display. "
                "Ensure fit() produced at least 2 selected features."
            )

        import matplotlib.pyplot as plt

        feats = self.feature_correlation_.index.tolist()
        corr = self.feature_correlation_.astype(float)
        n = len(corr)

        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
        values = corr.to_numpy(dtype=float)

        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        display_values = np.where(upper_mask, np.nan, values)

        vmax = float(np.nanmax(np.abs(np.where(np.isnan(display_values), 0.0, display_values)))) or 1.0
        im = ax.imshow(display_values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman rho")

        for i in range(n):
            for j in range(i + 1):
                val = display_values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        ax.set_xticks(range(n))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_title(
            f"Feature Correlation  |rho| >= {self.correlation_threshold}",
            fontsize=10,
        )
        fig.tight_layout()
        return fig

    def plot_interactions(self, top_n: int = 5) -> dict[str, Figure]:
        """Return one diagnostic figure per top-N ranked interaction pair.

        Each figure has three panels:
        - Left: hexbin of mean interaction effect (depth-2 − depth-1), RdBu_r
        - Centre: conditional mean of y in the same feature space, YlOrRd
        - Right: bootstrap lift vs null distribution

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

    def to_html(
        self,
        path: str | None = None,
        title: str | None = None,
        top_n: int = 5,
    ) -> str:
        """Render a self-contained HTML report and optionally write it to disk.

        Sections:
        1. Summary (task, obs count, selection thresholds)
        2. Feature ranking table (top_n rows)
        3. Feature selection figure
        4. Correlation & clusters (always shown after fit)
        5. Interactions (shown if find_interactions() was called)

        Args:
            path: Optional file path to write the HTML. If None, returns the string only.
            title: Report title. Defaults to "Feature Selection Report".
            top_n: Number of top features in the ranking table and top pairs in interactions.

        Returns:
            The full HTML string.
        """
        _require_fitted(self, "to_html()")
        import matplotlib.pyplot as plt

        effective_title = title or "Feature Selection Report"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        metric_label = "Mean Gini (2·AUC−1)" if self._is_binary else "Mean Spearman ρ"
        task_label = "binary" if self._is_binary else "regression"

        exec_items: list[tuple[str, str]] = [
            ("Task", task_label),
            ("Observations", f"{len(self._X):,}"),
            ("Features evaluated", str(len(self._candidate_features))),
            ("Features selected", str(len(self.selected_features_))),
            ("Bootstrap runs", str(self.n_bootstraps)),
            ("Null beat rate threshold", f"{self.null_beat_rate_threshold:.0%}"),
            ("Positive score rate threshold", f"{self.positive_score_rate_threshold:.0%}"),
        ]

        ranking_cols = [
            "feature", "dtype", "mean_robust_metric", "std_robust_metric",
            "null_beat_rate", "positive_score_rate", "selected",
        ]
        ranking_cols = [c for c in ranking_cols if c in self.summary_.columns]
        top_summary = self.summary_.head(top_n)[ranking_cols].copy()
        col_labels = {
            "feature": "Feature",
            "dtype": "Type",
            "mean_robust_metric": metric_label,
            "std_robust_metric": "Std",
            "null_beat_rate": "Null Beat Rate",
            "positive_score_rate": "Positive Score Rate",
            "selected": "Selected",
        }

        try:
            sel_fig = self.plot_selected_features(top_n=min(top_n * 4, len(self.summary_)))
            selection_fig_b64: str | None = _fs_figure_to_base64(sel_fig)
            plt.close(sel_fig)
        except Exception:
            selection_fig_b64 = None

        corr_fig_b64: str | None = None
        if not self.feature_correlation_.empty:
            try:
                corr_fig = self.plot_feature_correlations()
                corr_fig_b64 = _fs_figure_to_base64(corr_fig)
                plt.close(corr_fig)
            except Exception:
                pass

        interaction_figs: list[tuple[str, str | None]] = []
        if self._interactions_fitted and not self.interaction_summary_.empty:
            for _, row in self.interaction_summary_.head(top_n).iterrows():
                fa, fb = str(row["feature_1"]), str(row["feature_2"])
                pair_label = f"{fa} × {fb}"
                try:
                    pair_boot = self.interaction_bootstrap_results_.loc[
                        self.interaction_bootstrap_results_["feature_1"].eq(fa)
                        & self.interaction_bootstrap_results_["feature_2"].eq(fb)
                    ]
                    fig = self._plot_pair(fa, fb, row, pair_boot)
                    img_b64: str | None = _fs_figure_to_base64(fig)
                    plt.close(fig)
                except Exception:
                    img_b64 = None
                interaction_figs.append((pair_label, img_b64))

        html_str = _build_fs_html_report(
            title=effective_title,
            generated_at=generated_at,
            exec_items=exec_items,
            ranking_df=top_summary,
            col_labels=col_labels,
            metric_label=metric_label,
            selection_fig_b64=selection_fig_b64,
            corr_fig_b64=corr_fig_b64,
            interaction_summary=(
                self.interaction_summary_ if self._interactions_fitted else None
            ),
            interaction_figs=interaction_figs,
            top_n=top_n,
        )

        if path is not None:
            import pathlib
            pathlib.Path(path).write_text(html_str, encoding="utf-8")

        return html_str

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

    # --- Private: correlation ---

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
        group_series: pd.Series | None,
        custom_splits: list[CustomSplitLike] | None,
    ) -> list[_SplitSpec]:
        n_obs = len(X)
        positions = np.arange(n_obs)

        if custom_splits is not None:
            return _normalize_custom_splits(custom_splits)

        if self.split_strategy == "repeated_kfold":
            splits: list[_SplitSpec] = []
            for repeat in range(max(1, self.n_bootstraps // self.n_splits)):
                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state + repeat,
                )
                for fold_idx, (tr, vl) in enumerate(kf.split(X)):
                    splits.append(
                        _SplitSpec(
                            split_id=len(splits),
                            train_idx=np.sort(tr),
                            validation_idx=np.sort(vl),
                        )
                    )
                    if len(splits) >= self.n_bootstraps:
                        return splits
            return splits

        if self.split_strategy == "group_kfold":
            if group_series is None:
                raise ValueError("group_col is required for group_kfold split strategy")
            groups = group_series.to_numpy()
            gkf = GroupKFold(n_splits=self.n_splits)
            return [
                _SplitSpec(split_id=i, train_idx=np.sort(tr), validation_idx=np.sort(vl))
                for i, (tr, vl) in enumerate(gkf.split(X, groups=groups))
            ]

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
                "n_estimators": 50,
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
            "n_estimators": 50,
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
        val_idx, net = self._one_shot_net_predict(feat_a, feat_b, is_cat_a, is_cat_b)

        if not is_cat_a and not is_cat_b:
            self._plot_interaction_hexbin(feat_a, feat_b, val_idx, net, axes[0])
            self._plot_target_hexbin(feat_a, feat_b, val_idx, axes[1])
        elif is_cat_a != is_cat_b:
            cat_feat = feat_a if is_cat_a else feat_b
            cont_feat = feat_b if is_cat_a else feat_a
            self._plot_sliced_interaction_curves(cont_feat, cat_feat, val_idx, net, 30, axes[0])
            self._plot_target_conditional_curves(cont_feat, cat_feat, axes[1])
        else:
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

    def _plot_interaction_hexbin(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        vals_a = pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(
            dtype=float
        )
        vals_b = pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(
            dtype=float
        )
        valid = np.isfinite(vals_a) & np.isfinite(vals_b) & np.isfinite(net)
        if valid.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.axis("off")
            return

        sc = ax.hexbin(
            vals_a[valid], vals_b[valid], C=net[valid],
            reduce_C_function=np.mean, gridsize=25, mincnt=5,
            cmap="RdBu_r",
        )
        arr = sc.get_array()
        vmax = float(np.nanmax(np.abs(arr))) if arr is not None and len(arr) > 0 else 1.0
        sc.set_clim(-vmax, vmax)
        plt.colorbar(sc, ax=ax, label="Mean interaction effect (depth-2 − depth-1)")
        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        ax.set_title(
            f"Interaction Effect: {feat_a} × {feat_b}\n(hexbin · min 5 obs/bin)",
            fontsize=10,
        )

    def _plot_target_hexbin(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        vals_a = pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(
            dtype=float
        )
        vals_b = pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(
            dtype=float
        )
        y_vals = self._y.iloc[val_idx].to_numpy(dtype=float)
        valid = np.isfinite(vals_a) & np.isfinite(vals_b) & np.isfinite(y_vals)
        if valid.sum() < 5:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.axis("off")
            return

        sc = ax.hexbin(
            vals_a[valid], vals_b[valid], C=y_vals[valid],
            reduce_C_function=np.mean, gridsize=25, mincnt=5,
            cmap="YlOrRd",
        )
        plt.colorbar(sc, ax=ax, label="Mean target value")
        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        ax.set_title(
            f"Conditional Mean y: {feat_a} × {feat_b}\n(hexbin · min 5 obs/bin)",
            fontsize=10,
        )

    def _plot_sliced_interaction_curves(
        self,
        cont_feat: str,
        cat_feat: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        min_support: int,
        ax: Any,
    ) -> tuple[str, list[str]]:
        cont_raw = pd.to_numeric(self._X[cont_feat], errors="coerce").to_numpy(dtype=float)[
            val_idx
        ]
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
            f"Interaction Effect: {cont_feat} by {cat_feat}\n"
            "(net effect = depth-2 − depth-1 prediction)",
            fontsize=10,
        )
        ax.legend(title=cat_feat, fontsize=8, framealpha=0.7)
        return "category_specific_slope", warnings

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
        ax.set_title(f"Conditional Mean y: {cont_feat} by {cat_feat}")
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
        ax.set_title("Conditional Mean y (bubble size = count)")

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


def _build_fs_html_report(
    title: str,
    generated_at: str,
    exec_items: list[tuple[str, str]],
    ranking_df: pd.DataFrame,
    col_labels: dict[str, str],
    metric_label: str,
    selection_fig_b64: str | None,
    corr_fig_b64: str | None,
    interaction_summary: pd.DataFrame | None,
    interaction_figs: list[tuple[str, str | None]],
    top_n: int,
) -> str:
    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      font-size: 14px; line-height: 1.65; color: #222; background: #fff;
      max-width: 1200px; margin: 0 auto; padding: 44px 36px;
    }
    .report-header { border-top: 3px solid #2C7FB8; padding-top: 22px; margin-bottom: 40px; }
    h1 { font-size: 22px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }
    .meta { font-size: 12px; color: #999; }
    h2 {
      font-size: 12px; font-weight: 700; color: #2C7FB8;
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
    .badge-selected { background: #2E7D32; color: #fff; }
    .fig-block { margin-top: 24px; }
    .fig-block img { max-width: 100%; height: auto; }
    .pair-card { margin-top: 36px; padding-top: 24px; border-top: 1px solid #e8e8e8; }
    .pair-card h3 { font-size: 15px; font-weight: 600; color: #1a1a1a; margin-bottom: 12px; }
    .pair-card img { max-width: 100%; height: auto; }
    """

    exec_rows = "".join(
        f"<tr><td>{_html.escape(k)}</td><td class='num'>{_html.escape(v)}</td></tr>"
        for k, v in exec_items
    )

    header_cells = "".join(
        f"<th>{_html.escape(col_labels.get(c, c))}</th>" for c in ranking_df.columns
    )
    numeric_cols = {"mean_robust_metric", "std_robust_metric", "positive_score_rate"}
    ranking_rows_html: list[str] = []
    for _, row in ranking_df.iterrows():
        cells: list[str] = []
        for col in ranking_df.columns:
            raw = row[col]
            val_str = str(raw) if pd.notna(raw) else ""
            if col == "null_beat_rate" and val_str:
                try:
                    rate = float(val_str)
                    badge_cls = (
                        "badge-strong" if rate > 0.75
                        else ("badge-moderate" if rate >= 0.50 else "badge-weak")
                    )
                    cells.append(
                        f"<td><span class='badge {badge_cls}'>{rate:.0%}</span></td>"
                    )
                    continue
                except ValueError:
                    pass
            if col == "selected":
                badge = (
                    "<span class='badge badge-selected'>✓</span>" if bool(raw) else ""
                )
                cells.append(f"<td>{badge}</td>")
                continue
            td_cls = " class='num'" if col in numeric_cols else ""
            if col in numeric_cols and val_str:
                try:
                    val_str = f"{float(val_str):.3f}"
                except ValueError:
                    pass
            cells.append(f"<td{td_cls}>{_html.escape(val_str)}</td>")
        ranking_rows_html.append(f"<tr>{''.join(cells)}</tr>")

    selection_img = (
        f"<div class='fig-block'>"
        f"<img src='data:image/png;base64,{selection_fig_b64}'>"
        f"</div>"
        if selection_fig_b64
        else "<p><em>Figure unavailable.</em></p>"
    )

    if corr_fig_b64:
        corr_section = (
            f"<h2>Feature Correlation</h2>"
            f"<div class='fig-block'><img src='data:image/png;base64,{corr_fig_b64}'></div>"
        )
    else:
        corr_section = (
            "<h2>Feature Correlation</h2>"
            "<p>Fewer than 2 features were selected — no correlation to compute.</p>"
        )

    interactions_section = ""
    if interaction_summary is not None and not interaction_summary.empty:
        int_cols = [
            "feature_1", "feature_2", "mean_interaction_lift",
            "positive_lift_rate", "interaction_null_beat_rate", "rank",
        ]
        int_cols = [c for c in int_cols if c in interaction_summary.columns]
        int_header = "".join(
            f"<th>{_html.escape(c.replace('_', ' ').title())}</th>" for c in int_cols
        )
        int_rows_html: list[str] = []
        for _, row in interaction_summary.head(top_n).iterrows():
            cells = []
            for col in int_cols:
                raw = row[col]
                val_str = str(raw) if pd.notna(raw) else ""
                if col == "interaction_null_beat_rate" and val_str:
                    try:
                        rate = float(val_str)
                        badge_cls = (
                            "badge-strong" if rate > 0.75
                            else ("badge-moderate" if rate >= 0.50 else "badge-weak")
                        )
                        cells.append(
                            f"<td><span class='badge {badge_cls}'>{rate:.0%}</span></td>"
                        )
                        continue
                    except ValueError:
                        pass
                num_cols_int = {"mean_interaction_lift", "positive_lift_rate"}
                td_cls = " class='num'" if col in num_cols_int or col == "rank" else ""
                if col in num_cols_int and val_str:
                    try:
                        val_str = f"{float(val_str):.4f}"
                    except ValueError:
                        pass
                cells.append(f"<td{td_cls}>{_html.escape(val_str)}</td>")
            int_rows_html.append(f"<tr>{''.join(cells)}</tr>")

        pair_cards = "".join(
            f"<div class='pair-card'>"
            f"<h3>{_html.escape(pair_label)}</h3>"
            f"<img src='data:image/png;base64,{img_b64}'>"
            f"</div>"
            if img_b64
            else f"<div class='pair-card'>"
            f"<h3>{_html.escape(pair_label)}</h3>"
            f"<p><em>Figure unavailable.</em></p></div>"
            for pair_label, img_b64 in interaction_figs
        )

        interactions_section = f"""
  <h2>Interactions — Top {top_n}</h2>
  <table>
    <thead><tr>{int_header}</tr></thead>
    <tbody>{''.join(int_rows_html)}</tbody>
  </table>
  {pair_cards}
"""

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

  <h2>Summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{exec_rows}</tbody>
  </table>

  <h2>Feature Rankings — Top {top_n}</h2>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(ranking_rows_html)}</tbody>
  </table>

  <h2>Feature Selection</h2>
  {selection_img}

  {corr_section}

  {interactions_section}
</body>
</html>"""
