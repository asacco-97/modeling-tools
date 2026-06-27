from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

from pe_tools.signal_finder.v2 import (
    CustomSplitLike,
    ModelType,
    SeriesLike,
    V2SplitStrategy,
    _normalize_custom_splits,
    _oof_r2,
    _SplitSpec,
)

_FEATURE_COLOR = "#2C7FB8"
_NULL_COLOR = "#9E9E9E"
_ZERO_LINE_COLOR = "#424242"


def _require_fitted(df: pd.DataFrame, attr: str) -> None:
    if df.empty:
        raise RuntimeError(f"InteractionFinder must be fitted before accessing {attr}")


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


@dataclass
class InteractionFinder:
    """Bootstrap-based pairwise interaction detector for residual signal.

    For each pair of candidate features (A, B), fits a depth-1 (additive baseline) and a
    depth-2 (interaction-capable) ensemble model on residuals across bootstrap splits.
    Interaction lift = R²(depth-2) − R²(depth-1). A permuted-second-feature null produces
    ``interaction_null_beat_rate`` to guard against spurious lifts in small or noisy data.

    Designed to be used after :class:`ResidualSignalFinderV2` to diagnose which top-ranked
    feature pairs show true interaction effects beyond their additive main effects::

        top_features = v2_finder.summary_.head(10)["feature"].tolist()
        residuals = y_true - base_pred

        finder = InteractionFinder(n_bootstraps=50)
        finder.fit(X, residuals, candidate_features=top_features)
        print(finder.interaction_summary_)
        finder.plot_interactions(top_n=3)

    # TODO: consider replacing the net interaction surface plot with 2D ALE in a future version.
    """

    n_bootstraps: int = 50
    test_size: float = 0.2
    split_strategy: V2SplitStrategy = "bootstrap"
    n_splits: int = 5
    model_type: ModelType = "xgboost"
    random_state: int = 42
    max_candidate_features: int = 15
    r2_epsilon: float = 0.001
    model_params: dict[str, Any] | None = None

    interaction_summary_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    interaction_bootstrap_results_: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    _X: pd.DataFrame = field(init=False, default_factory=pd.DataFrame, repr=False)
    _residuals: pd.Series = field(init=False, default_factory=pd.Series, repr=False)
    _candidate_features: list[str] = field(init=False, default_factory=list, repr=False)
    _categorical_features: set[str] = field(init=False, default_factory=set, repr=False)
    _y_true: pd.Series | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be at least 1")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        if self.split_strategy not in {"bootstrap", "repeated_kfold", "group_kfold"}:
            raise ValueError(
                "split_strategy must be one of: bootstrap, repeated_kfold, group_kfold"
            )
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.max_candidate_features < 2:
            raise ValueError("max_candidate_features must be at least 2")
        if self.model_type not in {"xgboost", "random_forest"}:
            raise ValueError("model_type must be one of: xgboost, random_forest")

    def fit(
        self,
        X: pd.DataFrame,
        residuals: SeriesLike,
        candidate_features: Sequence[str],
        categorical_features: Sequence[str] | None = None,
        *,
        y_true: SeriesLike | None = None,
        splits: Sequence[CustomSplitLike] | None = None,
        group_col: Any = None,
    ) -> InteractionFinder:
        """Fit pairwise interaction diagnostics on pre-screened candidate features.

        Args:
            X: Feature matrix. Categorical columns should use pd.Categorical dtype
               or be listed explicitly in ``categorical_features``.
            residuals: Pre-computed residuals (y − base_pred). Must be finite.
            candidate_features: Features to evaluate pairwise. Must be ≤ max_candidate_features.
            categorical_features: Explicit list of categorical feature names (overrides dtype
               inference).
            y_true: Original target values (same index as X). When provided, enables the
               target-view panel in ``plot_interactions()``, showing where actual target
               events concentrate in the feature space alongside the interaction effect.
            splits: Custom splits — list of dicts with train/validation keys or
               (train_idx, val_idx) tuples.
            group_col: Group labels when split_strategy='group_kfold'.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        features_list = list(candidate_features)
        if len(features_list) > self.max_candidate_features:
            raise ValueError(
                f"candidate_features has {len(features_list)} features, but "
                f"max_candidate_features={self.max_candidate_features}. Pre-screen features "
                f"to at most {self.max_candidate_features} before calling fit()."
            )
        missing = [f for f in features_list if f not in X.columns]
        if missing:
            raise ValueError(f"candidate_features not found in X: {missing}")

        residual_array = np.asarray(residuals, dtype=float)
        if len(residual_array) != len(X):
            raise ValueError("residuals must have the same length as X")
        if not np.isfinite(residual_array).all():
            raise ValueError("residuals must be finite")
        residual_series = pd.Series(residual_array, index=X.index, name="residual")

        explicit_cats = set(categorical_features or [])
        self._categorical_features = {
            f for f in features_list if _is_categorical_feature(X[f], explicit_cats)
        }
        self._X = X[features_list].copy()
        self._residuals = residual_series
        self._candidate_features = features_list
        self._y_true = (
            pd.Series(np.asarray(y_true, dtype=float), index=X.index, name="y_true")
            if y_true is not None
            else None
        )

        group_series: pd.Series | None = None
        if group_col is not None:
            group_series = pd.Series(np.asarray(group_col), index=X.index, name="group")

        split_specs = self._make_splits(X, group_series, splits)
        if not split_specs:
            raise ValueError("No valid train/validation splits were created")

        pairs = list(combinations(features_list, 2))
        rng = np.random.default_rng(self.random_state)
        rows: list[dict[str, Any]] = []

        for split_spec in split_specs:
            train_idx = split_spec.train_idx
            val_idx = split_spec.validation_idx
            train_y = residual_series.iloc[train_idx].to_numpy(dtype=float)
            val_y = residual_series.iloc[val_idx].to_numpy(dtype=float)
            null_mean = float(np.mean(train_y))

            for feat_a, feat_b in pairs:
                is_cat_a = feat_a in self._categorical_features
                is_cat_b = feat_b in self._categorical_features
                enc_a = _encode_feature(self._X[feat_a], is_cat_a, train_idx)
                enc_b = _encode_feature(self._X[feat_b], is_cat_b, train_idx)

                X_tr = pd.DataFrame({"feature_a": enc_a[train_idx], "feature_b": enc_b[train_idx]})
                X_vl = pd.DataFrame({"feature_a": enc_a[val_idx], "feature_b": enc_b[val_idx]})

                # Permuted null: shuffle B independently within train and validation
                X_tr_null = X_tr.copy()
                X_vl_null = X_vl.copy()
                X_tr_null["feature_b"] = rng.permutation(enc_b[train_idx])
                X_vl_null["feature_b"] = rng.permutation(enc_b[val_idx])

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
        self.interaction_summary_ = self._build_summary()
        return self

    def plot_interactions(self, top_n: int = 5) -> dict[str, Figure]:
        """Return one diagnostic figure per top-N ranked interaction pair.

        Each figure has three panels:
        - Left: interaction effect — scatterplot (or conditional curves / bubble chart) colored
          by the per-point net effect (depth-2 prediction minus depth-1 prediction). Red areas
          have positive interaction; blue areas have negative interaction.
        - Centre: target view — same axes but colored by actual ``y_true`` values, so the
          interaction effect can be read against where target events concentrate. Requires
          ``y_true`` to be passed to ``fit()``.
        - Right: bootstrap interaction lift vs null lift distribution.
        """
        _require_fitted(self.interaction_summary_, "interaction_summary_")
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

    # --- private helpers ---

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
        """Fit depth-1 and depth-2 models on 80% of data; return (val_idx, net_effect)."""
        n = len(self._X)
        rng = np.random.default_rng(self.random_state)
        train_size = max(1, int(round(0.8 * n)))
        train_idx = np.sort(rng.choice(np.arange(n), size=train_size, replace=False))
        val_idx = np.setdiff1d(np.arange(n), train_idx)

        enc_a = _encode_feature(self._X[feat_a], is_cat_a, train_idx)
        enc_b = _encode_feature(self._X[feat_b], is_cat_b, train_idx)
        X_tr = pd.DataFrame({"feature_a": enc_a[train_idx], "feature_b": enc_b[train_idx]})
        X_vl = pd.DataFrame({"feature_a": enc_a[val_idx], "feature_b": enc_b[val_idx]})
        train_y = self._residuals.iloc[train_idx].to_numpy(dtype=float)

        m1 = self._build_model(depth=1, seed=self.random_state)
        m2 = self._build_model(depth=2, seed=self.random_state)
        m1.fit(X_tr, train_y)
        m2.fit(X_tr, train_y)
        net = (
            np.asarray(m2.predict(X_vl), dtype=float)
            - np.asarray(m1.predict(X_vl), dtype=float)
        )
        return val_idx, net

    def _plot_scatter_interaction(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        net: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        vals_a = (
            pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        )
        vals_b = (
            pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        )
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
        residuals = self._residuals.to_numpy(dtype=float)
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
                float(residuals[cat_mask & cont_bins.eq(b)].mean())
                if (cat_mask & cont_bins.eq(b)).sum() >= 5
                else np.nan
                for b in bin_cats
            ]
            ax.plot(x, means, marker="o", label=str(cat), color=colors[i % len(colors)])

        ax.axhline(0.0, linestyle=":", color=_ZERO_LINE_COLOR, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bin_cats], rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(f"{cont_feat} (quantile bins)")
        ax.set_ylabel("Mean residual")
        ax.set_title(f"Conditional Effect: {cont_feat} by {cat_feat}")
        ax.legend(title=cat_feat, fontsize=8, framealpha=0.7)

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
        colors: list[float] = []
        for i, ca in enumerate(cats_a):
            for j, cb in enumerate(cats_b):
                mask = (a_vals == ca) & (b_vals == cb)
                count = int(mask.sum())
                if count > 0:
                    xs.append(j)
                    ys.append(i)
                    sizes.append(max(30.0, count * 5.0))
                    colors.append(float(net[mask].mean()))

        if not xs:
            ax.text(0.5, 0.5, "No data for bubble chart", ha="center", va="center")
            ax.axis("off")
            return

        vmax = float(np.nanmax(np.abs(colors))) if colors else 1.0
        sc = ax.scatter(
            xs, ys, c=colors, s=sizes, cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, alpha=0.8,
        )
        plt.colorbar(sc, ax=ax, label="Mean interaction effect (depth-2 − depth-1)")
        ax.set_xticks(range(len(cats_b)))
        ax.set_xticklabels(cats_b, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cats_a)))
        ax.set_yticklabels(cats_a, fontsize=8)
        ax.set_xlabel(feat_b)
        ax.set_ylabel(feat_a)
        ax.set_title("Interaction Effect (bubble size = count)")

    def _plot_target_scatter(
        self,
        feat_a: str,
        feat_b: str,
        val_idx: np.ndarray[Any, Any],
        ax: Any,
    ) -> None:
        import matplotlib.pyplot as plt

        if self._y_true is None:
            ax.text(
                0.5, 0.5, "Pass y_true= to fit() to enable this view",
                ha="center", va="center", color="grey", fontsize=10,
            )
            ax.set_title("Target View (unavailable)")
            ax.axis("off")
            return

        vals_a = (
            pd.to_numeric(self._X[feat_a], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        )
        vals_b = (
            pd.to_numeric(self._X[feat_b], errors="coerce").iloc[val_idx].to_numpy(dtype=float)
        )
        y_vals = self._y_true.iloc[val_idx].to_numpy(dtype=float)
        sc = ax.scatter(
            vals_a, vals_b, c=y_vals, cmap="YlOrRd", alpha=0.6, s=18, rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="Target value")
        ax.set_xlabel(feat_a)
        ax.set_ylabel(feat_b)
        ax.set_title("Target Distribution")

    def _plot_target_conditional_curves(
        self, cont_feat: str, cat_feat: str, ax: Any
    ) -> None:
        if self._y_true is None:
            ax.text(
                0.5, 0.5, "Pass y_true= to fit() to enable this view",
                ha="center", va="center", color="grey", fontsize=10,
            )
            ax.set_title("Target View (unavailable)")
            ax.axis("off")
            return

        y_vals = self._y_true.to_numpy(dtype=float)
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
                else np.nan
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

        if self._y_true is None:
            ax.text(
                0.5, 0.5, "Pass y_true= to fit() to enable this view",
                ha="center", va="center", color="grey", fontsize=10,
            )
            ax.set_title("Target View (unavailable)")
            ax.axis("off")
            return

        cats_a = self._X[feat_a].astype(str).value_counts().head(8).index.tolist()
        cats_b = self._X[feat_b].astype(str).value_counts().head(8).index.tolist()
        a_vals = self._X[feat_a].astype(str).iloc[val_idx].to_numpy()
        b_vals = self._X[feat_b].astype(str).iloc[val_idx].to_numpy()
        y_vals = self._y_true.iloc[val_idx].to_numpy(dtype=float)

        xs: list[int] = []
        ys: list[int] = []
        sizes: list[float] = []
        colors: list[float] = []
        for i, ca in enumerate(cats_a):
            for j, cb in enumerate(cats_b):
                mask = (a_vals == ca) & (b_vals == cb)
                count = int(mask.sum())
                if count > 0:
                    xs.append(j)
                    ys.append(i)
                    sizes.append(max(30.0, count * 5.0))
                    colors.append(float(y_vals[mask].mean()))

        if not xs:
            ax.text(0.5, 0.5, "No data for bubble chart", ha="center", va="center")
            ax.axis("off")
            return

        sc = ax.scatter(xs, ys, c=colors, s=sizes, cmap="YlOrRd", alpha=0.8)
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
                0.97, 0.97,
                f"Null beat rate: {beat_rate:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "#cccccc",
                },
            )
        ax.set_xlabel("Interaction lift (depth-2 R² − depth-1 R²)")
        ax.set_ylabel("Bootstrap run count")
        ax.set_title("Interaction Lift vs Null")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

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

    def _make_splits(
        self,
        X: pd.DataFrame,
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
        for _ in range(self.n_bootstraps):
            tr = np.sort(rng.choice(positions, size=train_size, replace=False))
            vl = np.setdiff1d(positions, tr)
            if len(vl) > 0:
                splits.append(_SplitSpec(split_id=len(splits), train_idx=tr, validation_idx=vl))
        return splits

    def _build_summary(self) -> pd.DataFrame:
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
                    float((lift > self.r2_epsilon).mean()) if len(lift) else np.nan
                ),
                "interaction_null_beat_rate": (
                    float(group["beats_null"].mean()) if len(group) else np.nan
                ),
                "mean_depth2_r2": float(group["depth2_r2"].mean()),
                "mean_depth1_r2": float(group["depth1_r2"].mean()),
            })

        summary = pd.DataFrame(rows).sort_values(
            "mean_interaction_lift", ascending=False, ignore_index=True
        )
        summary["rank"] = np.arange(1, len(summary) + 1)
        return summary
