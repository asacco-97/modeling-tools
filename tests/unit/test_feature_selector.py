from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from tests.fixtures.synthetic_data import make_residual_signal_data

from pe_tools.signal_finder import FeatureSelector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOOTSTRAP_COLUMNS = [
    "feature",
    "split_id",
    "r2",
    "robust_metric",
    "null_r2",
    "null_robust_metric",
    "beats_null",
]

SUMMARY_COLUMNS = [
    "feature",
    "dtype",
    "mean_r2",
    "std_r2",
    "mean_robust_metric",
    "std_robust_metric",
    "null_beat_rate",
    "positive_score_rate",
    "metric_name",
    "selected",
    "rank",
]

INTERACTION_BOOTSTRAP_COLUMNS = [
    "feature_1",
    "feature_2",
    "bootstrap_id",
    "depth1_r2",
    "depth2_r2",
    "interaction_lift",
    "null_lift",
    "beats_null",
]

INTERACTION_SUMMARY_COLUMNS = [
    "feature_1",
    "feature_2",
    "mean_interaction_lift",
    "median_interaction_lift",
    "positive_lift_rate",
    "interaction_null_beat_rate",
    "mean_depth2_r2",
    "mean_depth1_r2",
    "rank",
]


def _make_selector(
    n: int = 500,
    n_bootstraps: int = 20,
    include_interaction: bool = True,
    random_state: int = 42,
    **kwargs: object,
) -> FeatureSelector:
    X, y_true, _y_pred, *_ = make_residual_signal_data(
        n=n,
        random_state=random_state,
        include_interaction=include_interaction,
        heterogeneity=False,
    )
    selector = FeatureSelector(
        n_bootstraps=n_bootstraps,
        random_state=random_state,
        **kwargs,
    )
    selector.fit(X, y_true)
    return selector


# ---------------------------------------------------------------------------
# 1. Schema tests: bootstrap_results_
# ---------------------------------------------------------------------------


def test_bootstrap_results_has_expected_columns() -> None:
    selector = _make_selector()
    assert list(selector.bootstrap_results_.columns) == BOOTSTRAP_COLUMNS


def test_bootstrap_results_one_row_per_feature_per_split() -> None:
    n_bootstraps = 10
    selector = _make_selector(n_bootstraps=n_bootstraps)
    counts = selector.bootstrap_results_.groupby("feature")["split_id"].nunique()
    assert (counts == n_bootstraps).all()


# ---------------------------------------------------------------------------
# 2. Schema tests: summary_
# ---------------------------------------------------------------------------


def test_summary_has_expected_columns() -> None:
    selector = _make_selector()
    assert list(selector.summary_.columns) == SUMMARY_COLUMNS


def test_summary_one_row_per_feature() -> None:
    selector = _make_selector()
    X, *_ = make_residual_signal_data(n=100, random_state=42)
    assert len(selector.summary_) == len(X.columns)
    assert not selector.summary_.duplicated("feature").any()


def test_summary_rank_is_consecutive() -> None:
    selector = _make_selector()
    ranks = selector.summary_["rank"].tolist()
    assert ranks == list(range(1, len(ranks) + 1))


def test_summary_metric_name_regression() -> None:
    selector = _make_selector()
    assert (selector.summary_["metric_name"] == "spearman").all()


def test_summary_metric_name_binary() -> None:
    X, y_true, *_ = make_residual_signal_data(n=500, random_state=42)
    y_binary = (y_true > y_true.median()).astype(float)
    selector = FeatureSelector(n_bootstraps=10, random_state=42)
    selector.fit(X, y_binary)
    assert (selector.summary_["metric_name"] == "gini").all()


# ---------------------------------------------------------------------------
# 3. Detection tests
# ---------------------------------------------------------------------------


def test_signal_features_rank_above_noise() -> None:
    """x1, x2 carry real signal; x_noise should rank below them."""
    selector = _make_selector(n_bootstraps=30)
    summary = selector.summary_.set_index("feature")
    x1_rank = int(summary.loc["x1", "rank"])
    x2_rank = int(summary.loc["x2", "rank"])
    noise_rank = int(summary.loc["x_noise", "rank"])
    assert x1_rank < noise_rank or x2_rank < noise_rank


def test_null_beat_rate_is_between_0_and_1() -> None:
    selector = _make_selector()
    nbr = selector.summary_["null_beat_rate"].dropna()
    assert (nbr >= 0.0).all() and (nbr <= 1.0).all()


def test_selected_features_subset_of_all_features() -> None:
    selector = _make_selector()
    all_feats = set(selector.summary_["feature"])
    assert set(selector.selected_features_).issubset(all_feats)


# ---------------------------------------------------------------------------
# 4. Validation / error handling
# ---------------------------------------------------------------------------


def test_raises_on_non_dataframe_input() -> None:
    arr = np.ones((50, 3))
    selector = FeatureSelector(n_bootstraps=5)
    with pytest.raises(ValueError, match="DataFrame"):
        selector.fit(arr, np.ones(50))  # type: ignore[arg-type]


def test_raises_on_length_mismatch() -> None:
    X, y_true, *_ = make_residual_signal_data(n=100)
    selector = FeatureSelector(n_bootstraps=5)
    with pytest.raises(ValueError, match="same length"):
        selector.fit(X, y_true.iloc[:50])


def test_raises_on_infinite_y() -> None:
    X, y_true, *_ = make_residual_signal_data(n=100)
    y_bad = y_true.copy()
    y_bad.iloc[0] = float("inf")
    selector = FeatureSelector(n_bootstraps=5)
    with pytest.raises(ValueError, match="finite"):
        selector.fit(X, y_bad)


def test_raises_before_fit_on_plot() -> None:
    selector = FeatureSelector()
    with pytest.raises(RuntimeError, match="fitted"):
        selector.plot_selected_features()


def test_raises_before_find_interactions() -> None:
    selector = _make_selector(n_bootstraps=5)
    with pytest.raises(RuntimeError, match="find_interactions"):
        selector.plot_interactions()


def test_invalid_split_strategy_raises() -> None:
    with pytest.raises(ValueError, match="split_strategy"):
        FeatureSelector(split_strategy="invalid")


def test_invalid_model_type_raises() -> None:
    with pytest.raises(ValueError, match="model_type"):
        FeatureSelector(model_type="catboost")


# ---------------------------------------------------------------------------
# 5. Interaction schema
# ---------------------------------------------------------------------------


def test_interaction_bootstrap_columns() -> None:
    selector = _make_selector(n_bootstraps=10)
    selector.find_interactions(top_n=4)
    assert list(selector.interaction_bootstrap_results_.columns) == INTERACTION_BOOTSTRAP_COLUMNS


def test_interaction_summary_columns() -> None:
    selector = _make_selector(n_bootstraps=10)
    selector.find_interactions(top_n=4)
    assert list(selector.interaction_summary_.columns) == INTERACTION_SUMMARY_COLUMNS


def test_interaction_summary_rank_consecutive() -> None:
    selector = _make_selector(n_bootstraps=10)
    selector.find_interactions(top_n=4)
    ranks = selector.interaction_summary_["rank"].tolist()
    assert ranks == list(range(1, len(ranks) + 1))


def test_find_interactions_row_count() -> None:
    n_bootstraps = 10
    top_n = 3  # 3 features → C(3,2)=3 pairs
    selector = _make_selector(n_bootstraps=n_bootstraps)
    # Provide exactly 3 candidate features
    selector.find_interactions(
        top_n=top_n,
        candidate_features=["x1", "x2", "x3"],
    )
    expected_pairs = 3
    assert len(selector.interaction_bootstrap_results_) == expected_pairs * n_bootstraps
    assert len(selector.interaction_summary_) == expected_pairs


# ---------------------------------------------------------------------------
# 6. Plot return types
# ---------------------------------------------------------------------------


def test_plot_selected_features_returns_figure() -> None:
    selector = _make_selector(n_bootstraps=10)
    fig = selector.plot_selected_features()
    assert isinstance(fig, Figure)


def test_plot_interactions_returns_dict_of_figures() -> None:
    selector = _make_selector(n_bootstraps=10)
    # Pass explicit candidates so we always get >= 3 features → >= 3 pairs
    selector.find_interactions(top_n=10, candidate_features=["x1", "x2", "x3"])
    figs = selector.plot_interactions(top_n=2)
    assert len(figs) == 2
    assert all(isinstance(f, Figure) for f in figs.values())


# ---------------------------------------------------------------------------
# 7. exclude_features
# ---------------------------------------------------------------------------


def test_exclude_features_drops_column() -> None:
    X, y_true, *_ = make_residual_signal_data(n=200, random_state=42)
    selector = FeatureSelector(n_bootstraps=5, random_state=42)
    selector.fit(X, y_true, exclude_features=["x_noise"])
    assert "x_noise" not in selector.summary_["feature"].tolist()
