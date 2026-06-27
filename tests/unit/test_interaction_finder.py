import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from tests.fixtures.synthetic_data import make_residual_signal_data

from pe_tools.signal_finder import InteractionFinder

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


def fit_interaction_finder(
    n: int = 500,
    n_bootstraps: int = 30,
    include_interaction: bool = True,
    random_state: int = 42,
) -> InteractionFinder:
    X, y_true, y_pred, *_ = make_residual_signal_data(
        n=n,
        random_state=random_state,
        include_interaction=include_interaction,
        heterogeneity=False,
    )
    residuals = y_true - y_pred
    finder = InteractionFinder(
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )
    finder.fit(X, residuals, candidate_features=list(X.columns))
    return finder


# --- schema tests ---

def test_bootstrap_results_has_expected_columns() -> None:
    finder = fit_interaction_finder()
    assert list(finder.interaction_bootstrap_results_.columns) == INTERACTION_BOOTSTRAP_COLUMNS


def test_summary_has_expected_columns() -> None:
    finder = fit_interaction_finder()
    assert list(finder.interaction_summary_.columns) == INTERACTION_SUMMARY_COLUMNS


def test_summary_has_one_row_per_pair() -> None:
    finder = fit_interaction_finder()
    # C(4, 2) = 6 pairs
    assert len(finder.interaction_summary_) == 6
    assert not finder.interaction_summary_.duplicated(["feature_1", "feature_2"]).any()


def test_summary_rank_is_consecutive() -> None:
    finder = fit_interaction_finder()
    assert finder.interaction_summary_["rank"].tolist() == list(range(1, 7))


def test_bootstrap_results_row_count() -> None:
    n_bootstraps = 20
    finder = fit_interaction_finder(n_bootstraps=n_bootstraps)
    # 6 pairs × 20 bootstraps = 120 rows
    assert len(finder.interaction_bootstrap_results_) == 6 * n_bootstraps


# --- detection tests (success criteria) ---

def test_x2_x3_interaction_ranks_in_top_2() -> None:
    finder = fit_interaction_finder(n_bootstraps=50)
    summary = finder.interaction_summary_
    x2_x3 = summary[
        ((summary["feature_1"] == "x2") & (summary["feature_2"] == "x3"))
        | ((summary["feature_1"] == "x3") & (summary["feature_2"] == "x2"))
    ]
    assert not x2_x3.empty
    assert int(x2_x3.iloc[0]["rank"]) <= 2


def test_x2_x3_null_beat_rate_above_threshold() -> None:
    finder = fit_interaction_finder(n_bootstraps=50)
    summary = finder.interaction_summary_
    x2_x3 = summary[
        ((summary["feature_1"] == "x2") & (summary["feature_2"] == "x3"))
        | ((summary["feature_1"] == "x3") & (summary["feature_2"] == "x2"))
    ]
    assert float(x2_x3.iloc[0]["interaction_null_beat_rate"]) >= 0.80


def test_x2_x3_positive_lift_rate_above_threshold() -> None:
    finder = fit_interaction_finder(n_bootstraps=50)
    summary = finder.interaction_summary_
    x2_x3 = summary[
        ((summary["feature_1"] == "x2") & (summary["feature_2"] == "x3"))
        | ((summary["feature_1"] == "x3") & (summary["feature_2"] == "x2"))
    ]
    assert float(x2_x3.iloc[0]["positive_lift_rate"]) >= 0.75


def test_x1_xnoise_ranks_in_bottom_half() -> None:
    finder = fit_interaction_finder(n_bootstraps=50)
    summary = finder.interaction_summary_
    noise_pair = summary[
        ((summary["feature_1"] == "x1") & (summary["feature_2"] == "x_noise"))
        | ((summary["feature_1"] == "x_noise") & (summary["feature_2"] == "x1"))
    ]
    assert not noise_pair.empty
    # Bottom half of 6 pairs = rank >= 4
    assert int(noise_pair.iloc[0]["rank"]) >= 4


def test_no_interaction_signal_suppressed_without_interaction() -> None:
    finder = fit_interaction_finder(n_bootstraps=30, include_interaction=False)
    summary = finder.interaction_summary_
    x2_x3 = summary[
        ((summary["feature_1"] == "x2") & (summary["feature_2"] == "x3"))
        | ((summary["feature_1"] == "x3") & (summary["feature_2"] == "x2"))
    ]
    assert not x2_x3.empty
    # Without interaction, x2_x3 should not rank first
    assert int(x2_x3.iloc[0]["rank"]) > 1


# --- validation tests ---

def test_raises_when_too_many_candidate_features() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=200)
    residuals = y_true - y_pred
    finder = InteractionFinder(max_candidate_features=3)
    with pytest.raises(ValueError, match="max_candidate_features"):
        finder.fit(X, residuals, candidate_features=list(X.columns))


def test_raises_when_feature_not_in_X() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=200)
    residuals = y_true - y_pred
    finder = InteractionFinder()
    with pytest.raises(ValueError, match="not found in X"):
        finder.fit(X, residuals, candidate_features=["x1", "does_not_exist"])


def test_raises_on_infinite_residuals() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=200)
    bad_residuals = y_true - y_pred
    bad_residuals.iloc[0] = float("inf")
    finder = InteractionFinder()
    with pytest.raises(ValueError, match="finite"):
        finder.fit(X, bad_residuals, candidate_features=["x1", "x2"])


def test_raises_before_fit() -> None:
    finder = InteractionFinder()
    with pytest.raises(RuntimeError, match="fitted"):
        finder.plot_interactions()


# --- plot tests ---

def test_plot_interactions_returns_figures() -> None:
    finder = fit_interaction_finder(n_bootstraps=10)
    figures = finder.plot_interactions(top_n=2)
    assert len(figures) == 2
    assert all(isinstance(f, Figure) for f in figures.values())


def test_plot_interactions_keys_match_pairs() -> None:
    finder = fit_interaction_finder(n_bootstraps=10)
    figures = finder.plot_interactions(top_n=1)
    top_row = finder.interaction_summary_.iloc[0]
    expected_key = f"{top_row['feature_1']}__x__{top_row['feature_2']}"
    assert expected_key in figures


# --- split strategy tests ---

def test_repeated_kfold_split_strategy() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=300, random_state=0)
    residuals = y_true - y_pred
    finder = InteractionFinder(
        n_bootstraps=10,
        split_strategy="repeated_kfold",
        n_splits=5,
        random_state=0,
    )
    finder.fit(X, residuals, candidate_features=["x1", "x2"])
    assert not finder.interaction_summary_.empty


def test_user_passed_splits() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=300, random_state=0)
    residuals = y_true - y_pred
    n = len(X)
    custom_splits = [
        {"train": np.arange(200), "validation": np.arange(200, 300)},
        {"train": np.arange(100, 300), "validation": np.arange(100)},
    ]
    finder = InteractionFinder(n_bootstraps=2, random_state=0)
    finder.fit(X, residuals, candidate_features=["x1", "x2"], splits=custom_splits)
    assert len(finder.interaction_bootstrap_results_) == 2  # 1 pair × 2 splits


def test_categorical_features_explicit_override() -> None:
    X, y_true, y_pred, *_ = make_residual_signal_data(n=300)
    residuals = y_true - y_pred
    finder = InteractionFinder(n_bootstraps=5)
    # x1 is numeric but we declare it categorical via override
    finder.fit(X, residuals, candidate_features=["x1", "x2"], categorical_features=["x1"])
    assert "x1" in finder._categorical_features
    assert not finder.interaction_summary_.empty
