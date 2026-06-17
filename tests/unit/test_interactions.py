from pathlib import Path

import pytest
from matplotlib.figure import Figure
from tests.fixtures.synthetic_data import make_residual_signal_data

from pe_tools.signal_finder import ResidualSignalFinder, ResidualSignalFinderResult


def fit_interaction_result() -> ResidualSignalFinderResult:
    pytest.importorskip("shap")
    X, y_true, y_pred, _weights, _sector, _vintage_year, _split_col = make_residual_signal_data(
        n=500,
        random_state=42,
        include_interaction=True,
        heterogeneity=False,
    )
    finder = ResidualSignalFinder(
        model_type="xgboost",
        max_depth=2,
        n_estimators=120,
        learning_rate=0.06,
        n_splits=3,
        random_state=42,
    )
    return finder.fit(X=X, y_true=y_true, y_pred=y_pred)


def test_synthetic_x2_x3_interaction_ranks_near_top() -> None:
    result = fit_interaction_result()
    interactions = result.interaction_importance
    x2_x3 = interactions[
        (interactions["feature_1"] == "x2") & (interactions["feature_2"] == "x3")
    ]

    assert not x2_x3.empty
    assert int(x2_x3.iloc[0]["rank"]) <= 2


def test_interaction_importance_has_expected_columns() -> None:
    result = fit_interaction_result()

    assert list(result.interaction_importance.columns) == [
        "feature_1",
        "feature_2",
        "importance",
        "rank",
    ]


def test_interactions_are_aggregated_across_folds() -> None:
    result = fit_interaction_result()
    interactions = result.interaction_importance

    assert len(result.fold_scores) == 3
    assert not interactions.duplicated(["feature_1", "feature_2"]).any()
    assert interactions["rank"].tolist() == list(range(1, len(interactions) + 1))


def test_interaction_diagnostics_are_stored_for_top_interactions() -> None:
    result = fit_interaction_result()
    top_interaction = result.interaction_importance.iloc[0]
    key = f"{top_interaction['feature_1']}__x__{top_interaction['feature_2']}"

    assert key in result.interaction_diagnostics
    diagnostics = result.interaction_diagnostics[key]
    assert {
        "feature_1_bin",
        "feature_2_bin",
        "actual_mean",
        "predicted_mean",
        "error_mean",
        "n_obs",
    }.issubset(diagnostics.columns)
    assert not diagnostics.empty


def test_plot_top_interactions_returns_and_saves_figures(tmp_path: Path) -> None:
    result = fit_interaction_result()

    figures = result.plot_top_interactions(top_n=2, save_dir=tmp_path)

    assert figures
    assert len(figures) == 2
    assert all(isinstance(figure, Figure) for figure in figures.values())
    assert len(list(tmp_path.glob("*_interaction.png"))) == 2
