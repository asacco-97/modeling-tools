import pytest
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
