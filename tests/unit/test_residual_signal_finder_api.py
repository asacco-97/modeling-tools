import pandas as pd
import pytest
from tests.fixtures.synthetic_signal import make_synthetic_residual_data

from pe_tools.signal_finder import ResidualSignalFinder, ResidualSignalFinderResult


def test_residuals_equal_y_true_minus_y_pred() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    expected = data.y_true - data.y_pred
    pd.testing.assert_series_equal(result.residuals, expected, check_names=False)


def test_default_split_strategy_cv_works() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert isinstance(result, ResidualSignalFinderResult)
    assert result.metadata["split_strategy"] == "cv"
    assert not result.fold_scores.empty


def test_n_splits_controls_number_of_folds() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(n_splits=3, random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert len(result.fold_scores) == 3
    assert len(result.models) == 3


def test_sample_weight_is_accepted_and_used() -> None:
    data = make_synthetic_residual_data()
    weighted_finder = ResidualSignalFinder(random_state=42)
    unweighted_finder = ResidualSignalFinder(random_state=42)

    weighted_result = weighted_finder.fit(
        X=data.X,
        y_true=data.y_true,
        y_pred=data.y_pred,
        sample_weight=data.sample_weight,
    )
    unweighted_result = unweighted_finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert weighted_result.metadata["sample_weight_used"] is True
    weighted_scores = weighted_result.fold_scores["validation_score"].tolist()
    unweighted_scores = unweighted_result.fold_scores["validation_score"].tolist()
    assert weighted_scores != unweighted_scores


def test_feature_importance_is_non_empty_and_sorted_descending() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert not result.feature_importance.empty
    assert result.feature_importance["importance_type"].eq("gain").all()
    assert result.feature_importance["importance"].is_monotonic_decreasing
    assert result.feature_importance.iloc[0]["feature"] == "residual_signal"


def test_feature_stability_has_expected_columns() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(n_splits=4, random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert set(result.feature_stability.columns) == {
        "feature",
        "mean_importance",
        "std_importance",
        "cv_importance",
        "selection_rate",
        "mean_rank",
    }


def test_fold_scores_has_one_row_per_fold_for_cv() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(split_strategy="cv", n_splits=5, random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert len(result.fold_scores) == 5
    assert set(result.fold_scores["fold"]) == set(range(5))


def test_binned_diagnostics_contains_one_table_per_feature() -> None:
    data = make_synthetic_residual_data()
    feature_columns = ["residual_signal", "weak_signal", "unrelated"]
    finder = ResidualSignalFinder(n_bins=8, random_state=42)

    result = finder.fit(X=data.X.loc[:, feature_columns], y_true=data.y_true, y_pred=data.y_pred)

    assert set(result.binned_diagnostics) == set(feature_columns)
    assert all(not table.empty for table in result.binned_diagnostics.values())


def test_max_depth_1_works() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(max_depth=1, random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert result.metadata["max_depth"] == 1
    assert not result.feature_importance.empty


def test_max_depth_2_works() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(model_type="random_forest", max_depth=2, random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert result.metadata["model_type"] == "random_forest"
    assert result.metadata["max_depth"] == 2
    assert not result.feature_importance.empty


def test_invalid_split_strategy_raises_helpful_value_error() -> None:
    with pytest.raises(ValueError, match="split_strategy.*cv.*holdout.*group_cv.*bootstrap"):
        ResidualSignalFinder(split_strategy="bad_strategy")  # type: ignore[arg-type]


def test_holdout_requires_split_col() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(split_strategy="holdout", random_state=42)

    with pytest.raises(ValueError, match="split_col.*required.*holdout"):
        finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)


def test_group_cv_requires_group_col() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(split_strategy="group_cv", group_col=None, random_state=42)

    with pytest.raises(ValueError, match="group_col.*required.*group_cv"):
        finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)


def test_invalid_inputs_raise_helpful_value_error() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(random_state=42)

    with pytest.raises(ValueError, match="same length"):
        finder.fit(X=data.X.iloc[:-1], y_true=data.y_true, y_pred=data.y_pred)


def test_output_is_reproducible_with_fixed_random_state() -> None:
    data = make_synthetic_residual_data()
    first = ResidualSignalFinder(random_state=42).fit(
        X=data.X,
        y_true=data.y_true,
        y_pred=data.y_pred,
    )
    second = ResidualSignalFinder(random_state=42).fit(
        X=data.X,
        y_true=data.y_true,
        y_pred=data.y_pred,
    )

    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    pd.testing.assert_frame_equal(first.fold_scores, second.fold_scores)
    assert first.residual_model_score == second.residual_model_score


def test_categorical_columns_are_rejected_with_clear_error_for_v1() -> None:
    data = make_synthetic_residual_data()
    X = data.X.copy()
    X["category"] = ["a", "b"] * (len(X) // 2)
    finder = ResidualSignalFinder(random_state=42)

    with pytest.raises(ValueError, match="categorical.*not supported.*v1"):
        finder.fit(X=X, y_true=data.y_true, y_pred=data.y_pred)


def test_missing_values_are_rejected_with_clear_error_for_v1() -> None:
    data = make_synthetic_residual_data()
    X = data.X.copy()
    X.loc[0, "residual_signal"] = pd.NA
    finder = ResidualSignalFinder(random_state=42)

    with pytest.raises(ValueError, match="missing.*not supported.*v1"):
        finder.fit(X=X, y_true=data.y_true, y_pred=data.y_pred)
