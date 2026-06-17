import numpy as np
import pandas as pd
import pytest
from tests.fixtures.synthetic_signal import make_synthetic_residual_data

from pe_tools.signal_finder import ResidualSignalFinder, ResidualSignalFinderResult


def make_binary_residual_signal_data(
    row_count: int = 300,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(42)
    baseline_score = rng.normal(size=row_count)
    residual_signal = rng.normal(size=row_count)
    weak_signal = rng.normal(size=row_count)
    noise = rng.normal(size=row_count)
    y_pred = 1.0 / (1.0 + np.exp(-baseline_score))
    probability = 1.0 / (1.0 + np.exp(-(baseline_score + 2.0 * residual_signal)))
    y_true = rng.binomial(1, probability, size=row_count)

    X = pd.DataFrame(
        {
            "residual_signal": residual_signal,
            "weak_signal": weak_signal,
            "noise": noise,
        }
    )
    split_col = pd.Series(
        np.select(
            [np.arange(row_count) < 200, np.arange(row_count) < 250],
            ["train", "validation"],
            default="holdout",
        ),
        name="split",
    )
    return (
        X,
        pd.Series(y_true, name="y_true"),
        pd.Series(y_pred, name="y_pred"),
        pd.Series(1.0 + rng.random(row_count), name="sample_weight"),
        split_col,
    )


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
        "mean_rank",
        "rank_std",
        "top_1_rate",
        "top_3_rate",
        "selected_rate",
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


def test_holdout_works_with_train_validation_holdout_labels() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(split_strategy="holdout", random_state=42)

    result = finder.fit(
        X=data.X,
        y_true=data.y_true,
        y_pred=data.y_pred,
        split_col=data.split_col,
    )

    assert result.metadata["split_strategy"] == "holdout"
    assert result.fold_scores["split"].tolist() == ["validation", "holdout"]
    assert len(result.models) == 1
    assert not result.feature_importance.empty
    assert result.binned_diagnostics["residual_signal"]["n_obs"].sum() == 40


def test_holdout_works_with_train_holdout_only() -> None:
    data = make_synthetic_residual_data()
    split_col = data.split_col.replace({"validation": "train"})
    finder = ResidualSignalFinder(split_strategy="holdout", random_state=42)

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred, split_col=split_col)

    assert result.fold_scores["split"].tolist() == ["holdout"]
    assert result.binned_diagnostics["residual_signal"]["n_obs"].sum() == 20


def test_holdout_rejects_missing_train_label() -> None:
    data = make_synthetic_residual_data()
    split_col = data.split_col.replace({"train": "validation"})
    finder = ResidualSignalFinder(split_strategy="holdout", random_state=42)

    with pytest.raises(ValueError, match="train label"):
        finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred, split_col=split_col)


def test_group_cv_requires_group_col() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(split_strategy="group_cv", group_col=None, random_state=42)

    with pytest.raises(ValueError, match="group_col.*required.*group_cv"):
        finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)


def test_group_cv_keeps_groups_separated() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(
        split_strategy="group_cv",
        group_col="group",
        n_splits=4,
        random_state=42,
    )

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    seen_groups: set[int] = set()
    for groups in result.fold_scores["test_groups"]:
        fold_groups = set(groups)
        assert seen_groups.isdisjoint(fold_groups)
        seen_groups.update(fold_groups)
    assert seen_groups == set(data.X["group"].unique())
    assert "group" not in set(result.feature_importance["feature"])


def test_group_cv_returns_expected_number_of_folds() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(
        split_strategy="group_cv",
        group_col="group",
        n_splits=3,
        random_state=42,
    )

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert len(result.fold_scores) == 3
    assert len(result.models) == 3


def test_group_cv_is_reproducible() -> None:
    data = make_synthetic_residual_data()
    first = ResidualSignalFinder(
        split_strategy="group_cv",
        group_col="group",
        n_splits=4,
        random_state=42,
    ).fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)
    second = ResidualSignalFinder(
        split_strategy="group_cv",
        group_col="group",
        n_splits=4,
        random_state=42,
    ).fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    pd.testing.assert_frame_equal(first.fold_scores, second.fold_scores)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    assert first.residual_model_score == second.residual_model_score


def test_bootstrap_returns_n_repeats_or_fewer_rows_if_repeats_are_skipped() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(
        split_strategy="bootstrap",
        n_repeats=6,
        sample_fraction=0.8,
        random_state=42,
    )

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert 0 < len(result.fold_scores) <= 6
    assert len(result.models) == len(result.fold_scores)
    assert result.fold_scores["repeat"].between(0, 5).all()
    assert result.metadata["sample_fraction"] == 0.8


def test_bootstrap_feature_stability_is_non_empty() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(
        split_strategy="bootstrap",
        n_repeats=5,
        random_state=42,
    )

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    assert not result.feature_stability.empty
    assert not result.binned_diagnostics["residual_signal"].empty


def test_bootstrap_is_reproducible_with_random_state() -> None:
    data = make_synthetic_residual_data()
    first = ResidualSignalFinder(
        split_strategy="bootstrap",
        n_repeats=5,
        random_state=42,
    ).fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)
    second = ResidualSignalFinder(
        split_strategy="bootstrap",
        n_repeats=5,
        random_state=42,
    ).fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)

    pd.testing.assert_frame_equal(first.fold_scores, second.fold_scores)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    assert first.residual_model_score == second.residual_model_score


@pytest.mark.parametrize("sample_fraction", [0.0, -0.1, 1.1])
def test_bootstrap_rejects_invalid_sample_fraction(sample_fraction: float) -> None:
    with pytest.raises(ValueError, match="sample_fraction"):
        ResidualSignalFinder(split_strategy="bootstrap", sample_fraction=sample_fraction)


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


def test_offbalance_levels_predictions_to_target_across_cv_splits() -> None:
    data = make_synthetic_residual_data()
    finder = ResidualSignalFinder(
        n_estimators=20,
        n_splits=4,
        offbalance=True,
        random_state=42,
    )

    result = finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)
    diagnostics = result.binned_diagnostics["residual_signal"]
    row_count = diagnostics["n_obs"].sum()
    actual_mean = float((diagnostics["actual_mean"] * diagnostics["n_obs"]).sum() / row_count)
    predicted_mean = float(
        (diagnostics["predicted_mean"] * diagnostics["n_obs"]).sum() / row_count
    )

    assert result.metadata["offbalance"] is True
    assert actual_mean == pytest.approx(predicted_mean)


def test_offbalance_levels_displayed_predictions_with_sample_weight() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_binary_residual_signal_data()
    finder = ResidualSignalFinder(
        model_type="xgboost",
        n_estimators=25,
        n_splits=3,
        offbalance=True,
        random_state=42,
    )

    with pytest.warns(UserWarning, match="Binary classification target detected"):
        result = finder.fit(X=X, y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    diagnostics = result.binned_diagnostics["residual_signal"]
    row_count = diagnostics["n_obs"].sum()
    actual_mean = float((diagnostics["actual_mean"] * diagnostics["n_obs"]).sum() / row_count)
    predicted_mean = float(
        (diagnostics["predicted_mean"] * diagnostics["n_obs"]).sum() / row_count
    )

    assert actual_mean == pytest.approx(predicted_mean)


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


def test_binary_target_models_target_with_prediction_control() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_binary_residual_signal_data()
    finder = ResidualSignalFinder(
        model_type="xgboost",
        n_estimators=25,
        n_splits=3,
        random_state=42,
    )

    with pytest.warns(UserWarning, match="Binary classification target detected"):
        result = finder.fit(X=X, y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    expected_residuals = y_true - y_pred
    pd.testing.assert_series_equal(result.residuals, expected_residuals, check_names=False)
    assert result.metadata["task_type"] == "binary_classification"
    assert result.metadata["model_target"] == "y_true"
    control_column = result.metadata["prediction_control_column"]
    assert control_column
    assert control_column not in set(result.feature_importance["feature"])
    assert control_column in result.models[0].get_booster().feature_names


def test_binary_target_importance_and_diagnostics_focus_on_public_features() -> None:
    X, y_true, y_pred, _sample_weight, _split_col = make_binary_residual_signal_data()
    finder = ResidualSignalFinder(
        model_type="xgboost",
        n_estimators=40,
        n_splits=3,
        random_state=42,
    )

    with pytest.warns(UserWarning, match="Binary classification target detected"):
        result = finder.fit(X=X, y_true=y_true, y_pred=y_pred)

    assert result.feature_importance.iloc[0]["feature"] == "residual_signal"
    assert set(result.binned_diagnostics) == set(X.columns)
    diagnostics = result.binned_diagnostics["residual_signal"]
    assert {"actual_mean", "predicted_mean", "error_mean"}.issubset(diagnostics.columns)
    assert "residual_mean" not in diagnostics.columns
    assert "predicted_residual_mean" not in diagnostics.columns
    assert "prediction_error_mean" not in diagnostics.columns
    assert diagnostics["actual_mean"].between(0.0, 1.0).all()


def test_binary_target_holdout_uses_classification_behavior() -> None:
    X, y_true, y_pred, sample_weight, split_col = make_binary_residual_signal_data()
    finder = ResidualSignalFinder(
        model_type="xgboost",
        n_estimators=25,
        split_strategy="holdout",
        random_state=42,
    )

    with pytest.warns(UserWarning, match="Binary classification target detected"):
        result = finder.fit(
            X=X,
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
            split_col=split_col,
        )

    assert result.metadata["task_type"] == "binary_classification"
    assert result.fold_scores["split"].tolist() == ["validation", "holdout"]
    assert result.binned_diagnostics["residual_signal"]["actual_mean"].between(0.0, 1.0).all()
