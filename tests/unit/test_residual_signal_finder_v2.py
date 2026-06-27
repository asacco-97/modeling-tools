from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from tests.fixtures.synthetic_data import make_residual_signal_data

from pe_tools.signal_finder import ResidualSignalFinderV2
from pe_tools.signal_finder.v2 import _assign_bins, _make_bin_spec

matplotlib.use("Agg")


def make_v2_finder(**overrides: object) -> ResidualSignalFinderV2:
    params: dict[str, Any] = {
        "screening_enabled": False,
        "n_bootstraps": 5,
        "test_size": 0.25,
        "n_bins": 5,

        "univariate_model_type": "random_forest",
        "random_state": 42,
        "use_sample_weight": True,
        "model_params": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 5},
    }
    params.update(overrides)
    return ResidualSignalFinderV2(**params)


def make_v2_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    X, y_true, y_pred, sample_weight, _sector, _vintage, split_col = (
        make_residual_signal_data(
            n=300,
            random_state=42,
            include_interaction=False,
            heterogeneity=False,
        )
    )
    return X, y_true, y_pred, sample_weight, split_col


def test_v2_basic_fit_returns_primary_summary_columns() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()

    finder = make_v2_finder().fit(
        X,
        y=y_true,
        base_pred=y_pred,
        sample_weight=sample_weight,
        original_model_features=["x1"],
    )

    summary = finder.get_summary()
    expected_columns = {
        "feature",
        "feature_type",
        "n_observations",
        "mean_oof_residual_r2",
        "median_oof_residual_r2",
        "p05_oof_residual_r2",
        "p95_oof_residual_r2",
        "mean_oof_abs_residual_r2",
        "median_oof_abs_residual_r2",
        "p05_oof_abs_residual_r2",
        "p95_oof_abs_residual_r2",
        "null_beat_rate",
        "prob_residual_signal_gt_zero",
        "prob_abs_residual_signal_gt_zero",
        "mean_residual_signal_rank",
        "median_residual_signal_rank",
        "mean_feature_residual_spearman",
        "median_feature_residual_spearman",
        "pct_positive_feature_residual_spearman",
        "mean_effect_curve_spearman_stability",
        "median_effect_curve_spearman_stability",
        "std_effect_curve_spearman_stability",
        "min_bin_n",
        "max_bin_share",
        "sparse_bin_warning",
        "residual_shape_class",
        "action_category",
        "action_recommendation",
    }
    assert set(summary.columns) == expected_columns
    assert "x1" in summary.head(2)["feature"].tolist()
    assert finder.residual_summary_["rmse_residual"] > 0
    assert finder.warnings_


def test_v2_bootstrap_and_null_outputs_have_expected_columns() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()

    finder = make_v2_finder().fit(X, y=y_true, base_pred=y_pred, sample_weight=sample_weight)

    assert len(finder.bootstrap_results_["bootstrap_id"].unique()) == 5
    assert {
        "bootstrap_id",
        "feature",
        "oof_r2",
        "abs_oof_r2",
        "rank",
        "spearman",
        "is_top_5",
        "is_top_10",
        "null_oof_r2",
        "beats_null",
    }.issubset(finder.bootstrap_results_.columns)
    assert {
        "bootstrap_id",
        "null_feature",
        "source_feature",
        "oof_r2",
        "rank",
        "mean_effect_curve_spearman_stability",
        "median_effect_curve_spearman_stability",
        "std_effect_curve_spearman_stability",
    }.issubset(finder.null_results_.columns)
    assert not finder.effect_curves_.empty


def test_v2_custom_splits_and_screening_work() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    splits = [
        (np.arange(0, 140), np.arange(140, 200)),
        (np.arange(40, 190), np.arange(200, 260)),
    ]
    finder = make_v2_finder(
        screening_enabled=True,
        screening_model_type="random_forest",
        screening_top_k=2,
        screening_cv_folds=3,
        screening_n_repeats=1,
        split_strategy="bootstrap",
        n_bootstraps=1,
        model_params={"n_estimators": 15, "max_depth": 3, "min_samples_leaf": 5},
    )

    finder.fit(X, y=y_true, base_pred=y_pred, sample_weight=sample_weight, splits=splits)

    assert len(finder.get_summary()) == 2
    assert not finder.screening_results_.empty
    assert finder.bootstrap_results_["split_id"].nunique() == 2
    assert finder.bootstrap_results_["split_role"].unique().tolist() == ["validation"]


def test_v2_custom_tvh_splits_store_holdout_without_summary_leakage() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    splits = [
        {
            "train": X.index[:140],
            "validation": X.index[140:200],
            "holdout": X.index[200:240],
        },
        {
            "train": X.index[40:190],
            "validation": X.index[190:240],
            "holdout": X.index[240:280],
        },
    ]

    finder = make_v2_finder(n_bootstraps=1).fit(
        X,
        y=y_true,
        base_pred=y_pred,
        sample_weight=sample_weight,
        splits=splits,
    )

    assert set(finder.bootstrap_results_["split_role"]) == {"validation", "holdout"}
    assert set(finder.effect_curves_["split_role"]) == {"validation", "holdout"}
    validation_rows = finder.bootstrap_results_.loc[
        finder.bootstrap_results_["split_role"].eq("validation")
    ]
    assert validation_rows["split_id"].nunique() == 2
    assert finder.get_summary()["feature"].nunique() == X.shape[1]


def test_v2_custom_splits_accept_boolean_masks() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    train_mask = pd.Series(False, index=X.index)
    validation_mask = pd.Series(False, index=X.index)
    train_mask.iloc[:180] = True
    validation_mask.iloc[180:240] = True

    finder = make_v2_finder(n_bootstraps=1).fit(
        X,
        y=y_true,
        base_pred=y_pred,
        sample_weight=sample_weight,
        splits=[{"train": train_mask, "validation": validation_mask}],
    )

    assert finder.bootstrap_results_["split_id"].unique().tolist() == [0]


def test_v2_custom_splits_reject_overlap_and_empty_validation() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    finder = make_v2_finder()

    with pytest.raises(ValueError, match="train and validation.*disjoint"):
        finder.fit(
            X,
            y=y_true,
            base_pred=y_pred,
            sample_weight=sample_weight,
            splits=[(np.arange(0, 100), np.arange(90, 150))],
        )

    with pytest.raises(ValueError, match="validation indices cannot be empty"):
        finder.fit(
            X,
            y=y_true,
            base_pred=y_pred,
            sample_weight=sample_weight,
            splits=[(np.arange(0, 100), np.asarray([], dtype=int))],
        )


def test_v2_plot_methods_return_figures() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    finder = make_v2_finder().fit(X, y=y_true, base_pred=y_pred, sample_weight=sample_weight)
    top_feature = str(finder.get_summary().iloc[0]["feature"])

    diagnostics = finder.plot_feature_diagnostics(top_feature)
    assert isinstance(diagnostics, Figure)
    assert isinstance(finder.plot_rank_stability(n=3), Figure)
    assert isinstance(finder.plot_residual_signal_map(), Figure)
    # Layout: header, null_r2, spearman, boxplot, boxplot_count, actual, count_strip = 7 axes
    assert len(diagnostics.axes) == 7
    assert diagnostics.axes[1].get_title() == "OOF R² vs. Null Distribution"
    assert diagnostics.axes[2].get_title() == "Bootstrap Spearman Effect Curve Correlation"
    assert diagnostics.axes[3].get_title() == "Residual Distribution by Bin"
    # axes[4] is the boxplot count strip — no title
    assert diagnostics.axes[5].get_title() == "Actual vs. Base Prediction by Feature Bin"
    # axes[6] is the actual count strip — no title
    top_figures = finder.plot_top_features(n=2)
    assert len(top_figures) == 2
    assert all(isinstance(figure, Figure) for figure in top_figures.values())


def test_v2_classification_diagnostics_produce_seven_panel_figure() -> None:
    X, y_true, y_pred, _sample_weight, _split_col = make_v2_data()
    binary_target = (y_true > y_true.median()).astype(float)
    base_probability = pd.Series(
        1.0 / (1.0 + np.exp(-y_pred.to_numpy())),
        index=X.index,
        name="base_probability",
    )
    finder = make_v2_finder(use_sample_weight=False).fit(
        X,
        y=binary_target,
        base_pred=base_probability,
    )
    top_feature = str(finder.get_summary().iloc[0]["feature"])

    diagnostics = finder.plot_feature_diagnostics(top_feature)

    assert len(diagnostics.axes) == 7


def test_v2_plot_handles_missing_null_distribution() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    finder = make_v2_finder().fit(X, y=y_true, base_pred=y_pred, sample_weight=sample_weight)
    finder.null_results_ = pd.DataFrame()
    top_feature = str(finder.get_summary().iloc[0]["feature"])

    fig = finder.plot_feature_diagnostics(top_feature)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 7


def test_v2_discrete_numeric_bins_never_emit_zero_observation_rows() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    rng = np.random.default_rng(42)
    X = X.assign(
        pay_0=rng.choice(
            [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            size=len(X),
            p=[0.12, 0.20, 0.38, 0.12, 0.06, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01],
        )
    )
    finder = make_v2_finder(n_bins=5).fit(
        X.loc[:, ["pay_0"]],
        y=y_true,
        base_pred=y_pred,
        sample_weight=sample_weight,
    )
    pay_curves = finder.effect_curves_.loc[finder.effect_curves_["feature"].eq("pay_0")]

    assert not pay_curves.empty
    assert pay_curves["n_obs"].min() > 0
    assert not pay_curves["centered_mean_residual"].isna().any()
    assert all("–" in str(label) or str(label).lstrip("-").replace(".", "", 1).isdigit() for label in pay_curves["bin_label"].unique())


def test_v2_weighted_bins_have_positive_and_balanced_weight() -> None:
    values = pd.Series(np.arange(40, dtype=float), name="x")
    weights = pd.Series(np.linspace(1.0, 6.0, len(values)), name="weight")
    finder = ResidualSignalFinderV2(n_bins=4)

    spec = _make_bin_spec(values, finder, weights)
    labels = _assign_bins(values, spec)
    bin_weight = weights.groupby(labels).sum()

    assert len(bin_weight) == 4
    assert bin_weight.min() > 0
    assert bin_weight.max() - bin_weight.min() <= weights.max() * 1.25


def test_v2_summary_includes_new_diagnostic_metrics() -> None:
    X, y_true, y_pred, sample_weight, _split_col = make_v2_data()
    finder = make_v2_finder().fit(X, y=y_true, base_pred=y_pred, sample_weight=sample_weight)
    summary = finder.get_summary()

    new_cols = [
        "prob_residual_signal_gt_zero",
        "prob_abs_residual_signal_gt_zero",
        "min_bin_n",
        "max_bin_share",
        "sparse_bin_warning",
        "residual_shape_class",
        "action_category",
        "action_recommendation",
    ]
    for col in new_cols:
        assert col in summary.columns, f"Missing column: {col}"

    assert summary["prob_residual_signal_gt_zero"].between(0.0, 1.0, inclusive="both").all()
    assert summary["prob_abs_residual_signal_gt_zero"].between(0.0, 1.0, inclusive="both").all()
    assert summary["min_bin_n"].gt(0).all()
    assert summary["residual_shape_class"].notna().all()
    assert summary["action_category"].isin({"strong", "moderate", "weak"}).all()
    assert summary["action_recommendation"].notna().all()


def test_v2_classify_residual_shape_monotonic_increasing() -> None:
    from pe_tools.signal_finder.v2 import _classify_residual_shape

    curve = pd.DataFrame({
        "centered_mean_residual": [-0.3, -0.1, 0.1, 0.3, 0.5],
        "bin_label": list("abcde"),
    })
    assert _classify_residual_shape(curve, stability=0.80) == "monotonic_increasing"


def test_v2_classify_residual_shape_monotonic_decreasing() -> None:
    from pe_tools.signal_finder.v2 import _classify_residual_shape

    curve = pd.DataFrame({
        "centered_mean_residual": [0.5, 0.3, 0.1, -0.1, -0.3],
        "bin_label": list("abcde"),
    })
    assert _classify_residual_shape(curve, stability=0.80) == "monotonic_decreasing"


def test_v2_classify_residual_shape_unstable_when_low_stability() -> None:
    from pe_tools.signal_finder.v2 import _classify_residual_shape

    curve = pd.DataFrame({
        "centered_mean_residual": [-0.3, 0.4, -0.2, 0.3, -0.1],
        "bin_label": list("abcde"),
    })
    assert _classify_residual_shape(curve, stability=0.15) == "unstable"


def test_v2_classify_residual_shape_flat_or_noisy() -> None:
    from pe_tools.signal_finder.v2 import _classify_residual_shape

    curve = pd.DataFrame({
        "centered_mean_residual": [0.0001, -0.0001, 0.0002, -0.0001, 0.0],
        "bin_label": list("abcde"),
    })
    assert _classify_residual_shape(curve, stability=0.80) == "flat_or_noisy"


def test_v2_action_recommendation_strong_signal() -> None:
    from pe_tools.signal_finder.v2 import _action_recommendation

    category, text = _action_recommendation(
        mean_oof_r2=0.05,
        prob_signal_gt_zero=0.95,
        null_beat_rate=0.90,
        stability=0.75,
        sparse_bin_warning=False,
        shape_class="monotonic_increasing",
    )
    assert category == "strong"
    assert "re-specification" in text


def test_v2_action_recommendation_weak_signal() -> None:
    from pe_tools.signal_finder.v2 import _action_recommendation

    category, text = _action_recommendation(
        mean_oof_r2=0.001,
        prob_signal_gt_zero=0.55,
        null_beat_rate=0.50,
        stability=0.30,
        sparse_bin_warning=False,
        shape_class="flat_or_noisy",
    )
    assert category == "weak"
    assert "sparse bins" not in text


def test_v2_action_recommendation_appends_sparse_warning() -> None:
    from pe_tools.signal_finder.v2 import _action_recommendation

    _cat, text = _action_recommendation(
        mean_oof_r2=0.001,
        prob_signal_gt_zero=0.60,
        null_beat_rate=0.55,
        stability=0.30,
        sparse_bin_warning=True,
        shape_class="flat_or_noisy",
    )
    assert "sparse bins" in text
