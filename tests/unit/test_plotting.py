from pathlib import Path

import pytest
from matplotlib.figure import Figure
from tests.fixtures.synthetic_signal import make_synthetic_residual_data

from pe_tools.signal_finder import ResidualSignalFinder, ResidualSignalFinderResult


def make_result(weighted: bool = False) -> ResidualSignalFinderResult:
    data = make_synthetic_residual_data(row_count=80)
    finder = ResidualSignalFinder(
        n_estimators=5,
        n_splits=3,
        random_state=42,
    )
    weights = data.sample_weight if weighted else None
    return finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred, sample_weight=weights)


def test_plot_top_residual_signals_returns_figures() -> None:
    result = make_result()

    figures = result.plot_top_residual_signals(top_n=2)

    assert figures
    assert len(figures) == 2
    assert all(isinstance(figure, Figure) for figure in figures.values())


def test_plot_figures_can_be_saved_to_temporary_paths(tmp_path: Path) -> None:
    result = make_result()

    figures = result.plot_top_residual_signals(top_n=1)
    assert not list(tmp_path.iterdir())

    feature, figure = next(iter(figures.items()))
    manual_path = tmp_path / f"{feature}.png"
    figure.savefig(manual_path)
    assert manual_path.is_file()

    result.plot_top_residual_signals(top_n=1, save_dir=tmp_path)
    assert any(path.name.endswith(".png") for path in tmp_path.iterdir())


def test_plotting_works_for_weighted_and_unweighted_diagnostics() -> None:
    unweighted = make_result(weighted=False)
    weighted = make_result(weighted=True)

    unweighted_figures = unweighted.plot_top_residual_signals(top_n=1)
    weighted_figures = weighted.plot_top_signals(top_n=1)

    assert all(isinstance(figure, Figure) for figure in unweighted_figures.values())
    assert all(isinstance(figure, Figure) for figure in weighted_figures.values())


def test_binned_diagnostics_include_confidence_intervals() -> None:
    result = make_result()
    diagnostics = next(iter(result.binned_diagnostics.values()))

    assert {
        "actual_ci_low",
        "actual_ci_high",
        "predicted_ci_low",
        "predicted_ci_high",
        "error_ci_low",
        "error_ci_high",
    }.issubset(diagnostics.columns)
    assert (diagnostics["actual_ci_low"] <= diagnostics["actual_mean"]).all()
    assert (diagnostics["actual_mean"] <= diagnostics["actual_ci_high"]).all()
    assert (diagnostics["predicted_ci_low"] <= diagnostics["predicted_mean"]).all()
    assert (diagnostics["predicted_mean"] <= diagnostics["predicted_ci_high"]).all()
    assert (diagnostics["error_ci_low"] <= diagnostics["error_mean"]).all()
    assert (diagnostics["error_mean"] <= diagnostics["error_ci_high"]).all()


def test_residual_plot_uses_stacked_error_panel_and_shaded_confidence_bands() -> None:
    result = make_result()

    figure = next(iter(result.plot_top_residual_signals(top_n=1).values()))

    assert len(figure.axes) == 2
    primary_axis, error_axis = figure.axes
    assert "Out-of-Sample" in primary_axis.get_title()
    assert "Error" in error_axis.get_ylabel()
    assert len(primary_axis.collections) >= 2
    assert len(error_axis.collections) >= 1
    assert all(line.get_color() == "darkgray" for line in error_axis.lines[:1])
    lower, upper = error_axis.get_ylim()
    assert abs(lower) == pytest.approx(abs(upper))
    assert any(line.get_linestyle() == ":" for line in error_axis.lines)
