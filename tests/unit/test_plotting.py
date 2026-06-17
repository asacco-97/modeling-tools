from pathlib import Path

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
