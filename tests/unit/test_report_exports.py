from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from tests.fixtures.synthetic_signal import make_synthetic_residual_data

from pe_tools.signal_finder import ResidualSignalFinder, ResidualSignalFinderResult


def make_result() -> ResidualSignalFinderResult:
    data = make_synthetic_residual_data(row_count=80)
    finder = ResidualSignalFinder(
        n_estimators=5,
        n_splits=3,
        random_state=42,
    )
    return finder.fit(X=data.X, y_true=data.y_true, y_pred=data.y_pred)


def test_report_exports_create_files(tmp_path: Path) -> None:
    result = make_result()
    excel_path = tmp_path / "report.xlsx"
    html_path = tmp_path / "report.html"

    result.to_excel(excel_path)
    result.to_html(html_path)

    assert excel_path.is_file()
    assert html_path.is_file()


def test_expected_excel_sheets_exist(tmp_path: Path) -> None:
    result = make_result()
    excel_path = tmp_path / "report.xlsx"

    result.to_excel(excel_path)

    sheets = set(pd.ExcelFile(excel_path).sheet_names)
    assert {
        "summary",
        "feature_importance",
        "feature_stability",
        "fold_scores",
        "fold_feature_importance",
        "metadata",
    }.issubset(sheets)
    assert any(sheet.startswith("binned_") for sheet in sheets)


def test_expected_text_appears_in_html(tmp_path: Path) -> None:
    result = make_result()
    html_path = tmp_path / "report.html"

    result.to_html(html_path)

    html = html_path.read_text(encoding="utf-8")
    assert "Summary" in html
    assert "Residual Model Score" in html
    assert "Fold Score Distribution" in html
    assert "Feature Stability" in html
    assert "Top Feature Binned Diagnostics" in html
    assert "Warnings / Limitations" in html


def test_report_exports_reject_invalid_paths(tmp_path: Path) -> None:
    result = make_result()

    with pytest.raises(ValueError, match="Parent directory"):
        result.to_excel(tmp_path / "missing" / "report.xlsx")

    with pytest.raises(ValueError, match="must be a file"):
        result.to_html(tmp_path)


def test_excel_sheet_names_are_sanitized(tmp_path: Path) -> None:
    result = make_result()
    bad_feature = "bad/feature:name*with?[very]long sheet title"
    binned_diagnostics = dict(result.binned_diagnostics)
    binned_diagnostics[bad_feature] = next(iter(result.binned_diagnostics.values()))
    feature_importance = pd.concat(
        [
            pd.DataFrame(
                {
                    "feature": [bad_feature],
                    "importance": [999.0],
                    "importance_type": ["gain"],
                }
            ),
            result.feature_importance,
        ],
        ignore_index=True,
    )
    report = replace(
        result,
        feature_importance=feature_importance,
        binned_diagnostics=binned_diagnostics,
    )
    excel_path = tmp_path / "sanitized.xlsx"

    report.to_excel(excel_path)

    sheets = pd.ExcelFile(excel_path).sheet_names
    assert all(len(sheet) <= 31 for sheet in sheets)
    assert all(not set(sheet).intersection("[]:*?/\\") for sheet in sheets)
    assert any(sheet.startswith("binned_bad_feature_name") for sheet in sheets)
