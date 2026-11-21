from __future__ import annotations

import pandas as pd

from scorecard.data import audit


def test_income_status_labels_each_case():
    income = pd.Series([float("nan"), 0.0, 1.0, 2500.0])
    assert list(audit.income_status(income).astype(str)) == ["missing", "0", "1", ">1"]


def test_past_due_special_rows_detects_codes_and_cap(tiny_raw):
    out = audit.past_due_special_rows(tiny_raw)
    assert out["rows_any_column"] == 1
    assert out["rows_all_three_columns"] == 1
    assert out["bad_rate"] == 1.0
    assert out["util_at_cap_share"] == 1.0  # 0.9999999 counts as "at cap"
    assert out["max_real_value"] == 1


def test_missing_overlap_counts_combinations(tiny_raw):
    out = audit.missing_overlap(tiny_raw)
    assert out.loc["income missing / dependents missing", "rows"] == 1
    assert out.loc["income missing / dependents present", "rows"] == 1
    assert out["rows"].sum() == len(tiny_raw)


def test_duplicate_summary_counts_extra_copies_only(tiny_raw):
    doubled = pd.concat([tiny_raw, tiny_raw.iloc[[0]]], ignore_index=True)
    out = audit.duplicate_summary(doubled)
    assert out["full_row_duplicates"] == 1  # the original is not counted
    assert out["rows_in_conflicting_label_groups"] == 0


def test_md_table_formats_percent_columns():
    df = pd.DataFrame({"rows": [10], "bad_rate": [0.0668]}, index=pd.Index(["a"], name="g"))
    lines = audit.md_table(df, pct_cols=["bad_rate"]).splitlines()
    assert lines[0] == "| g | rows | bad_rate |"
    assert lines[2] == "| a | 10 | 6.68% |"


def test_render_report_runs_on_tiny_data(tiny_raw):
    report = audit.render_report(tiny_raw)
    assert report.startswith("# Data audit")
    assert "## 11. Overlap" in report
