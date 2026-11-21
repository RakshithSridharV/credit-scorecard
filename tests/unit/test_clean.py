from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.data.clean import clean, drop_invalid_rows
from scorecard.data.load import PAST_DUE_COLUMNS, TARGET

# Row positions in the tiny_raw fixture (see tests/conftest.py).
CLEAN_ROW, CODE_98, INCOME_NAN, UTIL_DOLLARS, AGE_ZERO, INCOME_ZERO = range(6)


def test_98_codes_become_nan_in_all_past_due_columns(tiny_raw):
    out = clean(tiny_raw)
    assert out.loc[CODE_98, PAST_DUE_COLUMNS].isna().all()
    # A real count (the 1 in row AGE_ZERO) is untouched.
    assert out.loc[AGE_ZERO, "NumberOfTime30-59DaysPastDueNotWorse"] == 1


def test_96_code_also_becomes_nan(tiny_raw):
    df = tiny_raw.copy()
    df.loc[CLEAN_ROW, "NumberOfTimes90DaysLate"] = 96
    assert np.isnan(clean(df).loc[CLEAN_ROW, "NumberOfTimes90DaysLate"])


@pytest.mark.parametrize("placeholder", [0.0, 1.0])
def test_income_0_and_1_become_nan_and_take_debt_ratio_with_them(tiny_raw, placeholder):
    df = tiny_raw.copy()
    df.loc[CLEAN_ROW, "MonthlyIncome"] = placeholder
    out = clean(df)
    assert np.isnan(out.loc[CLEAN_ROW, "MonthlyIncome"])
    assert np.isnan(out.loc[CLEAN_ROW, "DebtRatio"])


def test_debt_ratio_nan_where_income_missing(tiny_raw):
    out = clean(tiny_raw)
    assert out.loc[[INCOME_NAN, INCOME_ZERO, CODE_98], "DebtRatio"].isna().all()


def test_real_income_and_its_debt_ratio_are_kept(tiny_raw):
    out = clean(tiny_raw)
    assert out.loc[CLEAN_ROW, "MonthlyIncome"] == 5000.0
    assert out.loc[CLEAN_ROW, "DebtRatio"] == 0.30


@pytest.mark.parametrize(
    ("value", "kept"), [(2500.0, False), (10.01, False), (10.0, True), (2.5, True)]
)
def test_utilisation_above_10_becomes_nan(tiny_raw, value, kept):
    df = tiny_raw.copy()
    df.loc[CLEAN_ROW, "RevolvingUtilizationOfUnsecuredLines"] = value
    result = clean(df).loc[CLEAN_ROW, "RevolvingUtilizationOfUnsecuredLines"]
    assert (result == value) if kept else np.isnan(result)


def test_dependents_nan_left_alone(tiny_raw):
    out = clean(tiny_raw)
    pd.testing.assert_series_equal(out["NumberOfDependents"], tiny_raw["NumberOfDependents"])


def test_clean_never_drops_rows_and_keeps_index(tiny_raw):
    df = tiny_raw.set_axis(pd.Index([10, 20, 30, 40, 50, 60], name="row_id"))
    out = clean(df)
    assert out.index.equals(df.index)


def test_clean_is_idempotent(tiny_raw):
    # Running twice must equal running once: guards the D-004 order trap, where a
    # rule that only checked "income is NaN" would behave differently on the second pass.
    once = clean(tiny_raw)
    pd.testing.assert_frame_equal(clean(once), once)


def test_input_not_mutated(tiny_raw):
    before = tiny_raw.copy()
    clean(tiny_raw)
    pd.testing.assert_frame_equal(tiny_raw, before)


def test_works_without_target_like_the_api(tiny_raw):
    out = clean(tiny_raw.drop(columns=[TARGET]))
    assert TARGET not in out.columns


def test_missing_feature_raises(tiny_raw):
    with pytest.raises(ValueError, match="missing columns"):
        clean(tiny_raw.drop(columns=["DebtRatio"]))


def test_drop_invalid_rows_removes_only_under_18(tiny_raw):
    df = tiny_raw.copy()
    df.loc[CLEAN_ROW, "age"] = 18  # boundary: 18 is an adult and is kept
    out = drop_invalid_rows(df)
    assert AGE_ZERO not in out.index
    assert len(out) == len(df) - 1
