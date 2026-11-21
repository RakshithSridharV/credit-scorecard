"""Load the raw Give Me Some Credit training file and check its schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scorecard.config import load_config, resolve_path

# The Kaggle CSV is written with its pandas index as the first, unnamed column.
# It is a row number, not a customer ID, so it carries no information.
ID_COLUMN = "Unnamed: 0"

TARGET = "SeriousDlqin2yrs"  # 1 = 90+ days past due within 2 years ("bad")

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
# Placeholder codes, not counts (see reports/data_audit.md, section 6).
PAST_DUE_SPECIAL_CODES = (96, 98)

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

COLUMNS = [TARGET, *FEATURES]


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read cs-training.csv, drop the row-number column, and validate columns.

    Fails loudly if the file does not have exactly the expected columns: a
    renamed or missing column should stop the pipeline, not flow silently
    into a model.
    """
    if path is None:
        path = resolve_path(load_config()["data"]["raw_train"])
    df = pd.read_csv(path)
    df = df.drop(columns=[ID_COLUMN])

    missing = set(COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(COLUMNS)
    if missing or extra:
        raise ValueError(f"Unexpected schema. Missing: {sorted(missing)}, extra: {sorted(extra)}")
    return df[COLUMNS]
