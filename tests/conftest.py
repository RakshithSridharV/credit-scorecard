"""Shared test fixtures: tiny hand-made frames, so tests never depend on data/raw."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.data.load import COLUMNS


@pytest.fixture
def tiny_raw() -> pd.DataFrame:
    """Six rows covering each data issue found in the audit."""
    rows = [
        # target, util, age, 30-59, debt_ratio, income, open, 90, re, 60-89, deps
        [0, 0.20, 45, 0, 0.30, 5000.0, 8, 0, 1, 0, 2.0],  # clean row
        [1, 0.9999999, 30, 98, 0.0, np.nan, 0, 98, 0, 98, 0.0],  # 98 placeholder
        [0, 0.10, 50, 0, 1500.0, np.nan, 5, 0, 1, 0, np.nan],  # income & deps missing
        [0, 2500.0, 40, 0, 0.40, 7000.0, 6, 0, 1, 0, 1.0],  # utilisation in dollars
        [0, 0.50, 0, 1, 0.44, 6000.0, 6, 0, 2, 0, 2.0],  # age 0
        [0, 0.30, 60, 0, 800.0, 0.0, 4, 0, 0, 0, 0.0],  # income 0
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------------------------------- #
# Scoring service on a small synthetic model (shared by unit and API tests)
# --------------------------------------------------------------------------- #
UTIL, AGE, DEBT = "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio"
D90, INCOME = "NumberOfTimes90DaysLate", "MonthlyIncome"
SYNTH_BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}
LOW_RISK = {
    UTIL: 0.05,
    AGE: 60,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    DEBT: 0.2,
    INCOME: 6000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    D90: 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}
HIGH_RISK = {UTIL: 1.1, D90: 3, DEBT: 1.4, AGE: 22}  # overrides: declined by the synthetic model


@pytest.fixture
def applicant():
    """Factory: a complete raw request for a low-risk applicant, any field overridable."""

    def make(**overrides):
        return {**LOW_RISK, **overrides}

    return make


@pytest.fixture(scope="session")
def scoring_service():
    """A real ScoringService around a scorecard fitted on synthetic data.

    Uses 4 of the 7 production features, including DebtRatio with a populated
    Missing bin, so every code path (income rule, reasons, decline) is exercised
    without touching models/ or data/.
    """
    from scorecard.api.scoring import ScoringService
    from scorecard.explain.reason_codes import reference_points
    from scorecard.models.scorecard import Scorecard

    rng = np.random.default_rng(7)
    n = 10_000
    df = pd.DataFrame(
        {
            UTIL: rng.uniform(0, 1.2, n),
            AGE: rng.uniform(20, 80, n).round(),
            D90: rng.poisson(0.1, n).astype(float),
            DEBT: rng.uniform(0, 1.5, n),
        }
    )
    logit = -3.5 + 2.5 * df[UTIL] - 0.02 * (df[AGE] - 50) + 1.2 * df[D90] + 0.8 * df[DEBT]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[:499, DEBT] = np.nan
    sc = Scorecard([UTIL, D90, AGE, DEBT], SYNTH_BINNING).fit(df, pd.Series(y))
    sc.calibrate(sc.score(df), y, pd_floor=0.0005)
    cutoff = int(np.median(sc.score(df)))  # about half declined: both paths get exercised
    return ScoringService(sc, reference_points(sc, df), cutoff, 4, 5.0, "test")
