"""Subgroup report: bands, the per-group arithmetic, and small-group handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.evaluation.subgroups import MIN_CLASS, age_band, income_status, subgroup_table


def test_age_bands_split_at_the_regulation_b_threshold():
    bands = age_band(pd.Series([18, 29, 30, 44, 45, 61, 62, 109]))
    assert list(bands) == ["18-29", "18-29", "30-44", "30-44", "45-61", "45-61", "62+", "62+"]


def test_income_status_treats_missing_as_not_reported():
    s = income_status(pd.Series([np.nan, 0.0, 5000.0]))
    assert list(s) == ["not reported", "reported", "reported"]


def test_hand_checked_group_numbers():
    #            group a: 4 rows, 1 bad      group b: 2 rows, 1 bad
    groups = pd.Series(["a", "a", "a", "a", "b", "b"])
    y = [0, 0, 0, 1, 0, 1]
    pd_ = [0.1, 0.1, 0.2, 0.4, 0.3, 0.5]
    scores = [600, 600, 560, 500, 540, 500]  # cutoff 549: a approves 3 of 4, b approves 0
    t = subgroup_table(groups, y, pd_, scores, cutoff=549)

    a, b = t.loc["a"], t.loc["b"]
    assert a["rows"] == 4 and a["share"] == pytest.approx(4 / 6)
    assert a["bad_rate"] == pytest.approx(0.25)
    assert a["pd_to_bad_rate"] == pytest.approx(0.2 / 0.25)
    assert a["approval_rate"] == pytest.approx(0.75)
    assert a["bad_rate_approved"] == 0  # the one bad was declined
    assert a["goods_declined"] == 0
    assert b["approval_rate"] == 0 and np.isnan(b["bad_rate_approved"])
    assert b["goods_declined"] == 1  # b's only good was declined
    assert t["approval_ratio"].tolist() == [1.0, 0.0]


def test_auc_needs_enough_bads_and_goods():
    rng = np.random.default_rng(0)
    n = 4 * MIN_CLASS
    y = np.r_[np.ones(MIN_CLASS), np.zeros(n - MIN_CLASS)].astype(int)
    pd_ = rng.uniform(size=n)
    groups = pd.Series(["big"] * n + ["small"] * 10)
    t = subgroup_table(
        groups, np.r_[y, [1] + [0] * 9], np.r_[pd_, rng.uniform(size=10)], np.full(n + 10, 600), 549
    )
    assert not np.isnan(t.loc["big", "auc"])
    assert np.isnan(t.loc["small", "auc"])  # 1 bad: no AUC reported


def test_age_bands_are_listed_youngest_first():
    groups = age_band(pd.Series([70, 20, 50, 35] * 10))
    t = subgroup_table(groups, [0, 1] * 20, np.linspace(0, 1, 40), np.full(40, 600), 549)
    assert list(t.index) == ["18-29", "30-44", "45-61", "62+"]
