"""Reason codes: points-lost maths, threshold, cap, de-duplication, Reg B age check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.data.load import FEATURES
from scorecard.explain.reason_codes import (
    REASONS,
    applicant_points,
    elderly_rule_check,
    load_reference,
    reference_points,
    save_reference,
    top_reasons,
)
from scorecard.models.scorecard import Scorecard

UTIL, AGE = "RevolvingUtilizationOfUnsecuredLines", "age"
D90, D30 = "NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"
BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 10_000
    df = pd.DataFrame(
        {
            UTIL: rng.uniform(0, 1.2, n),
            AGE: rng.uniform(20, 80, n).round(),
            D90: rng.poisson(0.1, n).astype(float),
            D30: rng.poisson(0.3, n).astype(float),
        }
    )
    logit = -3.5 + 2.5 * df[UTIL] - 0.02 * (df[AGE] - 50) + 1.2 * df[D90] + 0.6 * df[D30]
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[:299, [D90, D30]] = np.nan  # like the 96/98 rows: both missing together
    df.loc[:299, "bad"] = (rng.uniform(size=300) < 0.5).astype(int)
    return df


@pytest.fixture(scope="module")
def sc(train) -> Scorecard:
    return Scorecard([UTIL, D90, D30, AGE], BINNING).fit(train, train["bad"])


@pytest.fixture(scope="module")
def ref(sc, train) -> dict[str, float]:
    return reference_points(sc, train)


def applicant(**kw: float) -> pd.DataFrame:
    base = {UTIL: 0.05, AGE: 70.0, D90: 0.0, D30: 0.0}  # a safe applicant
    return pd.DataFrame([{**base, **kw}])


def test_every_feature_has_a_reason_and_codes_are_unique():
    assert set(REASONS) == set(FEATURES)
    codes = [r.code for pair in REASONS.values() for r in pair if r is not None]
    # PAY_UNKNOWN is deliberately shared by the three past-due features.
    assert {c for c in codes if codes.count(c) > 1} == {"PAY_UNKNOWN"}


def test_reference_is_train_mean_points_and_round_trips(sc, train, ref, tmp_path):
    pts, _ = applicant_points(sc, train)
    assert ref[UTIL] == pytest.approx(pts[UTIL].mean())
    save_reference(ref, tmp_path / "r.json")
    assert load_reference(tmp_path / "r.json") == pytest.approx(ref)


def test_points_lost_is_reference_minus_points_and_ranked(sc, ref):
    X = applicant(**{UTIL: 1.15, D90: 3.0})
    r = top_reasons(sc, X, ref, max_reasons=4, min_points_lost=0.0).iloc[0]
    pts, _ = applicant_points(sc, X)
    lost = {f: ref[f] - pts.at[0, f] for f in sc.features}
    biggest = max(lost, key=lost.get)
    assert r["reason_1"] == REASONS[biggest][0].code
    assert r["points_lost_1"] == pytest.approx(round(lost[biggest], 1))
    assert {r["reason_1"], r["reason_2"]} == {"UTIL_HIGH", "DLQ_90"}


def test_safe_applicant_gets_no_reason(sc, ref):
    r = top_reasons(sc, applicant(), ref).iloc[0]
    assert (r[[f"reason_{k}" for k in range(1, 5)]] == "").all()


def test_threshold_drops_small_losses_but_keeps_at_least_one(sc, ref):
    X = applicant(**{UTIL: 1.15, D30: 1.0})
    huge = top_reasons(sc, X, ref, max_reasons=4, min_points_lost=1_000).iloc[0]
    assert huge["reason_1"] != "" and huge["reason_2"] == ""  # only the largest survives
    capped = top_reasons(sc, X, ref, max_reasons=1, min_points_lost=0).iloc[0]
    assert list(capped.index) == ["reason_1", "points_lost_1"]


def test_missing_past_due_is_one_reason_not_three(sc, ref):
    X = applicant(**{D90: np.nan, D30: np.nan})
    r = top_reasons(sc, X, ref, max_reasons=4, min_points_lost=0).iloc[0]
    codes = [c for c in r[[f"reason_{k}" for k in range(1, 5)]] if c]
    assert codes.count("PAY_UNKNOWN") == 1


def test_elderly_rule_check(sc):
    assert elderly_rule_check(sc)["passes"]  # risk falls with age in the fixture
    broken = Scorecard.__new__(Scorecard)
    broken.encoder_, broken.features = sc.encoder_, sc.features
    broken.points_ = {**sc.points_, AGE: list(reversed(sc.points_[AGE][:-1])) + [0]}
    assert not elderly_rule_check(broken)["passes"]  # old applicants now get the fewest points
