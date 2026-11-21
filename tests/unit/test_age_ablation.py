"""Age-removal experiment (D-032): the refit, the side-by-side table and the bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.data.load import TARGET
from scorecard.evaluation.age_ablation import compare_by_age, fit_without, paired_bootstrap_age
from scorecard.models.scorecard import Scaling

BANDS = pd.Series(["18-29", "18-29", "30-44", "30-44", "45-61", "45-61", "62+", "62+"])
Y = [0, 1, 0, 0, 0, 1, 0, 1]
ALL = [1] * 8


def test_bootstrap_estimates_are_hand_checkable():
    approved_dep = [1, 0, 1, 1, 1, 1, 1, 1]  # young approve 1 of 2: ratio 0.5
    approved_no = [1, 1, 1, 1, 1, 1, 1, 0]  # young 2 of 2 (ratio 1.0); 62+ falls to 1 of 2
    pd_dep = [0.1] * 6 + [0.2, 0.6]  # 62+: mean PD 0.4 / bad rate 0.5 = 0.8
    pd_no = [0.1] * 6 + [0.5, 0.5]  # 62+: 0.5 / 0.5 = 1.0
    t = paired_bootstrap_age(BANDS, Y, approved_dep, approved_no, pd_dep, pd_no, n=50, seed=0)
    est = t["estimate"]
    assert est["18-29 approval ratio, deployed"] == pytest.approx(0.5)
    assert est["18-29 approval ratio, without age"] == pytest.approx(1.0)
    assert est["18-29 approval ratio, change"] == pytest.approx(0.5)
    assert est["62+ approval rate, change"] == pytest.approx(-0.5)
    assert est["62+ mean PD / bad rate, deployed"] == pytest.approx(0.8)
    assert est["62+ mean PD / bad rate, without age"] == pytest.approx(1.0)


def test_identical_models_show_exactly_zero_change():
    # Pairing: the same resampled rows go to both models, so equal inputs give a
    # change of exactly 0 in every resample, not just on average.
    rng = np.random.default_rng(1)
    bands = pd.Series(np.repeat(["18-29", "30-44", "45-61", "62+"], 50))
    y = rng.integers(0, 2, 200)
    approved = rng.integers(0, 2, 200)
    pd_ = rng.uniform(0.05, 0.5, 200)
    t = paired_bootstrap_age(bands, y, approved, approved, pd_, pd_, n=100, seed=0)
    for row in ("18-29 approval ratio, change", "62+ approval rate, change"):
        assert t.loc[row].tolist() == [0.0, 0.0, 0.0]


def test_bootstrap_refuses_rows_without_an_age_band():
    bands = BANDS.copy()
    bands.iloc[0] = None
    with pytest.raises(ValueError, match="age band"):
        paired_bootstrap_age(bands, Y, ALL, ALL, [0.1] * 8, [0.1] * 8, n=5, seed=0)


def test_compare_by_age_puts_models_side_by_side_with_the_change():
    dep = pd.DataFrame({"approval_rate": [0.76, 0.96]}, index=["18-29", "62+"])
    no = pd.DataFrame({"approval_rate": [0.78, 0.94]}, index=["18-29", "62+"])
    t = compare_by_age(dep, no, ("approval_rate",))
    assert list(t.columns) == [
        "approval_rate (deployed)",
        "approval_rate (no age)",
        "approval_rate change",
    ]
    assert t["approval_rate change"].tolist() == pytest.approx([0.02, -0.02])


def test_fit_without_drops_the_feature_and_keeps_the_recipe():
    rng = np.random.default_rng(42)
    n = 4000
    x = pd.DataFrame(
        {
            "util": rng.uniform(0, 1, n),
            "late": rng.integers(0, 4, n).astype(float),
            "age": rng.integers(18, 90, n).astype(float),
        }
    )
    logit = -3 + 2.5 * x["util"] + 0.8 * x["late"] - 0.02 * (x["age"] - 50)
    x[TARGET] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    binning = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}

    sc = fit_without(x, ["util", "late", "age"], "age", binning, Scaling(), 0.0005, seed=42)

    assert sc.features == ["util", "late"]
    assert hasattr(sc, "score_to_pd_")  # calibrated like the deployed model
    assert all(c < 0 for c in sc.coef_.values())  # WoE up = safer = lower PD
