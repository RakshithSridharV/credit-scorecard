"""Selection gates on small synthetic data where the right answer is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from scorecard.features.selection import (
    backward_elimination,
    correlation_filter,
    gates_table,
    iv_filter,
    vif,
    vif_filter,
    wrong_sign_features,
)

BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}


def test_iv_filter_drops_weak_and_sorts_strongest_first():
    iv = pd.Series({"a": 0.05, "b": 0.01, "c": 0.4})
    assert iv_filter(iv, min_iv=0.02) == ["c", "a"]


def test_correlation_filter_keeps_the_stronger_of_a_correlated_pair():
    iv = pd.Series({"strong": 0.5, "copy": 0.3, "other": 0.1})
    corr = pd.DataFrame(
        [[1, 0.9, 0.1], [0.9, 1, 0.1], [0.1, 0.1, 1]],
        index=iv.index,
        columns=iv.index,
    )
    assert correlation_filter(iv, corr, max_corr=0.7) == ["strong", "other"]


def test_vif_is_one_for_independent_and_large_for_overlapping_columns():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 5_000))
    v = vif(pd.DataFrame({"a": a, "b": b}))
    assert v.to_numpy() == pytest.approx([1, 1], abs=0.01)

    c = a + b + rng.normal(scale=0.05, size=a.size)  # almost a + b
    v = vif(pd.DataFrame({"a": a, "b": b, "c": c}))
    assert (v > 100).all()


def test_vif_filter_drops_the_weakest_offender():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 5_000))
    woe = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(scale=0.05, size=a.size)})
    iv = pd.Series({"a": 0.5, "b": 0.4, "c": 0.1})
    assert vif_filter(iv, woe, max_vif=5) == ["a", "b"]


def test_wrong_sign_means_coefficient_not_negative():
    assert wrong_sign_features([-0.5, 0.1, 0.0], ["a", "b", "c"]) == ["b", "c"]


def test_backward_elimination_drops_pure_noise_and_keeps_signal():
    rng = np.random.default_rng(42)
    n = 6_000
    X = pd.DataFrame(
        {
            "signal_1": rng.normal(size=n),
            "signal_2": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    logit = -2.5 + 1.2 * X["signal_1"] + 0.8 * X["signal_2"]
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))

    kept, history = backward_elimination(
        X, y, list(X.columns), binning=BINNING, folds=folds, min_auc_loss=0.0005
    )

    assert sorted(kept) == ["signal_1", "signal_2"]
    assert history["dropped"].iloc[0] == "noise"
    assert history["dropped"].iloc[-1] == "(stop)"


def test_gates_table_matches_flags_by_name_not_position():
    # IV order (b, c, a) differs from alphabetical order: the old report bug
    # attached the flags by position after pandas sorted the rows alphabetically.
    iv = pd.Series({"b": 0.9, "c": 0.5, "a": 0.1})
    vif_all = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    t = gates_table(iv, vif_all, after_iv=["b", "c", "a"], after_vif=["b", "c"], selected=["c"])
    assert list(t.index) == ["b", "c", "a"]  # highest IV first
    assert t.loc["c", "selected"] and not t.loc["b", "selected"] and not t.loc["a", "selected"]
    assert list(t["passed_redundancy"]) == [True, True, False]
    assert t.loc["a", "vif_all"] == 1.0  # numeric columns follow the name too
