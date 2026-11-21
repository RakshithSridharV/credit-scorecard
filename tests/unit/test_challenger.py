"""Challenger: tuning returns usable params, constraints hold, bootstrap is paired and seeded."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from scorecard.data.load import FEATURES
from scorecard.models.challenger import (
    MONOTONE_DIRECTION,
    gain_importance,
    monotone_vector,
    monotone_violations,
    paired_bootstrap_auc_gap,
    tune,
)

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    """Risk rises with x overall, but with a dip in the middle the data can overfit to."""
    rng = np.random.default_rng(3)
    n = 6_000
    X = pd.DataFrame({"x": rng.uniform(0, 10, n), "z": rng.normal(size=n)})
    logit = -3 + 0.4 * X["x"] - 1.5 * ((X["x"] > 4) & (X["x"] < 6)) + 0.3 * X["z"]
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_every_feature_has_a_direction_in_column_order():
    assert set(MONOTONE_DIRECTION) == set(FEATURES)
    assert monotone_vector(FEATURES) == [MONOTONE_DIRECTION[f] for f in FEATURES]


def test_tune_returns_integer_params_and_the_best_grid_point(data):
    X, y = data
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    tuned = tune(X, y, folds, PARAMS, {"num_leaves": [3, 7]}, max_rounds=200, early_stopping=20)
    assert isinstance(tuned.params["num_leaves"], int)  # LightGBM rejects 7.0
    assert len(tuned.grid) == 2
    assert tuned.cv_auc == pytest.approx(tuned.grid["cv_auc"].max())
    assert 1 <= tuned.n_rounds <= 200


def test_monotone_constraint_removes_wrong_way_steps(data):
    X, y = data
    free = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=100)
    mono = lgb.train({**PARAMS, "monotone_constraints": [1, 0]}, lgb.Dataset(X, y), 100)
    assert monotone_violations(free, X, "x", +1, n_rows=50)["rows"] > 0  # learns the dip
    assert monotone_violations(mono, X, "x", +1, n_rows=50) == {"rows": 0, "max_pp": 0.0}


def test_gain_importance_is_a_share():
    X = pd.DataFrame({"a": np.arange(200.0), "b": np.zeros(200)})
    y = (X["a"] > 100).astype(int)
    model = lgb.train({**PARAMS, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 10)
    share = gain_importance(model)
    assert share.sum() == pytest.approx(1.0)
    assert share.index[0] == "a" and share["b"] == 0


def test_bootstrap_gap_is_zero_for_identical_models_and_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    same = paired_bootstrap_auc_gap(y, p, p, n=50, seed=1)
    assert same == {"gap": 0.0, "ci_low": 0.0, "ci_high": 0.0, "share_gap_positive": 0.0}
    better = y + rng.normal(0, 0.8, 500)  # noisy but informative
    a = paired_bootstrap_auc_gap(y, p, better, n=50, seed=1)
    assert a == paired_bootstrap_auc_gap(y, p, better, n=50, seed=1)  # same seed, same answer
    assert a["gap"] > 0 and a["ci_low"] > 0
