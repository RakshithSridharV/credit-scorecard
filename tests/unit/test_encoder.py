"""The WoEEncoder contract: fit on train, transform anything, survive a JSON round trip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from scorecard.features.encoder import WoEEncoder


@pytest.fixture
def train() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 5_000
    util = rng.uniform(0, 1.5, n)
    late = rng.poisson(0.3, n).astype(float)
    p_bad = np.clip(0.02 + 0.1 * util + 0.1 * late, 0, 0.9)
    y = pd.Series((rng.uniform(size=n) < p_bad).astype(int), name="y")
    late[:50] = np.nan  # this feature has a Missing bin; util does not
    return pd.DataFrame({"util": util, "late": late}), y


def test_transform_returns_one_woe_per_bin(train):
    X, y = train
    enc = WoEEncoder(min_bin_share=0.05).fit(X, y)
    out = enc.transform(X)
    assert out.shape == X.shape and out.index.equals(X.index)
    assert out.notna().all().all()
    for col, fb in enc.features_.items():
        assert set(out[col].round(12)) <= set(np.round(fb.woe, 12))


def test_riskier_values_get_lower_woe(train):
    X, y = train
    enc = WoEEncoder(min_bin_share=0.05).fit(X, y)
    woe = enc.transform(pd.DataFrame({"util": [0.05, 1.4], "late": [0.0, 5.0]}))
    assert (woe.iloc[0] > woe.iloc[1]).all()  # low util / never late is safer


def test_unseen_nan_gets_neutral_woe(train):
    X, y = train
    enc = WoEEncoder(min_bin_share=0.05).fit(X, y)
    woe = enc.transform(pd.DataFrame({"util": [np.nan], "late": [0.0]}))
    assert woe.loc[0, "util"] == 0  # util had no NaN in train


def test_fit_is_deterministic(train):
    X, y = train
    a = WoEEncoder().fit(X, y).features_
    b = WoEEncoder().fit(X, y).features_
    assert a == b


def test_transform_does_not_refit(train):
    """Transforming other data must not change what was learned (no leakage)."""
    X, y = train
    enc = WoEEncoder(min_bin_share=0.05).fit(X, y)
    before = dict(enc.features_)
    enc.transform(X * 3)
    assert enc.features_ == before


def test_json_round_trip_gives_identical_output(train, tmp_path):
    X, y = train
    enc = WoEEncoder(min_bin_share=0.05).fit(X, y)
    path = tmp_path / "enc.json"
    enc.to_json(path)
    loaded = WoEEncoder.from_json(path)
    assert loaded.get_params() == enc.get_params()
    pd.testing.assert_frame_equal(loaded.transform(X), enc.transform(X))


def test_unfitted_and_bad_inputs_fail_loudly(train):
    X, y = train
    with pytest.raises(NotFittedError):
        WoEEncoder().transform(X)
    with pytest.raises(ValueError, match="only 0"):
        WoEEncoder().fit(X, y * 2)
    enc = WoEEncoder().fit(X, y)
    with pytest.raises(ValueError, match="missing columns"):
        enc.transform(X[["util"]])


def test_iv_report_is_sorted_and_non_negative(train):
    X, y = train
    iv = WoEEncoder(min_bin_share=0.05).fit(X, y).iv()
    assert (iv >= 0).all()
    assert iv.is_monotonic_decreasing
