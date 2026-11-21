"""The points scorecard: scale maths, points add up, round trip, sensible direction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scorecard.features.binning import assign_bins, bin_labels
from scorecard.models.scorecard import Scaling, Scorecard

BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}


def test_scale_hits_base_score_and_doubles_odds_every_pdo():
    s = Scaling(base_score=600, base_odds=50, pdo=20)
    pd_600, pd_620 = s.pd_from_score([600, 620])
    assert (1 - pd_600) / pd_600 == pytest.approx(50)  # 50:1 good:bad at 600
    assert (1 - pd_620) / pd_620 == pytest.approx(100)  # 20 points later: odds doubled
    assert s.factor == pytest.approx(20 / math.log(2))


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 8_000
    df = pd.DataFrame({"util": rng.uniform(0, 1.2, n), "age": rng.uniform(20, 80, n)})
    logit = -3 + 2.5 * df["util"] - 0.02 * (df["age"] - 50)
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[:99, "util"] = np.nan
    return df


@pytest.fixture
def fitted(data) -> Scorecard:
    return Scorecard(["util", "age"], BINNING).fit(data, data["bad"])


def test_score_is_the_sum_of_bin_points(fitted, data):
    row = data.iloc[[0, 500]]
    manual = []
    for _, r in row.iterrows():
        total = 0
        for f in fitted.features:
            t = fitted.points_table().query("feature == @f")
            fb = fitted.encoder_.features_[f]
            label = bin_labels(fb.edges)[assign_bins([r[f]], fb.edges)[0]]
            total += int(t.loc[t["bin"] == label, "points"].iloc[0])
        manual.append(total)
    assert fitted.score(row).tolist() == manual


def test_coefficients_are_negative_and_riskier_applicants_score_lower(fitted):
    assert all(c < 0 for c in fitted.coef_.values())
    safe, risky = fitted.score(pd.DataFrame({"util": [0.05, 1.15], "age": [70, 22]}))
    assert safe > risky


def test_rounded_points_barely_change_the_ranking(fitted, data):
    exact = roc_auc_score(data["bad"], fitted.predict_pd(data))
    points = roc_auc_score(data["bad"], -fitted.score(data))
    assert points == pytest.approx(exact, abs=0.005)


def test_json_round_trip_keeps_scores_and_calibrated_pd(fitted, data, tmp_path):
    fitted.calibrate(fitted.score(data), data["bad"], pd_floor=0.0005)
    path = tmp_path / "sc.json"
    fitted.to_json(path)
    loaded = Scorecard.from_json(path)
    np.testing.assert_array_equal(loaded.score(data), fitted.score(data))
    np.testing.assert_allclose(loaded.score_pd(data), fitted.score_pd(data))


def test_score_pd_falls_back_to_raw_scale_before_calibration(fitted, data):
    np.testing.assert_allclose(
        fitted.score_pd(data), fitted.scaling.pd_from_score(fitted.score(data))
    )
