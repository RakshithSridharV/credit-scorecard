"""Population stability: PSI maths, frozen baseline, CSI points at the moved feature."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scorecard.evaluation.psi import (
    Baseline,
    band_counts,
    csi_table,
    psi,
    psi_terms,
    score_edges,
    score_psi_table,
    shares,
    status,
    stress_utilisation,
    worst,
)
from scorecard.models.scorecard import Scorecard

UTIL, AGE = "RevolvingUtilizationOfUnsecuredLines", "age"
BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}
FLOOR = 1e-4


def test_psi_by_hand_and_properties():
    e, a = [0.5, 0.5], [0.6, 0.4]
    by_hand = (0.6 - 0.5) * math.log(0.6 / 0.5) + (0.4 - 0.5) * math.log(0.4 / 0.5)
    assert psi(e, a) == pytest.approx(by_hand)
    assert psi(e, e) == 0.0
    assert psi(e, a) == pytest.approx(psi(a, e))  # symmetric
    assert (psi_terms(e, a) >= 0).all()  # every band adds, never cancels


def test_floor_keeps_psi_finite_when_a_band_is_empty():
    s = shares([100, 0, 50], FLOOR)
    assert s[1] == FLOOR
    assert np.isfinite(psi(shares([50, 50, 50], FLOOR), s))


def test_score_edges_make_roughly_equal_train_bands():
    rng = np.random.default_rng(0)
    scores = rng.integers(400, 650, 20_000)
    edges = score_edges(scores, 10)
    counts = band_counts(scores, edges)
    assert len(counts) == len(edges) + 1 == 10
    assert counts.min() / counts.sum() > 0.08


def test_status_and_worst():
    assert [status(v, 0.10, 0.25) for v in (0.05, 0.10, 0.3)] == ["stable", "investigate", "ACT"]
    assert worst("stable", "ACT", "investigate") == "ACT"
    assert worst("stable", "stable") == "stable"


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(5)
    n = 8_000
    df = pd.DataFrame({UTIL: rng.uniform(0, 1.2, n), AGE: rng.uniform(20, 80, n).round()})
    logit = -3 + 2.5 * df[UTIL] - 0.02 * (df[AGE] - 50)
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    sc = Scorecard([UTIL, AGE], BINNING).fit(df, df["bad"])
    return sc, df, Baseline.from_train(sc, df, n_bands=10, floor=FLOOR)


def test_baseline_against_itself_is_zero_and_round_trips(setup, tmp_path):
    sc, df, base = setup
    assert score_psi_table(base, sc.score(df), FLOOR)["psi_term"].sum() == pytest.approx(0)
    base.save(tmp_path / "b.json")
    assert Baseline.load(tmp_path / "b.json") == base


def test_csi_names_the_feature_that_moved(setup):
    sc, df, base = setup
    drifted = stress_utilisation(df, np.random.default_rng(0))
    c = csi_table(sc, base, drifted, FLOOR)
    assert c.index[0] == UTIL and c.loc[UTIL, "csi"] > 0.1
    assert c.loc[AGE, "csi"] == pytest.approx(0)  # age untouched
    assert c.loc[UTIL, "mean_points_shift"] < 0  # higher utilisation -> fewer points
