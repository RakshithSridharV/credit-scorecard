"""Metric definitions, checked on cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.evaluation.metrics import gini, ks, summary


def test_perfect_ranking_gives_gini_and_ks_of_one():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.9]
    assert gini(y, p) == pytest.approx(1)
    assert ks(y, p) == pytest.approx(1)


def test_reversed_ranking_gives_gini_minus_one():
    assert gini([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(-1)


def test_ks_matches_hand_calculation():
    # Cutoff "p >= 0.6" catches 2 of 3 bads (TPR 2/3) and 1 of 3 goods (FPR 1/3): gap 1/3.
    # Cutoff "p >= 0.4" catches 3/3 bads and 1/3 goods: gap 2/3, the maximum.
    y = [0, 0, 1, 0, 1, 1]
    p = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]
    assert ks(y, p) == pytest.approx(2 / 3)


def test_random_scores_give_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20_000)
    assert abs(gini(y, rng.uniform(size=y.size))) < 0.03


def test_summary_reports_mean_pd_and_bad_rate():
    s = summary([0, 1, 0, 0], [0.1, 0.6, 0.2, 0.1])
    assert s["mean_pd"] == pytest.approx(0.25)
    assert s["bad_rate"] == pytest.approx(0.25)
    assert set(s) == {"auc", "gini", "ks", "brier", "log_loss", "mean_pd", "bad_rate"}
