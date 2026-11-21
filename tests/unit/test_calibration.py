"""Calibration table and the score -> PD master scale."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.evaluation.calibration import (
    ScoreToPD,
    calibration_in_the_large,
    calibration_table,
)


def test_calibrated_predictions_give_ratios_near_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.4, 50_000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # outcomes drawn from p itself
    t = calibration_table(y, p)
    assert len(t) == 10 and t["rows"].sum() == p.size
    assert t["ratio"].between(0.9, 1.1).all()
    assert calibration_in_the_large(y, p) == pytest.approx(1, abs=0.02)


def test_deciles_can_be_formed_on_the_score_with_ties():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)  # every PD tied: ranking by PD alone would be arbitrary
    t = calibration_table(y, p, rank_by=-np.arange(100))
    assert (t["rows"] == 10).all()


def test_master_scale_never_raises_pd_as_score_rises():
    rng = np.random.default_rng(1)
    scores = rng.integers(450, 650, 20_000)
    true_pd = 1 / (1 + np.exp((scores - 520) / 25))
    y = (rng.uniform(size=scores.size) < true_pd).astype(int)

    table = ScoreToPD.fit(scores, y, pd_floor=0.0005)
    grid = table.predict(np.arange(400, 701))
    assert (np.diff(grid) <= 0).all()
    assert grid.min() >= 0.0005  # floor holds even where no bads were seen
    assert grid[0] == table.pds[0] and grid[-1] == table.pds[-1]  # clipped beyond the range
