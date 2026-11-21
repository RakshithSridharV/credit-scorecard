"""Cutoff strategy: money maths, strategy table by hand, PD rule = profit maximum."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.evaluation.calibration import ScoreToPD
from scorecard.evaluation.cutoff import (
    PER_N,
    Economics,
    band_bad_rates,
    cutoff_from_pd,
    row_at,
    strategy_table,
)

ECON = Economics(ead=5000, annual_margin=0.05, horizon_years=2, lgd=0.75)


def test_economics_and_break_even():
    assert ECON.profit_per_good == pytest.approx(500)
    assert ECON.loss_per_bad == pytest.approx(3750)
    be = ECON.break_even_pd
    # At break-even, expected profit of one more loan is zero.
    assert (1 - be) * ECON.profit_per_good - be * ECON.loss_per_bad == pytest.approx(0)
    assert ECON.scale_loss(2).loss_per_bad == pytest.approx(7500)


def test_strategy_table_by_hand():
    scores, y = [10, 10, 20, 30], [1, 0, 1, 0]
    t = strategy_table(scores, y, ECON)
    assert t.index.tolist() == [10, 20, 30]
    assert t.loc[10, "approval_rate"] == 1.0 and t.loc[10, "bad_rate_approved"] == 0.5
    assert t.loc[20, "approval_rate"] == 0.5 and t.loc[20, "bad_rate_approved"] == 0.5
    r = t.loc[30]  # approve only the 30: one good, both bads kept out, one good lost
    assert (r["bad_rate_approved"], r["bads_declined_share"], r["goods_declined_share"]) == (
        0.0,
        1.0,
        0.5,
    )
    assert r["profit_per_10k"] == pytest.approx(500 / 4 * PER_N)


def test_cutoff_from_pd_picks_lowest_score_at_or_below_break_even():
    table = ScoreToPD((500.0, 600.0), (0.30, 0.01))  # linear in between
    # PD(s) = 0.30 - 0.0029 (s - 500) <= 0.10  <=>  s >= 568.97
    assert cutoff_from_pd(table, 0.10, 400, 700) == 569
    with pytest.raises(ValueError):
        cutoff_from_pd(table, 0.001, 400, 700)  # floor of the table is 1%


def test_row_at_uses_next_observed_score():
    t = strategy_table([10, 20, 30], [1, 0, 0], ECON)
    assert row_at(t, 15).equals(t.loc[20])


def test_pd_rule_cutoff_is_the_profit_maximum_when_pds_are_right():
    """If each score's bad rate equals its PD exactly, the break-even cutoff
    must maximise realised profit: the theory the D-018 rule rests on."""
    score_grid = np.arange(400, 500)
    pds = np.linspace(0.40, 0.01, len(score_grid))  # falls as score rises
    scores, y = [], []
    for s, p in zip(score_grid, pds, strict=True):
        bads = round(p * 1000)
        scores += [s] * 1000
        y += [1] * bads + [0] * (1000 - bads)
    t = strategy_table(scores, y, ECON)
    rule = cutoff_from_pd(
        ScoreToPD(tuple(map(float, score_grid)), tuple(pds)), ECON.break_even_pd, 400, 499
    )
    assert abs(int(t["profit_per_10k"].idxmax()) - rule) <= 1  # rounding of bads to integers


def test_band_bad_rates_marks_sides():
    b = band_bad_rates([95, 100, 105], [1, 0, 0], cutoff=100, width=5)
    assert b.loc["[95, 100)", "side"] == "decline" and b.loc["[95, 100)", "bad_rate"] == 1.0
    assert b.loc["[100, 105)", "side"] == "approve"
