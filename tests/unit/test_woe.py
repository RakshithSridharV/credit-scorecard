"""WoE and IV maths, checked against numbers worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.features.iv import information_value, iv_strength
from scorecard.features.woe import woe_values


def test_woe_matches_hand_calculation():
    # Bin A: 80% of goods, 40% of bads -> ln(0.8 / 0.4) = ln 2 (safer).
    # Bin B: 20% of goods, 60% of bads -> ln(0.2 / 0.6) = ln(1/3) (riskier).
    woe = woe_values([80, 20], [4, 6], smoothing=0)
    assert woe == pytest.approx([np.log(2), np.log(1 / 3)])


def test_bin_with_no_bads_gives_finite_woe_thanks_to_smoothing():
    woe = woe_values([50, 50], [0, 10], smoothing=0.5)
    assert np.isfinite(woe).all()
    assert woe[0] > 0  # no bads at all: much safer than average


def test_empty_bin_gets_neutral_woe():
    assert woe_values([60, 0, 40], [5, 0, 5], smoothing=0.5)[1] == 0


def test_woe_rejects_data_with_no_bads():
    with pytest.raises(ValueError, match="at least one good and one bad"):
        woe_values([10, 10], [0, 0], smoothing=0.5)


def test_iv_is_zero_when_bins_do_not_separate():
    # Both bins have the same 10% bad rate.
    assert information_value([90, 45], [10, 5], smoothing=0) == pytest.approx(0)


def test_iv_matches_hand_calculation():
    # (0.8 - 0.4) * ln 2 + (0.2 - 0.6) * ln(1/3) = 0.4 ln 2 + 0.4 ln 3
    expected = 0.4 * np.log(2) + 0.4 * np.log(3)
    assert information_value([80, 20], [4, 6], smoothing=0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("iv", "label"),
    [
        (0.01, "not useful"),
        (0.05, "weak"),
        (0.2, "medium"),
        (0.4, "strong"),
        (0.9, "suspicious: check for leakage"),
    ],
)
def test_iv_strength_labels(iv, label):
    assert iv_strength(iv) == label
