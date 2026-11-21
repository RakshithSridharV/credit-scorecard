"""Information Value: how well one binned feature separates goods from bads.

    IV = sum over bins of (share of goods - share of bads) * WoE

Each term is >= 0 (the two factors always share a sign), so IV is never
negative and is 0 only when every bin has the population's bad rate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from scorecard.features.woe import woe_values

# Common rule of thumb (Siddiqi, "Credit Risk Scorecards"). Upper bounds, exclusive.
IV_STRENGTH = (
    (0.02, "not useful"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (np.inf, "suspicious: check for leakage"),
)


def information_value(goods: ArrayLike, bads: ArrayLike, smoothing: float) -> float:
    """IV of one feature from its per-bin good and bad counts."""
    g = np.asarray(goods, dtype=float)
    b = np.asarray(bads, dtype=float)
    woe = woe_values(g, b, smoothing)
    return float(np.sum((g / g.sum() - b / b.sum()) * woe))


def iv_strength(iv: float) -> str:
    """Rule-of-thumb label for an IV value."""
    for upper, label in IV_STRENGTH:
        if iv < upper:
            return label
    return IV_STRENGTH[-1][1]
