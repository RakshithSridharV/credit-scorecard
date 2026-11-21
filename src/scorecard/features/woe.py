"""Weight of Evidence: one number per bin from that bin's good/bad counts.

    WoE = ln( bin's share of all goods / bin's share of all bads )

Positive WoE = safer than average, negative = riskier, 0 = average.
Pure maths, nothing is fitted here; the bins come from binning.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def woe_values(goods: ArrayLike, bads: ArrayLike, smoothing: float) -> np.ndarray:
    """WoE for each bin.

    `smoothing` is added to each bin's good and bad count before dividing, so a
    bin with zero bads gives a large but finite WoE instead of ln(x / 0) = inf.
    Totals are the raw totals: smoothing only nudges the bin itself.
    """
    g = np.asarray(goods, dtype=float)
    b = np.asarray(bads, dtype=float)
    if g.shape != b.shape:
        raise ValueError(f"goods and bads differ in shape: {g.shape} vs {b.shape}")
    if g.sum() == 0 or b.sum() == 0:
        raise ValueError("WoE needs at least one good and one bad in total")

    woe = np.log(((g + smoothing) / g.sum()) / ((b + smoothing) / b.sum()))
    # An empty bin carries no evidence either way, so it gets the neutral 0.
    # (Smoothing alone would invent ln(total bads / total goods), i.e. "very risky".)
    return np.where(g + b == 0, 0.0, woe)
