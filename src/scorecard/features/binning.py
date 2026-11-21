"""Fine and coarse classing: learn the bin edges of one numeric feature.

Edges are LEARNED, so they are fitted on train only (D-002). Bins are
right-closed: with edges (e1, e2) the bins are (-inf, e1], (e1, e2], (e2, inf),
so a new value outside the train range still lands in an end bin. NaN always
gets its own Missing bin, stored last, and is never merged with numeric bins.

Steps (D-014):
1. Fine classing: up to `fine_bins` quantile bins.
2. Merge any bin holding fewer than `min_bin_share` of all rows.
3. Merge until the bad rate is monotonic, in both directions.
4. Merge the least informative cuts until at most `max_bins` bins remain.
Keep whichever direction ends with the higher IV.
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import ArrayLike

from scorecard.features.iv import information_value

INCREASING, DECREASING = "increasing", "decreasing"  # direction of the bad rate as x grows


def fine_edges(x: ArrayLike, n_bins: int) -> np.ndarray:
    """Interior cut points at the quantiles of the non-missing values."""
    values = np.asarray(x, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.array([])
    # inverted_cdf returns real data values, so every bin (e_prev, e] contains at
    # least the value e itself: no empty fine bins, and edges like age 35, not 34.6.
    quantiles = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1], method="inverted_cdf")
    edges = np.unique(quantiles)  # count features (mostly 0) give many equal quantiles
    # An edge at the maximum would leave the last bin (max, inf) empty.
    return edges[edges < values.max()]


def missing_index(edges: ArrayLike) -> int:
    """Position of the Missing bin: after the len(edges) + 1 numeric bins."""
    return len(edges) + 1


def assign_bins(x: ArrayLike, edges: ArrayLike) -> np.ndarray:
    """Bin index per value: i means (edges[i-1], edges[i]]; NaN -> missing_index."""
    arr = np.asarray(x, dtype=float)
    # side="left" gives the i with edges[i-1] < v <= edges[i]: right-closed bins.
    idx = np.searchsorted(np.asarray(edges, dtype=float), arr, side="left")
    return np.where(np.isnan(arr), missing_index(edges), idx)


def bin_counts(x: ArrayLike, y: ArrayLike, edges: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Goods and bads per bin, numeric bins first and the Missing bin last."""
    y_arr = np.asarray(y, dtype=float)
    idx = assign_bins(x, edges)
    size = missing_index(edges) + 1
    bads = np.bincount(idx, weights=y_arr, minlength=size)
    goods = np.bincount(idx, weights=1 - y_arr, minlength=size)
    return goods, bads


def bin_labels(edges: ArrayLike) -> list[str]:
    """Readable labels, e.g. ['(-inf, 0]', '(0, 2]', '(2, inf)', 'Missing']."""
    bounds = [-np.inf, *np.asarray(edges, dtype=float), np.inf]
    numeric = [
        f"({lo:g}, {hi:g}]" if np.isfinite(hi) else f"({lo:g}, inf)"
        for lo, hi in itertools.pairwise(bounds)
    ]
    return [*numeric, "Missing"]


# --------------------------------------------------------------------------- #
# Coarse classing: merges on the numeric bins only (Missing excluded)
# --------------------------------------------------------------------------- #
Bins = tuple[np.ndarray, np.ndarray, np.ndarray]  # (edges, goods, bads)


def _bad_rate(goods: np.ndarray, bads: np.ndarray) -> np.ndarray:
    rows = goods + bads
    return np.divide(bads, rows, out=np.zeros_like(rows), where=rows > 0)


def _merge_pair(bins: Bins, i: int) -> Bins:
    """Merge bin i with bin i + 1 by deleting the edge between them."""
    edges, goods, bads = bins

    def merged(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a[:i], [a[i] + a[i + 1]], a[i + 2 :]])

    return np.delete(edges, i), merged(goods), merged(bads)


def merge_small_bins(bins: Bins, min_rows: float) -> Bins:
    """Merge the smallest bin into a neighbour until every bin has >= min_rows."""
    while len(bins[1]) > 1:
        rows = bins[1] + bins[2]
        i = int(np.argmin(rows))
        if rows[i] >= min_rows:
            break
        if i == 0:
            left = 0
        elif i == len(rows) - 1:
            left = i - 1
        else:  # join the neighbour whose bad rate is closer: the merge loses less information
            rate = _bad_rate(bins[1], bins[2])
            left = i - 1 if abs(rate[i - 1] - rate[i]) <= abs(rate[i + 1] - rate[i]) else i
        bins = _merge_pair(bins, left)
    return bins


def make_monotonic(bins: Bins, direction: str) -> Bins:
    """Merge adjacent bins until the bad rate strictly rises (or falls) with x.

    This is the pool-adjacent-violators algorithm behind isotonic regression;
    its final result does not depend on which violation is merged first.
    """
    if direction not in (INCREASING, DECREASING):
        raise ValueError(f"Unknown direction {direction!r}")
    while len(bins[1]) > 1:
        steps = np.diff(_bad_rate(bins[1], bins[2]))
        # Equal neighbouring rates are merged too: two bins with one WoE add nothing.
        wrong = steps <= 0 if direction == INCREASING else steps >= 0
        violations = np.flatnonzero(wrong)
        if violations.size == 0:
            break
        bins = _merge_pair(bins, int(violations[0]))
    return bins


def cap_bins(bins: Bins, max_bins: int, smoothing: float) -> Bins:
    """Merge adjacent bins until at most `max_bins` remain.

    Each round tries every adjacent merge and keeps the one that leaves the
    highest IV, i.e. it removes the cut that carried the least information.
    Merging two neighbours of a monotonic sequence keeps it monotonic, because
    the merged bad rate lies between the two old ones.
    """
    if max_bins < 1:
        raise ValueError(f"max_bins must be >= 1, got {max_bins}")
    while len(bins[1]) > max_bins:
        iv_after = [
            information_value(*_merge_pair(bins, i)[1:], smoothing) for i in range(len(bins[1]) - 1)
        ]
        bins = _merge_pair(bins, int(np.argmax(iv_after)))
    return bins


def coarse_edges(
    x: ArrayLike,
    y: ArrayLike,
    *,
    fine_bins: int,
    min_bin_share: float,
    max_bins: int,
    smoothing: float,
) -> tuple[np.ndarray, str]:
    """Learn the final edges and the bad-rate direction of one feature."""
    x_arr = np.asarray(x, dtype=float)
    edges = fine_edges(x_arr, fine_bins)
    goods, bads = bin_counts(x_arr, y, edges)
    bins: Bins = (edges, goods[:-1], bads[:-1])  # drop Missing: it is never merged

    # Share of ALL rows (Missing included), so the floor means the same for every feature.
    bins = merge_small_bins(bins, min_rows=min_bin_share * len(x_arr))

    candidates = {
        d: cap_bins(make_monotonic(bins, d), max_bins, smoothing) for d in (INCREASING, DECREASING)
    }
    # Keep the direction that preserves more separation. Ties (e.g. a single bin) go
    # to INCREASING because max() keeps the first maximum.
    direction = max(
        candidates, key=lambda d: information_value(candidates[d][1], candidates[d][2], smoothing)
    )
    return candidates[direction][0], direction
