"""Bin assignment, fine classing and the two coarse-classing rules."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.features.binning import (
    DECREASING,
    INCREASING,
    assign_bins,
    bin_counts,
    bin_labels,
    cap_bins,
    coarse_edges,
    fine_edges,
    make_monotonic,
    merge_small_bins,
)


def test_bins_are_right_closed_and_open_ended():
    edges = [0.0, 2.0]
    # -5 and 0 -> bin 0 (-inf, 0]; 1 and 2 -> bin 1 (0, 2]; 3 and 999 -> bin 2 (2, inf)
    x = [-5, 0, 1, 2, 3, 999, np.nan]
    assert assign_bins(x, edges).tolist() == [0, 0, 1, 1, 2, 2, 3]  # NaN -> Missing (3)


def test_labels_match_bins():
    assert bin_labels([0.0, 2.0]) == ["(-inf, 0]", "(0, 2]", "(2, inf)", "Missing"]


def test_fine_edges_on_a_count_feature_are_real_values_and_unique():
    x = np.array([0] * 90 + [1] * 6 + [2] * 3 + [5])
    edges = fine_edges(x, n_bins=20)
    assert set(edges) <= set(x)  # inverted_cdf: no interpolated edges like 0.4
    assert len(edges) == len(set(edges))
    assert edges.max() < x.max()  # no edge at the max -> last bin is never empty


def test_bin_counts_put_missing_last():
    goods, bads = bin_counts([0, 1, np.nan, np.nan], [0, 1, 1, 0], edges=[0.0])
    assert goods.tolist() == [1, 0, 1]
    assert bads.tolist() == [0, 1, 1]


def test_small_bin_joins_the_neighbour_with_closer_bad_rate():
    edges = np.array([1.0, 2.0])
    goods = np.array([500.0, 9.0, 200.0])
    bads = np.array([10.0, 1.0, 50.0])  # middle bin: 10 rows, 10% bad; neighbours 2% and 20%
    out_edges, out_goods, _ = merge_small_bins((edges, goods, bads), min_rows=50)
    # Gap to the left is 8 points, to the right 10 points, so it joins the LEFT bin.
    assert out_edges.tolist() == [2.0]
    assert out_goods.tolist() == [509.0, 200.0]


def test_make_monotonic_pools_violators():
    edges = np.array([1.0, 2.0, 3.0])
    goods = np.array([90.0, 80.0, 90.0, 60.0])
    bads = np.array([10.0, 20.0, 10.0, 40.0])  # 10%, 20%, 10%, 40%: one dip
    out_edges, out_goods, out_bads = make_monotonic((edges, goods, bads), INCREASING)
    rate = out_bads / (out_goods + out_bads)
    assert (np.diff(rate) > 0).all()
    assert out_edges.tolist() == [1.0, 3.0]  # bins 2 and 3 pooled into 15%


def test_coarse_edges_give_monotonic_and_large_enough_bins():
    rng = np.random.default_rng(42)
    x = rng.uniform(18, 90, size=20_000)
    y = (rng.uniform(size=x.size) < 0.25 - 0.0025 * (x - 18)).astype(float)  # risk falls with age
    x[:300] = np.nan

    edges, direction = coarse_edges(
        x, y, fine_bins=20, min_bin_share=0.05, max_bins=4, smoothing=0.5
    )

    assert direction == DECREASING
    goods, bads = bin_counts(x, y, edges)
    rows, rate = (goods + bads)[:-1], (bads / (goods + bads))[:-1]  # numeric bins only
    assert (rows >= 0.05 * x.size).all()
    assert len(rows) <= 4
    assert (np.diff(rate) < 0).all()


def test_cap_removes_the_least_informative_cut_and_keeps_order():
    edges = np.array([1.0, 2.0, 3.0])
    goods = np.array([95.0, 94.0, 85.0, 60.0])
    bads = np.array([5.0, 6.0, 15.0, 40.0])  # 5%, 6%, 15%, 40%: the 5 -> 6 cut says least
    out_edges, out_goods, out_bads = cap_bins((edges, goods, bads), max_bins=3, smoothing=0.5)
    assert out_edges.tolist() == [2.0, 3.0]
    assert (np.diff(out_bads / (out_goods + out_bads)) > 0).all()


def test_unknown_direction_is_rejected():
    bins = (np.array([1.0]), np.array([5.0, 5.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="Unknown direction"):
        make_monotonic(bins, "sideways")
