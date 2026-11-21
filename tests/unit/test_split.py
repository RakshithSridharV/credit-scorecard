from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.data.load import TARGET
from scorecard.data.split import ROW_ID, load_split, save_splits, split_data


@pytest.fixture
def frame() -> pd.DataFrame:
    """1,000 rows, exactly 10% bad, with a row_id index like the real data."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({TARGET: [1] * 100 + [0] * 900, "x": rng.normal(size=1000)})
    df.index.name = ROW_ID
    return df


def _split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    return split_data(df, valid_size=0.15, test_size=0.15, seed=seed)


def test_sizes_are_70_15_15(frame):
    s = _split(frame)
    assert (len(s["train"]), len(s["valid"]), len(s["test"])) == (700, 150, 150)


def test_splits_are_disjoint_and_complete(frame):
    s = _split(frame)
    ids = [set(part.index) for part in s.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert set().union(*ids) == set(frame.index)


def test_each_split_keeps_the_bad_rate(frame):
    for part in _split(frame).values():
        assert part[TARGET].mean() == pytest.approx(0.10, abs=0.005)


def test_same_seed_same_split_different_seed_different_split(frame):
    a, b, c = _split(frame, 42), _split(frame, 42), _split(frame, 7)
    assert a["test"].index.equals(b["test"].index)
    assert not a["test"].index.equals(c["test"].index)


def test_input_not_mutated(frame):
    before = frame.copy()
    _split(frame)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(("valid", "test"), [(0, 0.15), (0.15, 0), (0.5, 0.5), (-0.1, 0.2)])
def test_rejects_bad_sizes(frame, valid, test):
    with pytest.raises(ValueError, match="Bad split sizes"):
        split_data(frame, valid_size=valid, test_size=test, seed=42)


def test_save_and_load_round_trip_keeps_row_id(frame, tmp_path):
    s = _split(frame)
    save_splits(s, tmp_path)
    back = load_split("valid", tmp_path)
    assert back.index.name == ROW_ID
    pd.testing.assert_frame_equal(back, s["valid"])


def test_load_split_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        load_split("holdout", tmp_path)
