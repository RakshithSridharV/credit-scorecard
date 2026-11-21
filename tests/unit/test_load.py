from __future__ import annotations

import pandas as pd
import pytest

from scorecard.data.load import COLUMNS, ID_COLUMN, load_raw


def test_load_raw_drops_id_and_keeps_expected_columns(tmp_path, tiny_raw):
    path = tmp_path / "raw.csv"
    tiny_raw.insert(0, ID_COLUMN, range(1, len(tiny_raw) + 1))
    tiny_raw.to_csv(path, index=False)

    df = load_raw(path)

    assert list(df.columns) == COLUMNS
    assert len(df) == len(tiny_raw)


def test_load_raw_rejects_missing_column(tmp_path, tiny_raw):
    path = tmp_path / "raw.csv"
    broken = tiny_raw.drop(columns=["age"])
    broken.insert(0, ID_COLUMN, range(len(broken)))
    broken.to_csv(path, index=False)

    with pytest.raises(ValueError, match="age"):
        load_raw(path)


def test_load_raw_rejects_extra_column(tmp_path, tiny_raw):
    path = tmp_path / "raw.csv"
    tiny_raw.insert(0, ID_COLUMN, range(len(tiny_raw)))
    tiny_raw["surprise"] = 1
    tiny_raw.to_csv(path, index=False)

    with pytest.raises(ValueError, match="surprise"):
        load_raw(path)


def test_load_raw_requires_id_column(tmp_path, tiny_raw):
    # A file without the row-number column is not the file we audited.
    path = tmp_path / "raw.csv"
    pd.DataFrame(tiny_raw).to_csv(path, index=False)

    with pytest.raises(KeyError):
        load_raw(path)
