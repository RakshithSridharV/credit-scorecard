"""D-009 duplicates check: the definition, the lock guard and the addendum record."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scorecard.evaluation.duplicates import (
    auc_with_without,
    check_addendum,
    keep_mask,
    record_addendum,
    row_keys,
    verify_frozen,
)
from scorecard.integrity import sha256


def test_row_keys_treat_missing_as_equal():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 2.0], "b": [5, 7, 7, 5]})
    k = row_keys(df, ["a", "b"])
    assert k.iloc[1] == k.iloc[2]  # both missing `a`, same `b`: twins
    assert k.iloc[0] != k.iloc[3]


def test_keep_mask_drops_train_twins_and_repeats_but_keeps_one_copy():
    train = pd.DataFrame({"x": [1, 2]})
    split = pd.DataFrame({"x": [1, 3, 3, 3, 4]}, index=[10, 11, 12, 13, 14])
    m = keep_mask(row_keys(train, ["x"]), row_keys(split, ["x"]))
    assert m["in_train"].tolist() == [True, False, False, False, False]
    assert m["repeat_in_split"].tolist() == [False, False, True, True, False]
    assert m["keep"].tolist() == [False, True, False, False, True]  # one copy of 3 stays
    assert m.index.tolist() == [10, 11, 12, 13, 14]


def test_auc_with_without_reports_the_change():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.95, 0.3])
    keep = np.array([True, True, True, True, False, False])  # drop the two misranked rows
    out = auc_with_without(y, p, keep)
    assert out["auc_without_duplicates"] == 1.0
    assert out["auc_change"] == pytest.approx(out["auc_without_duplicates"] - out["auc_all_rows"])
    assert out["gini_change"] == pytest.approx(2 * out["auc_change"])


def _lock(tmp_path, prints):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"runs": [{"at": "t0", "fingerprints": prints}]}), encoding="utf-8")
    return lock


def test_verify_frozen_refuses_a_changed_model(tmp_path):
    model = tmp_path / "scorecard.json"
    model.write_text("tested", encoding="utf-8")
    lock = _lock(tmp_path, {"scorecard": sha256(model)})
    verify_frozen(lock, {"scorecard": model})  # unchanged: allowed
    model.write_text("rebuilt differently", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ from the tested version"):
        verify_frozen(lock, {"scorecard": model})


def test_addendum_is_recorded_once_and_leaves_the_official_run_alone(tmp_path):
    lock = _lock(tmp_path, {"scorecard": "abc"})
    check_addendum(lock, force=False)  # never run: allowed
    record_addendum(lock, {"scorecard": "abc"}, "reports/duplicate_check.md")
    with pytest.raises(RuntimeError, match="already run"):
        check_addendum(lock, force=False)
    check_addendum(lock, force=True)  # allowed when forced
    state = json.loads(lock.read_text(encoding="utf-8"))
    assert state["runs"] == [{"at": "t0", "fingerprints": {"scorecard": "abc"}}]
    assert state["addenda"][0]["check"] == "D-009 duplicates"
