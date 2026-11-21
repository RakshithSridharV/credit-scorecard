"""Test-set evaluation: the once-only lock, fingerprints, intervals, R-1 flag."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scorecard.pipelines.evaluate import (
    bootstrap_ci,
    check_lock,
    record_run,
    sha256,
    special_code_rows,
)


def test_sha256_changes_when_a_file_changes(tmp_path):
    f = tmp_path / "model.json"
    f.write_text("a", encoding="utf-8")
    before = sha256(f)
    f.write_text("b", encoding="utf-8")
    assert sha256(f) != before and len(before) == 64


def test_first_run_passes_and_second_is_refused(tmp_path):
    lock = tmp_path / "lock.json"
    check_lock(lock, force=False)  # no lock yet: allowed
    record_run(lock, {"scorecard": "abc"})
    with pytest.raises(RuntimeError, match="already evaluated"):
        check_lock(lock, force=False)


def test_forced_rerun_is_recorded_and_compared(tmp_path):
    lock = tmp_path / "lock.json"
    record_run(lock, {"scorecard": "abc"})
    check_lock(lock, force=True)  # allowed when forced
    record_run(lock, {"scorecard": "xyz"})
    runs = json.loads(lock.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 2
    assert runs[1]["forced"] is True
    assert runs[1]["same_artifacts_as_first"] is False  # the model changed between looks


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2_000)
    p = y + rng.normal(0, 1.0, 2_000)
    ci = bootstrap_ci(y, p, n=200, seed=1)
    auc = roc_auc_score(y, p)
    assert ci.loc["auc", "ci_low"] < auc < ci.loc["auc", "ci_high"]
    assert ci.loc["gini", "ci_low"] == pytest.approx(2 * ci.loc["auc", "ci_low"] - 1)
    pd.testing.assert_frame_equal(ci, bootstrap_ci(y, p, n=200, seed=1))


def test_special_code_rows_are_the_missing_90_day_rows():
    df = pd.DataFrame({"NumberOfTimes90DaysLate": [0.0, np.nan, 2.0]})
    assert special_code_rows(df).tolist() == [False, True, False]
