"""The shipped model, end to end. Skipped when the artifacts or splits are absent
(e.g. a fresh clone before `make train`), so unit tests never depend on them.

In CI, SCORECARD_REQUIRE_ARTIFACTS=1 turns a missing model into a FAILURE, not
a skip: a green run must mean these tests ran (D-027). The skew test needs the
Kaggle splits, which are never in git, so it stays a (reported) skip in CI.
"""

from __future__ import annotations

import os

import pytest

from scorecard.api.scoring import ScoringService
from scorecard.config import load_config, resolve_path
from scorecard.data.clean import drop_invalid_rows
from scorecard.data.load import FEATURES
from scorecard.data.split import load_split

CFG = load_config()
LOCK = resolve_path(CFG["evaluation"]["lock"])
SCORECARD = resolve_path(CFG["artifacts"]["scorecard"])
REQUIRE = os.environ.get("SCORECARD_REQUIRE_ARTIFACTS") == "1"
needs_model = pytest.mark.skipif(
    not (LOCK.exists() and SCORECARD.exists()) and not REQUIRE,
    reason="no trained model / test lock",
)


@pytest.fixture(scope="module")
def service() -> ScoringService:
    return ScoringService.from_config(CFG, verify=True)


@needs_model
def test_shipped_model_loads_verified_with_the_d018_cutoff(service):
    assert service.cutoff == 549
    assert len(service.model_version) == 12


@needs_model
def test_no_training_serving_skew_on_valid(service):
    """Service on RAW valid rows == offline scorecard on the CLEANED valid file.

    This is the skew test: it proves the live path transforms inputs exactly as
    the training pipeline did, for every one of ~22,500 real rows.
    """
    interim = resolve_path(CFG["data"]["interim_dir"])
    processed = resolve_path(CFG["data"]["processed_dir"])
    if not (interim / "valid.csv").exists() or not (processed / "valid.csv").exists():
        pytest.skip("splits not built")
    raw = drop_invalid_rows(load_split("valid", processed))
    cleaned = load_split("valid", interim).loc[raw.index]

    live = [d.score for d in service.score_batch(raw[FEATURES])]
    offline = service.scorecard.score(cleaned)
    assert live == offline.tolist()


@needs_model
def test_income_zero_vs_reported_income_example(service):
    """The D-023 walkthrough example: identical applicants except income."""
    base = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.3,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }
    a = service.score({**base, "MonthlyIncome": 0})
    b = service.score({**base, "MonthlyIncome": 4000})
    pts = service.scorecard.points_["DebtRatio"]
    assert a.score - b.score == pts[-1] - pts[1]  # Missing bin vs the (0.022, 0.425] bin
