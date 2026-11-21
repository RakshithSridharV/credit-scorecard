"""Scoring core: validation, skew guards, decision rule, reasons, artifact integrity."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scorecard.api.scoring import (
    ArtifactMismatch,
    InvalidApplicant,
    ScoringService,
    validate,
    verify_artifacts,
)
from scorecard.data.clean import clean
from scorecard.data.load import FEATURES
from scorecard.explain.reason_codes import code_texts
from scorecard.integrity import sha256
from scorecard.models.scorecard import Scorecard

UTIL, AGE, DEBT = "RevolvingUtilizationOfUnsecuredLines", "age", "DebtRatio"
D90, INCOME = "NumberOfTimes90DaysLate", "MonthlyIncome"
BINNING = {"fine_bins": 10, "min_bin_share": 0.05, "max_bins": 4, "smoothing": 0.5}


@pytest.fixture
def service(scoring_service):
    return scoring_service


# ------------------------------------------------------------------ the decision
def test_score_matches_the_offline_scorecard_on_cleaned_input(applicant, service):
    row = applicant(**{UTIL: 0.7})
    d = service.score(row)
    offline = service.scorecard.score(clean(pd.DataFrame([row])))[0]
    assert d.score == offline
    assert d.pd == pytest.approx(service.scorecard.score_to_pd_.predict([offline])[0])


def test_decision_follows_the_cutoff_and_only_declines_get_reasons(applicant, service):
    good = service.score(applicant())
    bad = service.score(applicant(**{UTIL: 1.1, D90: 3, DEBT: 1.4, AGE: 22}))
    assert good.score >= service.cutoff and good.decision == "approve" and good.reasons == ()
    assert bad.score < service.cutoff and bad.decision == "decline"
    assert 1 <= len(bad.reasons) <= 4
    assert all(r.text == code_texts()[r.code] and r.points_lost > 0 for r in bad.reasons)


def test_batch_equals_one_by_one_and_is_deterministic(applicant, service):
    rows = [applicant(), applicant(**{UTIL: 0.9}), applicant(**{D90: 2, AGE: 25})]
    batch = service.score_batch(pd.DataFrame(rows))
    assert batch == [service.score(r) for r in rows]
    assert batch == service.score_batch(pd.DataFrame(rows))


def test_to_dict_is_json_ready(applicant, service):
    d = service.score(applicant(**{UTIL: 1.1, D90: 3}))
    payload = json.loads(json.dumps(d.to_dict()))
    assert payload["decision"] == "decline" and payload["reasons"][0]["code"]


# ------------------------------------------------------------------ skew guards
def test_income_zero_is_scored_like_income_not_reported(applicant, service):
    """D-003/D-004 at serving time: income 0 makes DebtRatio untrustworthy."""
    zero = service.score(applicant(**{INCOME: 0}))
    unreported = service.score(applicant(**{INCOME: None}))
    reported = service.score(applicant(**{INCOME: 4000}))
    assert zero.score == unreported.score
    assert zero.score != reported.score  # DebtRatio moved to its Missing bin


def test_optional_fields_may_be_null_or_absent(applicant, service):
    absent = {k: v for k, v in applicant().items() if k not in (INCOME, "NumberOfDependents")}
    nulls = applicant(**{INCOME: None, "NumberOfDependents": None})
    assert service.score(absent) == service.score(nulls)


def test_past_due_special_code_is_accepted_and_cleaned(applicant, service):
    # 98 is a bureau code (D-006), not 98 late payments: it must not be scored as a count.
    coded = service.score(applicant(**{D90: 98}))
    huge = service.score(applicant(**{D90: 50}))
    assert coded.score != huge.score


# ------------------------------------------------------------------ rejections
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({AGE: 17}, "age must be at least 18"),
        ({AGE: None}, "age is required"),
        ({UTIL: -0.1}, "must not be negative"),
        ({DEBT: float("inf")}, "must be finite"),
        ({D90: 1.5}, "must be a whole number"),
        ({"MonthlyIncom": 5000}, "unknown fields"),
    ],
)
def test_invalid_input_is_refused_not_scored(applicant, service, change, message):
    with pytest.raises(InvalidApplicant, match=message):
        service.score(applicant(**change))


def test_age_18_is_accepted(applicant, service):
    assert service.score(applicant(**{AGE: 18})).score > 0


def test_missing_required_field_is_named(applicant):
    row = {k: v for k, v in applicant().items() if k != UTIL}
    with pytest.raises(InvalidApplicant, match=f"missing required fields: \\['{UTIL}'\\]"):
        validate(pd.DataFrame([row]))


def test_every_problem_is_reported_at_once(applicant):
    with pytest.raises(InvalidApplicant) as exc:
        validate(pd.DataFrame([applicant(**{AGE: 10, UTIL: -1, D90: 0.5})]))
    assert len(exc.value.errors) == 3


def test_non_numeric_value_is_refused(applicant):
    with pytest.raises(InvalidApplicant, match="must be a number"):
        validate(pd.DataFrame([applicant(**{AGE: "forty"})]))


def test_one_bad_row_rejects_the_whole_batch(applicant, service):
    rows = pd.DataFrame([applicant(), applicant(**{AGE: 5})])
    with pytest.raises(InvalidApplicant, match="row 1: age"):
        service.score_batch(rows)


# ------------------------------------------------------------------ integrity
def _lock(tmp_path, prints):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"runs": [{"fingerprints": prints}]}), encoding="utf-8")
    return lock


def test_verify_passes_for_the_tested_file_and_fails_after_a_change(tmp_path):
    f = tmp_path / "scorecard.json"
    f.write_text("{}", encoding="utf-8")
    lock = _lock(tmp_path, {"scorecard": sha256(f)})
    verify_artifacts({"scorecard": f}, lock)
    f.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ArtifactMismatch, match="scorecard"):
        verify_artifacts({"scorecard": f}, lock)


def test_verify_refuses_when_there_is_no_lock(tmp_path):
    with pytest.raises(ArtifactMismatch, match="never evaluated"):
        verify_artifacts({}, tmp_path / "missing.json")


def test_service_refuses_an_uncalibrated_scorecard(service):
    raw = Scorecard(service.scorecard.features, BINNING)
    for attr in ("encoder_", "intercept_", "coef_", "points_"):
        setattr(raw, attr, getattr(service.scorecard, attr))
    with pytest.raises(ArtifactMismatch, match="master scale"):
        ScoringService(raw, service.reference, 500, 4, 5.0, "x")


def test_service_refuses_a_reference_for_other_features(service):
    with pytest.raises(ArtifactMismatch, match="different features"):
        ScoringService(service.scorecard, {"age": 1.0}, 500, 4, 5.0, "x")


def test_required_and_optional_cover_every_raw_feature():
    from scorecard.api.scoring import OPTIONAL, REQUIRED

    assert sorted((*REQUIRED, *OPTIONAL)) == sorted(FEATURES)
