"""POST /score: the contract (strict 422s) and that it returns the core's decision."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from scorecard.api.scoring import InvalidApplicant
from scorecard.data.load import FEATURES

UTIL, AGE, D90 = "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTimes90DaysLate"
HIGH_RISK = {UTIL: 1.1, D90: 3, "DebtRatio": 1.4, AGE: 22}


def test_response_is_exactly_the_core_decision(client, scoring_service, applicant):
    body = applicant()
    r = client.post("/score", json=body)
    assert r.status_code == 200
    got = r.json()
    got.pop("decision_id")
    assert got == scoring_service.score(body).to_dict()


# ------------------------------------------------------------------ audit log (D-025)
def test_every_decision_is_recorded_with_inputs_and_model(client, repository, applicant):
    body = applicant(**HIGH_RISK, MonthlyIncome=None)
    r = client.post("/score", json=body).json()
    row = repository.get(r["decision_id"])
    assert row is not None
    assert (row.score, row.decision, row.model_version, row.cutoff) == (
        r["score"],
        r["decision"],
        r["model_version"],
        r["cutoff"],
    )
    assert row.reasons == r["reasons"]
    assert row.inputs == body  # exactly as received; None = not reported


def test_rejected_requests_are_not_decisions_and_are_not_recorded(client, repository, applicant):
    client.post("/score", json=applicant(age=17))
    assert repository.count() == 0


def test_no_record_means_no_decision_503(client, repository, applicant, monkeypatch):
    def broken(*_):
        raise OperationalError("INSERT ...", {}, Exception("disk full"))

    monkeypatch.setattr(repository, "record", broken)
    r = client.post("/score", json=applicant())
    assert r.status_code == 503
    assert r.headers["retry-after"] == "5"
    assert "score" not in r.json() and "decision" not in r.json()  # nothing issued


def test_each_decision_gets_its_own_id(client, repository, applicant):
    ids = {client.post("/score", json=applicant()).json()["decision_id"] for _ in range(3)}
    assert len(ids) == 3 and repository.count() == 3


def test_decline_comes_with_reasons(client, applicant):
    r = client.post("/score", json=applicant(**HIGH_RISK)).json()
    assert r["decision"] == "decline"
    assert 1 <= len(r["reasons"]) <= 4
    assert {"code", "text", "points_lost"} <= r["reasons"][0].keys()


def test_optional_fields_may_be_omitted_or_null(client, applicant):
    body = applicant(MonthlyIncome=None, NumberOfDependents=None)
    omitted = {k: v for k, v in body.items() if v is not None}
    a, b = client.post("/score", json=body), client.post("/score", json=omitted)
    assert a.status_code == b.status_code == 200
    a_body, b_body = a.json(), b.json()
    assert a_body.pop("decision_id") != b_body.pop("decision_id")  # two separate decisions
    assert a_body == b_body


def test_integer_amounts_are_accepted_for_float_fields(client, applicant):
    assert client.post("/score", json=applicant(DebtRatio=0, MonthlyIncome=5000)).status_code == 200


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({AGE: 17}, AGE),
        ({AGE: "45"}, AGE),  # strict: no string -> int coercion
        ({AGE: True}, AGE),  # strict: a boolean is not an age
        ({D90: 1.5}, D90),  # counts are whole numbers
        ({UTIL: -0.2}, UTIL),
        ({"MonthlyIncom": 5000}, "MonthlyIncom"),  # unknown field (typo)
    ],
)
def test_bad_input_is_422_and_names_the_field(client, applicant, change, field):
    r = client.post("/score", json=applicant(**change))
    assert r.status_code == 422
    assert any(field in err["loc"] for err in r.json()["detail"])


def test_missing_required_field_is_422(client, applicant):
    body = applicant()
    del body[UTIL]
    r = client.post("/score", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", UTIL]


def test_infinity_is_422(client, applicant):
    raw = str(applicant()).replace("'", '"').replace("0.2", "Infinity")  # DebtRatio
    r = client.post("/score", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_core_rejection_is_reported_as_422(client, scoring_service, applicant, monkeypatch):
    def refuse(_):
        raise InvalidApplicant(["row 0: something the schema missed"])

    monkeypatch.setattr(scoring_service, "score", refuse)
    r = client.post("/score", json=applicant())
    assert r.status_code == 422
    err = {
        "loc": ["body"],
        "msg": "row 0: something the schema missed",
        "type": "invalid_applicant",
    }
    assert r.json() == {"detail": [err]}


def test_openapi_contract_uses_the_bureau_field_names(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ApplicantIn"]
    assert sorted(schema["properties"]) == sorted(FEATURES)
    assert sorted(schema["required"]) == sorted(
        f for f in FEATURES if f not in ("MonthlyIncome", "NumberOfDependents")
    )
    assert schema["properties"][AGE]["minimum"] == 18


def test_validation_errors_never_echo_the_applicants_values(client, applicant):
    r = client.post("/score", json=applicant(age=17, MonthlyIncome=-123456))
    assert r.status_code == 422
    assert all("input" not in err for err in r.json()["detail"])
    assert "123456" not in r.text
