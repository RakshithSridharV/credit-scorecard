"""The what-if app, run headless with Streamlit's own test harness (D-030).

Uses the real, lock-verified model, like the integration tests: the app is only
worth testing against the model it will actually show.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit", reason="app extra not installed: pip install -e .[app]")
from streamlit.testing.v1 import AppTest

from scorecard.config import PROJECT_ROOT, load_config, resolve_path

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
CFG = load_config()
HAVE_MODEL = (
    resolve_path(CFG["artifacts"]["scorecard"]).exists()
    and resolve_path(CFG["evaluation"]["lock"]).exists()
)
REQUIRE = os.environ.get("SCORECARD_REQUIRE_ARTIFACTS") == "1"  # CI: missing model = failure
pytestmark = pytest.mark.skipif(not HAVE_MODEL and not REQUIRE, reason="no trained model")

UTIL, D90, D30 = (
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
)


def run(**changes) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    for key, value in changes.items():
        at.number_input(key=key).set_value(value)
    return at.run() if changes else at


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_opens_on_the_default_applicant_as_an_approval():
    at = run()
    assert not at.exception
    assert metric(at, "Score") == "594"
    assert metric(at, "Outcome") == "APPROVE"
    assert not at.table  # approvals carry no reasons


def test_says_it_is_a_simulation_and_records_nothing():
    at = run()
    assert any("Simulation only" in w.value for w in at.warning)


def test_matches_the_readme_decline_example():
    at = run(**{UTIL: 0.95, D90: 1, D30: 2, "age": 27})
    assert metric(at, "Score") == "487"
    assert metric(at, "Outcome") == "DECLINE"
    reasons = at.table[0].value
    assert list(reasons["code"]) == ["DLQ_90", "UTIL_HIGH", "DLQ_30", "AGE"]


def test_what_if_shows_the_change_in_score():
    at = run(**{UTIL: 0.95, D90: 1, D30: 2, "age": 27})
    at.multiselect(key="whatif_fields").set_value([UTIL]).run()
    at.number_input(key=f"whatif_{UTIL}").set_value(0.25).run()
    assert not at.exception
    assert metric(at, "What-if score") == "519"  # +32 points: 61 -> 93 for utilisation
    assert metric(at, "What-if outcome") == "DECLINE"  # still below 549
