"""Underwriter what-if workbench (D-030).

    streamlit run app/streamlit_app.py

SIMULATION ONLY. This app calls the same ScoringService as the API, so it can
never disagree with it, but it records nothing and issues no decision. Credit
decisions go only through POST /score, which writes the audit log (D-025).
Keeping the two channels apart means every real decision has exactly one record.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from scorecard.api.scoring import (
    OPTIONAL,
    ArtifactMismatch,
    Decision,
    InvalidApplicant,
    ScoringService,
)
from scorecard.data.load import FEATURES

# field -> (label, kind, help). kind: "ratio" | "count" | "age" | "money"
FIELDS: dict[str, tuple[str, str, str]] = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "Revolving utilisation",
        "ratio",
        "Card and revolving balances / credit limits (0.30 = 30%)",
    ),
    "age": ("Age", "age", "Years. Applicants under 18 are refused"),
    "NumberOfTime30-59DaysPastDueNotWorse": ("Times 30-59 days late", "count", "Last 2 years"),
    "NumberOfTime60-89DaysPastDueNotWorse": ("Times 60-89 days late", "count", "Last 2 years"),
    "NumberOfTimes90DaysLate": ("Times 90+ days late", "count", "Ever reported"),
    "DebtRatio": ("Debt ratio", "ratio", "Monthly debt payments / monthly income"),
    "MonthlyIncome": ("Monthly income (USD)", "money", "Leave unreported if not provided"),
    "NumberOfOpenCreditLinesAndLoans": ("Open credit lines and loans", "count", ""),
    "NumberRealEstateLoansOrLines": (
        "Real-estate loans",
        "count",
        "Mortgages and home-equity lines",
    ),
    "NumberOfDependents": ("Dependents", "count", "Excluding the applicant"),
}
assert set(FIELDS) == set(FEATURES), "every bureau field needs an input"

# A typical approved applicant (score 594), so the page opens on a sensible case.
DEFAULTS: dict[str, Any] = {
    "RevolvingUtilizationOfUnsecuredLines": 0.25,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfTimes90DaysLate": 0,
    "DebtRatio": 0.30,
    "MonthlyIncome": 5400.0,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfDependents": 2,
}


@st.cache_resource  # load and verify the model once per server process, not per click
def get_service() -> ScoringService:
    return ScoringService.from_config()  # refuses a model that differs from the tested one


def field_input(field: str, default: Any, key: str) -> Any:
    """One widget per field. Bounds mirror the API's rules, so most bad input
    cannot even be entered; the core still validates (defence in depth)."""
    label, kind, help_ = FIELDS[field]
    if field in OPTIONAL and st.checkbox(
        f"{label}: not reported", value=default is None, key=f"{key}_na"
    ):
        return None  # missing is a valid answer for these two fields (D-003, D-005)
    start = DEFAULTS[field] if default is None else default
    if kind == "ratio":
        return st.number_input(
            label, min_value=0.0, value=float(start), step=0.05, format="%.2f", help=help_, key=key
        )
    if kind == "money":
        return st.number_input(
            label, min_value=0.0, value=float(start), step=100.0, help=help_, key=key
        )
    low = 18 if kind == "age" else 0
    return st.number_input(
        label,
        min_value=low,
        max_value=120 if kind == "age" else 99,
        value=int(start),
        step=1,
        help=help_ or None,
        key=key,
    )


def show_decision(d: Decision, cutoff: int) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", d.score, help=f"Cutoff {cutoff}: approve at or above")
    c2.metric("Probability of default", f"{d.pd:.1%}", help="90+ days past due within 2 years")
    c3.metric("Outcome", d.decision.upper())
    gap = cutoff - d.score
    if gap > 0:
        st.caption(f"{gap} points below the cutoff.")
    else:
        st.caption(f"{-gap} points above the cutoff.")


def main() -> None:
    st.set_page_config(page_title="Scorecard what-if", layout="wide")
    st.title("Credit scorecard: underwriter what-if")
    st.warning(
        "Simulation only. Nothing on this page is a credit decision or is recorded. "
        "Decisions are issued only through the scoring API, which writes the audit log.",
        icon="⚠️",
    )

    try:
        service = get_service()
    except (ArtifactMismatch, FileNotFoundError) as exc:  # untested or missing model: do not score
        st.error(f"The model could not be loaded: {type(exc).__name__}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Application")
        applicant = {f: field_input(f, DEFAULTS[f], key=f) for f in FIELDS}
        st.divider()
        st.caption(
            f"Model `{service.model_version}` (verified against the test lock) · "
            f"cutoff {service.cutoff} · US-data method demo, not for real lending"
        )

    try:
        decision = service.score(applicant)
        breakdown = service.breakdown(applicant)
    except InvalidApplicant as exc:
        st.error("The application cannot be scored:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
        st.stop()

    st.subheader("Result")
    show_decision(decision, service.cutoff)
    if decision.reasons:
        st.markdown("**Principal reasons for decline** (as they would appear on the notice)")
        st.table(
            pd.DataFrame(
                [(r.code, r.text, r.points_lost) for r in decision.reasons],
                columns=["code", "reason", "points lost vs typical"],
            )
        )

    st.subheader("Where the score comes from")
    st.caption(
        "Points per feature add up exactly to the score. 'Typical' is the average applicant "
        "in the training data: the reference used for reason codes. Only the 7 scored "
        "features appear; income is not scored but decides whether the debt ratio is usable."
    )
    table = breakdown.rename(index=lambda f: FIELDS[f][0])
    st.dataframe(table, width="stretch")

    st.subheader("What-if")
    changed = st.multiselect(
        "Fields to change", list(FIELDS), format_func=lambda f: FIELDS[f][0], key="whatif_fields"
    )
    if not changed:
        st.caption("Pick one or more fields to see how the score and outcome would move.")
        return
    alternative = dict(applicant)
    cols = st.columns(2)
    for i, f in enumerate(changed):
        with cols[i % 2]:
            alternative[f] = field_input(f, applicant[f], key=f"whatif_{f}")

    try:
        alt = service.score(alternative)
        alt_breakdown = service.breakdown(alternative)
    except InvalidApplicant as exc:
        st.error("The what-if case cannot be scored:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("What-if score", alt.score, delta=alt.score - decision.score)
    c2.metric(
        "What-if PD", f"{alt.pd:.1%}", delta=f"{alt.pd - decision.pd:+.1%}", delta_color="inverse"
    )
    c3.metric("What-if outcome", alt.decision.upper())
    if alt.decision != decision.decision:
        st.success(f"The outcome changes: {decision.decision} -> {alt.decision}.")

    moved = pd.DataFrame({"before": breakdown["points"], "after": alt_breakdown["points"]}).assign(
        change=lambda t: t["after"] - t["before"]
    )
    moved = moved[moved["change"] != 0].rename(index=lambda f: FIELDS[f][0])
    if moved.empty:
        st.caption(
            "No points change: the new values fall in the same bins, or the field is not scored."
        )
    else:
        st.table(moved)


main()
