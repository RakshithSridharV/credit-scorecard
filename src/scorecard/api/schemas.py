"""HTTP contract: what a client may send and what it gets back (D-024).

Wire names are the bureau field names used everywhere else in the project
(data dictionary, model, reports), so there is no renaming layer to get wrong.
Python attributes are snake_case; `alias` maps one to the other.

The schema is strict on purpose. A credit decision should never rest on a
guess about what the client meant: "45" (a string), true (a boolean) or 1.5
late payments are refused with 422, not coerced.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scorecard.data.clean import MIN_AGE

_STRICT = ConfigDict(
    strict=True,  # no "45" -> 45 coercion; ints are still accepted where a float is expected
    extra="forbid",  # unknown field (e.g. a typo) -> 422, never silently ignored
    allow_inf_nan=False,  # Infinity / NaN are not amounts
    populate_by_name=False,  # clients must use the wire (bureau) names
)


class ApplicantIn(BaseModel):
    """One applicant's raw bureau fields, exactly as in the training data."""

    model_config = _STRICT

    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines",
        ge=0,
        description="Revolving balances / credit limits. Values > 10 are treated as missing (D-007).",
    )
    age: int = Field(ge=MIN_AGE, description="Age in years. Under 18 is refused (D-008).")
    past_due_30_59: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        ge=0,
        description="Times 30-59 days late in 2 years. Bureau codes 96/98 = unknown (D-006).",
    )
    debt_ratio: float = Field(
        alias="DebtRatio",
        ge=0,
        description="Monthly debt payments / monthly income. Ignored if income is 0, 1 or null.",
    )
    monthly_income: float | None = Field(
        default=None,
        alias="MonthlyIncome",
        ge=0,
        description="Not scored, but required to judge DebtRatio (D-004). Null = not reported.",
    )
    open_credit_lines: int = Field(alias="NumberOfOpenCreditLinesAndLoans", ge=0)
    past_due_90: int = Field(alias="NumberOfTimes90DaysLate", ge=0)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0)
    past_due_60_89: int = Field(alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0)
    dependents: int | None = Field(default=None, alias="NumberOfDependents", ge=0)

    def to_raw(self) -> dict[str, float | int | None]:
        """The dict the scoring core expects: bureau names, None = not reported."""
        return self.model_dump(by_alias=True)


class ReasonOut(BaseModel):
    code: str
    text: str
    points_lost: float


class DecisionOut(BaseModel):
    decision_id: str = Field(description="Audit record id: quote it in letters and disputes")
    score: int
    pd: float = Field(description="Calibrated probability of 90+ days past due within 2 years")
    decision: Literal["approve", "decline"]
    cutoff: int
    reasons: list[ReasonOut] = Field(description="Adverse-action reasons; empty when approved")
    model_version: str


class HealthOut(BaseModel):
    status: Literal["ok"]
    model_version: str
    cutoff: int


class ReadyOut(BaseModel):
    status: Literal["ready"]
