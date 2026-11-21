"""Score applicants: raw bureau fields in, a credit decision out (D-023).

This is the only place a live decision is made. It has no web code, so the
FastAPI routes, the Streamlit demo and any batch job all call the same thing,
and it can be unit-tested without a server.

Guards against training-serving skew:
  * the same `clean()` as training (D-003 to D-008), not a re-implementation;
  * only the artifacts that were fingerprinted in the test lock (D-020) are
    served: a changed or untested model file stops the service at start-up;
  * inputs the model was never trained on (age < 18, negative counts, unknown
    fields) are rejected, never scored.

    from scorecard.api.scoring import ScoringService
    service = ScoringService.from_config()
    service.score({"age": 45, ...})
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from scorecard.config import load_config, resolve_path
from scorecard.data.clean import AGE, MIN_AGE, clean
from scorecard.data.load import FEATURES, PAST_DUE_COLUMNS
from scorecard.evaluation.cutoff import economics_from_config, policy_cutoff
from scorecard.explain.reason_codes import (
    applicant_points,
    code_texts,
    load_reference,
    top_reasons,
)
from scorecard.features.binning import assign_bins, bin_labels
from scorecard.integrity import sha256
from scorecard.models.scorecard import Scorecard

# Missing in the training data too (D-003, D-005), so "not reported" is a valid
# answer with its own bin. Every other field was always present: a gap there is
# a broken request, not information.
OPTIONAL = ("MonthlyIncome", "NumberOfDependents")
REQUIRED = tuple(f for f in FEATURES if f not in OPTIONAL)
COUNTS = (
    *PAST_DUE_COLUMNS,
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
)
# The artifacts a decision depends on, by their key in the test lock.
SERVED_ARTIFACTS = ("scorecard", "reason_reference")

Applicant = Mapping[str, float | int | None]


class InvalidApplicant(ValueError):
    """The request cannot be scored. Carries every problem found, not just the first."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class ArtifactMismatch(RuntimeError):
    """The model files on disk are not the ones that were tested."""


@dataclass(frozen=True)
class ReasonItem:
    code: str
    text: str  # applicant-facing wording (Regulation B)
    points_lost: float  # vs the train-average applicant (D-019)


@dataclass(frozen=True)
class Decision:
    score: int
    pd: float  # calibrated 2-year PD from the master scale
    decision: Literal["approve", "decline"]
    cutoff: int
    reasons: tuple[ReasonItem, ...]  # empty when approved
    model_version: str  # first 12 hex of the scorecard's SHA-256

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "pd": self.pd,
            "decision": self.decision,
            "cutoff": self.cutoff,
            "reasons": [r.__dict__ for r in self.reasons],
            "model_version": self.model_version,
        }


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def validate(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the rows as floats in FEATURES order, or raise InvalidApplicant.

    The HTTP layer validates types too; this runs again here because batch and
    Streamlit callers never pass through HTTP (defence in depth).
    """
    unknown = sorted(set(rows.columns) - set(FEATURES))
    missing = sorted(set(REQUIRED) - set(rows.columns))
    errors = []
    if unknown:  # a typo like "MonthlyIncom" would otherwise score as "income not reported"
        errors.append(f"unknown fields: {unknown}")
    if missing:
        errors.append(f"missing required fields: {missing}")
    if errors:
        raise InvalidApplicant(errors)

    try:
        # Absent optional field -> NaN. Positional index: "row 3" means the 4th row sent.
        x = rows.reindex(columns=FEATURES).astype(float).reset_index(drop=True)
    except (TypeError, ValueError) as exc:
        raise InvalidApplicant([f"every field must be a number ({exc})"]) from None

    def flag(mask: pd.Series, col: str, problem: str) -> None:
        # The value itself is not repeated: error text ends up in responses and
        # logs, and applicant data must not (D-024).
        errors.extend(f"row {i}: {col} {problem}" for i in x.index[mask])

    for col in FEATURES:
        v = x[col]
        if col in REQUIRED:
            flag(v.isna(), col, "is required")
        flag(np.isinf(v), col, "must be finite")
        flag(v < 0, col, "must not be negative")
        if col in COUNTS:
            flag(np.isfinite(v) & (v % 1 != 0), col, "must be a whole number")
    # D-008: training dropped these rows; live, they are refused, not scored.
    flag(x[AGE] < MIN_AGE, AGE, f"must be at least {MIN_AGE}")

    if errors:
        raise InvalidApplicant(errors)
    return x


# --------------------------------------------------------------------------- #
# Artifact integrity
# --------------------------------------------------------------------------- #
def verify_artifacts(paths: Mapping[str, Path], lock: Path) -> None:
    """Refuse to serve files that differ from the FIRST (official) test-lock run."""
    if not lock.exists():
        raise ArtifactMismatch(f"No test lock at {lock}: this model was never evaluated on test")
    tested = json.loads(lock.read_text(encoding="utf-8"))["runs"][0]["fingerprints"]
    bad = [name for name, path in paths.items() if sha256(path) != tested.get(name)]
    if bad:
        raise ArtifactMismatch(
            f"{bad} differ from the tested version in {lock.name}. "
            "Re-run the evaluation before serving a changed model."
        )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #
class ScoringService:
    """Loaded once at start-up; scoring itself is stateless and thread-safe."""

    def __init__(
        self,
        scorecard: Scorecard,
        reference: dict[str, float],
        cutoff: int,
        max_reasons: int,
        min_points_lost: float,
        model_version: str,
    ):
        if not hasattr(scorecard, "score_to_pd_"):
            raise ArtifactMismatch("Scorecard has no master scale: PDs would be uncalibrated")
        if set(reference) != set(scorecard.features):
            raise ArtifactMismatch("Reason reference and scorecard cover different features")
        self.scorecard = scorecard
        self.reference = reference
        self.cutoff = cutoff
        self.max_reasons = max_reasons
        self.min_points_lost = min_points_lost
        self.model_version = model_version
        self._texts = code_texts()

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, verify: bool = True) -> ScoringService:
        cfg = cfg or load_config()
        paths = {name: resolve_path(cfg["artifacts"][name]) for name in SERVED_ARTIFACTS}
        if verify:
            verify_artifacts(paths, resolve_path(cfg["evaluation"]["lock"]))
        sc = Scorecard.from_json(paths["scorecard"])
        return cls(
            scorecard=sc,
            reference=load_reference(paths["reason_reference"]),
            # Recomputed from config, not stored: the cutoff is a policy on top of
            # the model (D-018), and the response always says which one was used.
            cutoff=policy_cutoff(sc, economics_from_config(cfg)),
            max_reasons=cfg["reasons"]["max_reasons"],
            min_points_lost=cfg["reasons"]["min_points_lost"],
            model_version=sha256(paths["scorecard"])[:12],
        )

    def score(self, applicant: Applicant) -> Decision:
        """One applicant (field name -> value; None = not reported)."""
        return self.score_batch(pd.DataFrame([dict(applicant)]))[0]

    def score_batch(self, rows: pd.DataFrame) -> list[Decision]:
        """Many applicants in one vectorised pass. All-or-nothing: one invalid
        row rejects the batch, so a caller never gets a partial answer by mistake."""
        x = clean(validate(rows))
        scores = self.scorecard.score(x)
        pds = self.scorecard.score_pd(x)
        reasons = top_reasons(
            self.scorecard, x, self.reference, self.max_reasons, self.min_points_lost
        )
        out = []
        for i, (s, p) in enumerate(zip(scores, pds, strict=True)):
            approve = bool(s >= self.cutoff)  # same rule as D-018: PD <= break-even
            items = () if approve else self._reasons(reasons.iloc[i])
            out.append(
                Decision(
                    score=int(s),
                    pd=float(p),
                    decision="approve" if approve else "decline",
                    cutoff=self.cutoff,
                    reasons=items,
                    model_version=self.model_version,
                )
            )
        return out

    def breakdown(self, applicant: Applicant) -> pd.DataFrame:
        """Where one applicant's score comes from: per scored feature, the cleaned
        value, the bin it fell in, the points received, and the points of the
        train-average applicant (the reason-code reference, D-019).

        The points column always sums to `score(applicant).score`: the scorecard
        is additive, so this is the exact explanation, not an approximation.
        """
        x = clean(validate(pd.DataFrame([dict(applicant)])))
        points, _ = applicant_points(self.scorecard, x)
        rows = []
        for f in self.scorecard.features:
            edges = self.scorecard.encoder_.features_[f].edges
            value = x[f].iloc[0]
            got = int(points[f].iloc[0])
            rows.append(
                {
                    "feature": f,
                    "value": float(value),  # NaN = missing after clean()
                    "bin": bin_labels(edges)[int(assign_bins(x[f].to_numpy(float), edges)[0])],
                    "points": got,
                    "typical_points": round(self.reference[f], 1),
                    "vs_typical": round(got - self.reference[f], 1),
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def _reasons(self, row: pd.Series) -> tuple[ReasonItem, ...]:
        items = []
        for k in range(1, self.max_reasons + 1):
            code = row[f"reason_{k}"]
            if code:
                items.append(ReasonItem(code, self._texts[code], float(row[f"points_lost_{k}"])))
        return tuple(items)
