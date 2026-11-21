"""POST /score: one applicant in, one recorded decision out (D-024, D-025)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from scorecard.api.schemas import ApplicantIn, DecisionOut

router = APIRouter(tags=["scoring"])
log = logging.getLogger("scorecard.api")


# A plain `def`, not `async def`: scoring is CPU work (pandas/numpy). FastAPI
# runs sync endpoints in a thread pool, so one request never blocks the event loop.
@router.post("/score", response_model=DecisionOut)
def score(applicant: ApplicantIn, request: Request) -> DecisionOut:
    raw = applicant.to_raw()
    decision = request.app.state.service.score(raw)
    try:
        decision_id = request.app.state.repository.record(raw, decision)
    except SQLAlchemyError as exc:
        # Fail closed (D-025): a decision that cannot be recorded is not issued.
        # Log the error type only; the message could carry applicant data.
        log.error("decision not recorded, none issued: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="The decision could not be recorded, so none was issued. Retry later.",
            headers={"Retry-After": "5"},
        ) from None
    return DecisionOut(decision_id=decision_id, **decision.to_dict())
