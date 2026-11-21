"""Probes (D-024, D-025).

GET /health  liveness: the process is up and the verified model is loaded
             (fail-fast start-up guarantees the second part). Restart if this fails.
GET /ready   readiness: the audit database answers, so decisions can be issued.
             Stop routing traffic here if this fails, but do NOT restart: a database
             outage is not fixed by restarting the API, and restarting every pod at
             once turns it into a full outage.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from scorecard.api.schemas import HealthOut, ReadyOut

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    service = request.app.state.service
    return HealthOut(status="ok", model_version=service.model_version, cutoff=service.cutoff)


@router.get("/ready", response_model=ReadyOut)
def ready(request: Request) -> ReadyOut:
    try:
        request.app.state.repository.ping()
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="audit database unavailable") from None
    return ReadyOut(status="ready")
