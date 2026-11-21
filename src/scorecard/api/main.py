"""FastAPI application (D-024, D-025).

    make migrate && make serve     # = alembic upgrade head; uvicorn scorecard.api.main:app

The model and the audit database are set up ONCE, in the lifespan hook, before
the server accepts any request. If either fails (ArtifactMismatch: the model is
not the tested one; SchemaNotCurrent or a connection error: the database is not
usable) the exception propagates and the process exits: fail fast.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from scorecard.api.db.repository import DecisionRepository
from scorecard.api.db.session import assert_migrated, database_url, make_engine
from scorecard.api.routes import health, score
from scorecard.api.scoring import InvalidApplicant, ScoringService

log = logging.getLogger("scorecard.api")


def default_repository() -> DecisionRepository:
    engine = make_engine(database_url())
    assert_migrated(engine)  # the app never creates tables; `make migrate` does
    return DecisionRepository(engine)


def create_app(
    service_factory: Callable[[], ScoringService] = ScoringService.from_config,
    repository_factory: Callable[[], DecisionRepository] = default_repository,
) -> FastAPI:
    """App factory. Tests pass a synthetic service and a throwaway SQLite database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = service_factory()  # raises -> the server never starts
        repository = repository_factory()  # same
        app.state.service, app.state.repository = service, repository
        log.info("model %s loaded, cutoff %d", service.model_version, service.cutoff)
        yield
        repository.close()  # release pooled connections on shutdown

    app = FastAPI(
        title="Credit scorecard API",
        version="1.0.0",
        summary="Give Me Some Credit scorecard: a US-data method demo, not a lending model.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(score.router)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes each bad value back ("input"). We drop it:
        # (1) applicant data must not be copied into responses and logs, and
        # (2) an echoed Infinity cannot be JSON-encoded, which turned a 422 into a 500.
        detail = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(InvalidApplicant)
    async def invalid_applicant(_: Request, exc: InvalidApplicant) -> JSONResponse:
        # The schema catches almost everything first; this is the core's second
        # line of defence, reported in the same 422 shape a client already handles.
        detail = [{"loc": ["body"], "msg": m, "type": "invalid_applicant"} for m in exc.errors]
        return JSONResponse(status_code=422, content={"detail": detail})

    return app


app = create_app()  # importing this module does not load the model; startup does
