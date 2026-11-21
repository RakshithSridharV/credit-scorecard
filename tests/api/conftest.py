"""API test client: synthetic scoring service + a throwaway SQLite audit database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scorecard.api.db.models import Base
from scorecard.api.db.repository import DecisionRepository
from scorecard.api.db.session import make_engine
from scorecard.api.main import create_app


@pytest.fixture
def repository(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    Base.metadata.create_all(engine)  # tests/unit/test_db.py proves migrations build the same
    repo = DecisionRepository(engine)
    yield repo
    repo.close()


@pytest.fixture
def client(scoring_service, repository):
    # `with` runs the lifespan hook, exactly like a real server start-up.
    app = create_app(lambda: scoring_service, lambda: repository)
    with TestClient(app) as c:
        yield c
