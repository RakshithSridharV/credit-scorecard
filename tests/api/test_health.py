"""Probes (/health, /ready) and start-up behaviour (fail fast)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from scorecard.api.db.session import SchemaNotCurrent
from scorecard.api.main import create_app
from scorecard.api.scoring import ArtifactMismatch


def test_health_reports_the_live_model_and_cutoff(client, scoring_service):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_version": scoring_service.model_version,
        "cutoff": scoring_service.cutoff,
    }


def test_an_untested_model_stops_the_server_at_startup(repository):
    def tampered():
        raise ArtifactMismatch("['scorecard'] differ from the tested version")

    with pytest.raises(ArtifactMismatch), TestClient(create_app(tampered, lambda: repository)):
        pass  # never reached: start-up itself must fail


def test_an_unmigrated_database_stops_the_server_at_startup(scoring_service):
    def unmigrated():
        raise SchemaNotCurrent("Database schema is at None, the code expects '0001'")

    app = create_app(lambda: scoring_service, unmigrated)
    with pytest.raises(SchemaNotCurrent), TestClient(app):
        pass


def test_the_model_is_loaded_once_not_per_request(scoring_service, repository):
    calls = []

    def factory():
        calls.append(1)
        return scoring_service

    with TestClient(create_app(factory, lambda: repository)) as c:
        for _ in range(3):
            c.get("/health")
    assert len(calls) == 1


def test_ready_when_the_database_answers(client):
    assert client.get("/ready").json() == {"status": "ready"}


def test_not_ready_but_still_alive_when_the_database_is_down(client, repository, monkeypatch):
    def down():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(repository, "ping", down)
    assert client.get("/ready").status_code == 503
    assert client.get("/health").status_code == 200  # a DB outage is no reason to restart


def test_the_api_does_not_import_training_only_libraries():
    """LightGBM needs libgomp, which the slim image lacks; matplotlib is dead weight.
    Run in a fresh interpreter so other tests' imports do not hide a regression."""
    import subprocess
    import sys

    code = (
        "import sys, scorecard.api.main; "
        "print([m for m in ('lightgbm', 'matplotlib') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
