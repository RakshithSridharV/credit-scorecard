"""Audit database: repository round trip, migrations == ORM, schema check, URL handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from scorecard.api.db.models import Base
from scorecard.api.db.repository import DecisionRepository
from scorecard.api.db.session import (
    ENV_VAR,
    SchemaNotCurrent,
    alembic_config,
    assert_migrated,
    database_url,
    make_engine,
)
from scorecard.api.scoring import Decision, ReasonItem
from scorecard.config import PROJECT_ROOT

UTC = timezone.utc
DECLINE = Decision(
    score=512,
    pd=0.21,
    decision="decline",
    cutoff=549,
    reasons=(ReasonItem("DLQ_90", "Serious delinquency", 38.7),),
    model_version="27f4de0069b8",
)
INPUTS = {"age": 30, "MonthlyIncome": None, "DebtRatio": 0.4}


@pytest.fixture
def url(tmp_path):
    return f"sqlite:///{tmp_path / 'audit.db'}"


@pytest.fixture
def migrated(url):
    command.upgrade(alembic_config(url), "head")
    engine = make_engine(url)
    yield engine
    engine.dispose()


def test_record_and_read_back(migrated):
    repo = DecisionRepository(migrated)
    before = datetime.now(UTC)
    row = repo.get(repo.record(INPUTS, DECLINE))
    assert (row.score, row.pd, row.decision, row.cutoff, row.model_version) == (
        512,
        0.21,
        "decline",
        549,
        "27f4de0069b8",
    )
    assert row.reasons == [{"code": "DLQ_90", "text": "Serious delinquency", "points_lost": 38.7}]
    assert row.inputs == INPUTS
    created = row.created_at.replace(tzinfo=UTC)  # SQLite drops the tz; the value is UTC
    assert before - timedelta(seconds=1) <= created <= datetime.now(UTC)
    assert repo.count() == 1


def test_repository_has_no_update_or_delete():
    names = [m for m in dir(DecisionRepository) if not m.startswith("_")]
    assert not [m for m in names if m.startswith(("update", "delete", "remove"))]


def test_database_rejects_an_impossible_decision(migrated):
    sql = (
        "INSERT INTO decisions VALUES ('x', '2026-01-01', 'v', 549, 600, 0.1, 'maybe', '[]', '{}')"
    )
    with pytest.raises(IntegrityError), migrated.begin() as conn:
        conn.execute(text(sql))


def test_database_errors_never_contain_the_sql_parameters(migrated):
    with pytest.raises(IntegrityError) as exc, migrated.begin() as conn:
        conn.execute(text("INSERT INTO decisions (id) VALUES (:id)"), {"id": "secret-987654"})
    assert "secret-987654" not in str(exc.value)  # hide_parameters=True


# ------------------------------------------------------------------ migrations
def test_migrations_build_exactly_the_orm_schema(migrated):
    with migrated.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_migration_and_orm_give_the_same_constraint_names(migrated, tmp_path):
    """compare_metadata ignores CHECK constraints, so compare their names directly.
    (Caught a real bug: the convention was applied twice -> "ck_decisions_ck_decisions_pd".)"""
    orm = make_engine(f"sqlite:///{tmp_path / 'orm.db'}")
    Base.metadata.create_all(orm)

    def names(engine):
        insp = inspect(engine)
        checks = {c["name"] for c in insp.get_check_constraints("decisions")}
        return checks, {i["name"] for i in insp.get_indexes("decisions")}

    assert names(migrated) == names(orm)
    assert names(migrated)[0] == {"ck_decisions_decision", "ck_decisions_pd"}
    orm.dispose()


def test_downgrade_removes_the_table_and_upgrade_restores_it(url, migrated):
    command.downgrade(alembic_config(url), "base")
    assert "decisions" not in inspect(migrated).get_table_names()
    command.upgrade(alembic_config(url), "head")
    assert "decisions" in inspect(migrated).get_table_names()


def test_schema_check_fails_on_an_empty_database_and_passes_after_migrate(url):
    engine = make_engine(url)
    with pytest.raises(SchemaNotCurrent, match="make migrate"):
        assert_migrated(engine)
    command.upgrade(alembic_config(url), "head")
    assert_migrated(engine)
    engine.dispose()


# ------------------------------------------------------------------ URL
def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "postgresql+psycopg://u:p@db:5432/scorecard")
    assert database_url({"database": {"url": "sqlite:///x.db"}}).startswith("postgresql")


def test_relative_sqlite_path_is_anchored_at_the_repo_root(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    url = database_url({"database": {"url": "sqlite:///data/audit/decisions.db"}})
    assert url == f"sqlite:///{PROJECT_ROOT / 'data' / 'audit' / 'decisions.db'}"
