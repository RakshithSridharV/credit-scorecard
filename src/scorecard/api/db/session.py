"""Database connection and the start-up schema check (D-025).

The URL comes from configs/config.yaml, overridden by the SCORECARD_DB_URL
environment variable. Config holds the local default; the environment holds
deployment values (Postgres host, password), which never belong in git.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from scorecard.config import PROJECT_ROOT, load_config, resolve_path

ENV_VAR = "SCORECARD_DB_URL"
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


class SchemaNotCurrent(RuntimeError):
    """The database is not at the schema version this code expects."""


def database_url(cfg: dict[str, Any] | None = None) -> str:
    """Environment first, then config. Relative SQLite paths are made absolute
    against the repo root, so the file does not depend on the working directory."""
    raw = os.environ.get(ENV_VAR) or (cfg or load_config())["database"]["url"]
    url = make_url(raw)
    if url.drivername.startswith("sqlite") and url.database not in (None, "", ":memory:"):
        url = url.set(database=str(resolve_path(url.database)))
    return url.render_as_string(hide_password=False)


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    kwargs: dict[str, Any] = {
        # Error messages and logs must not contain the SQL parameters: they are
        # the applicant's financial data.
        "hide_parameters": True,
        "pool_pre_ping": True,  # a dropped connection is replaced, not handed to a request
    }
    if parsed.drivername.startswith("sqlite"):
        # FastAPI runs sync endpoints in a thread pool, so a connection may be
        # used by a thread other than the one that opened it.
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database not in (None, "", ":memory:"):
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


def alembic_config(url: str | None = None) -> AlembicConfig:
    """Alembic settings in code, so the app, the tests and the CLI agree."""
    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url:
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # ini escaping
    return cfg


def assert_migrated(engine: Engine) -> None:
    """Fail fast (D-024) if the database is unreachable or not at the latest migration.

    The app never creates or alters tables itself: schema changes are a deliberate
    deploy step (`make migrate`), reviewed like code.
    """
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    with engine.connect() as conn:  # an unreachable database raises here
        current = MigrationContext.configure(conn).get_current_revision()
    if current != head:
        raise SchemaNotCurrent(
            f"Database schema is at {current!r}, the code expects {head!r}. Run `make migrate`."
        )
