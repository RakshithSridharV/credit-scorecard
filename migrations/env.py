"""Alembic environment (D-025): one URL source shared with the API."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from scorecard.api.db.models import Base
from scorecard.api.db.session import database_url, make_engine

config = context.config
if config.config_file_name and config.attributes.get("configure_logging", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Tests pass an explicit URL; otherwise use the same source as the API.
URL = config.get_main_option("sqlalchemy.url") or database_url()
target_metadata = Base.metadata


def run_offline() -> None:
    """Emit SQL to stdout (for a DBA to review) instead of running it."""
    context.configure(url=URL, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = make_engine(URL)
    with engine.connect() as conn:
        # render_as_batch: SQLite cannot ALTER most things in place; batch mode
        # rebuilds the table instead. Harmless on Postgres.
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
