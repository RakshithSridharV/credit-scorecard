"""Write and read audit records (D-025). The only module that touches the table."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone

from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import sessionmaker

from scorecard.api.db.models import DecisionRecord
from scorecard.api.scoring import Decision


class DecisionRepository:
    """Append-only on purpose: there is no update or delete method to call."""

    def __init__(self, engine: Engine):
        self.engine = engine
        # expire_on_commit=False: a returned record stays readable after its session closes.
        self._sessions = sessionmaker(engine, expire_on_commit=False)

    def record(self, inputs: Mapping[str, float | int | None], decision: Decision) -> str:
        """Store one decision; return its id. Commits before returning, so a caller
        that receives an id knows the record is durable."""
        row = DecisionRecord(
            id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            model_version=decision.model_version,
            cutoff=decision.cutoff,
            score=decision.score,
            pd=decision.pd,
            decision=decision.decision,
            reasons=[r.__dict__ for r in decision.reasons],
            inputs=dict(inputs),
        )
        with self._sessions.begin() as session:  # commit on success, rollback on error
            session.add(row)
        return row.id

    def get(self, decision_id: str) -> DecisionRecord | None:
        with self._sessions() as session:
            return session.get(DecisionRecord, decision_id)

    def count(self) -> int:
        with self._sessions() as session:
            return session.scalar(select(func.count()).select_from(DecisionRecord)) or 0

    def ping(self) -> None:
        """Raise if the database cannot be reached (used by /ready)."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def close(self) -> None:
        self.engine.dispose()
