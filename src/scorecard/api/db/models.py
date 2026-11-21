"""The audit table: one row per issued credit decision (D-025)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Named constraints and indexes, so a later migration can refer to (and drop)
# them by a predictable name on every database.
NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class DecisionRecord(Base):
    """An issued decision. Append-only: no code path updates or deletes a row.

    `inputs` keeps the 10 raw bureau fields as received, so any decision can be
    re-scored later with the model named by `model_version` (dispute handling,
    model validation). Together with `score` it also feeds monitoring (D-021).
    """

    __tablename__ = "decisions"
    __table_args__ = (
        CheckConstraint("decision IN ('approve', 'decline')", name="decision"),
        CheckConstraint("pd >= 0 AND pd <= 1", name="pd"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID4 as text: portable
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # UTC
    model_version: Mapped[str] = mapped_column(String(12), index=True)
    cutoff: Mapped[int]
    score: Mapped[int]
    pd: Mapped[float]
    decision: Mapped[str] = mapped_column(String(7))
    reasons: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON)
