"""Create the decisions audit table (D-025).

Revision ID: 0001
Revises:
Create Date: 2026-11-21
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# op.f(name) marks a name as final, so the naming convention in models.py is not
# applied a second time (without it: "ck_decisions_ck_decisions_decision").


def upgrade() -> None:
    op.create_table(
        "decisions",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(12), nullable=False),
        sa.Column("cutoff", sa.Integer(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("pd", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(7), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decisions")),
        sa.CheckConstraint(
            "decision IN ('approve', 'decline')", name=op.f("ck_decisions_decision")
        ),
        sa.CheckConstraint("pd >= 0 AND pd <= 1", name=op.f("ck_decisions_pd")),
    )
    op.create_index(op.f("ix_decisions_created_at"), "decisions", ["created_at"])
    op.create_index(op.f("ix_decisions_model_version"), "decisions", ["model_version"])


def downgrade() -> None:
    op.drop_index(op.f("ix_decisions_model_version"), table_name="decisions")
    op.drop_index(op.f("ix_decisions_created_at"), table_name="decisions")
    op.drop_table("decisions")
