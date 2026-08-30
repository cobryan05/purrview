"""add watch_events started_at and identity indexes

Revision ID: 7efcbb5f6ba0
Revises: fd74a3b899ec
Create Date: 2026-08-29 23:23:42.886657

`started_at` is the column every reader of `watch_events` sorts on and it carried no
index of its own; the composite one matches `DbClient._IDENTITY_COLUMNS` exactly, in
that order, with `started_at` last, so the grouped view's `ROW_NUMBER()` and the
session window function can walk each partition already ordered.

Autogenerate also proposed a `bboxes.confidence` REAL→Float alter. That was dropped by
hand: the two are the same type in SQLite, the change is drift autogenerate reports on
every run, and applying it would batch-rebuild the whole `bboxes` table — including its
computed column — to accomplish nothing.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7efcbb5f6ba0"
down_revision: Union[str, None] = "fd74a3b899ec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.create_index(
            "ix_watch_events_identity_started_at",
            ["rule_uuid", "subject_ref", "target_ref", "source_uuid", "started_at"],
            unique=False,
        )
        batch_op.create_index("ix_watch_events_started_at", ["started_at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_index("ix_watch_events_started_at")
        batch_op.drop_index("ix_watch_events_identity_started_at")
