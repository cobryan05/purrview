"""add review fields to watch_events

Revision ID: 387e8e43b380
Revises: 737629e0e63c
Create Date: 2026-08-18 09:06:34.991004

Adds the columns the NVR event review page filters and renders on: the denormalised
rule name, the matched subject/target refs and their display names, the runaway-cap
flag, and the path to the event's own card art.

Autogenerate also proposed a `bboxes.confidence` REAL→Float retype. That is pre-existing
drift between SQLite's reflected type and SQLAlchemy's, unrelated to this change, and a
no-op rewrite of a large table with a computed column — removed deliberately.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "387e8e43b380"
down_revision: Union[str, None] = "737629e0e63c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rule_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column("event_label", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column("subject_ref", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column("subject_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column("target_ref", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.add_column(sa.Column("target_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        # server_default, or adding a NOT NULL column fails on any table that already
        # has rows. Autogenerate omits it; it is not optional here.
        batch_op.add_column(sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("snapshot_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.create_index(batch_op.f("ix_watch_events_rule_uuid"), ["rule_uuid"], unique=False)
        batch_op.create_index(batch_op.f("ix_watch_events_subject_ref"), ["subject_ref"], unique=False)
        batch_op.create_index(batch_op.f("ix_watch_events_target_ref"), ["target_ref"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_watch_events_target_ref"))
        batch_op.drop_index(batch_op.f("ix_watch_events_subject_ref"))
        batch_op.drop_index(batch_op.f("ix_watch_events_rule_uuid"))
        batch_op.drop_column("snapshot_path")
        batch_op.drop_column("truncated")
        batch_op.drop_column("target_name")
        batch_op.drop_column("target_ref")
        batch_op.drop_column("subject_name")
        batch_op.drop_column("subject_ref")
        batch_op.drop_column("event_label")
        batch_op.drop_column("rule_name")
