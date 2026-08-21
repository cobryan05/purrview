"""add event_blips and watch_events.review_state

Revision ID: 9b19f7c2223f
Revises: e5c152967409
Create Date: 2026-08-20 16:21:25.108345

Hand-adjusted after autogenerate, in three places:

* `created_at` / `updated_at` are written as `sa.DateTime(timezone=True)`, matching every
  other migration here. Autogenerate emitted the `UtcDateTime` decorator class by name
  without importing it, which would raise at migration time rather than at import.
* `review_state` gets a `server_default`. Autogenerate emitted it `nullable=False` with
  no default, which fails against any database that already holds events.
* A spurious `bboxes.confidence` REAL -> Float retype was dropped. It is a no-op
  reflection artifact, unrelated to this change, and it sits on a *computed* column —
  batch_alter_table rebuilds the table to apply it, which is real risk for no benefit.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b19f7c2223f"
down_revision: Union[str, None] = "e5c152967409"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "event_blips",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("recording_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("start_s", sa.Float(), nullable=False),
        sa.Column("end_s", sa.Float(), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("fault", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("harvestable", sa.Boolean(), nullable=False),
        sa.Column("state", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("detail_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["event_uuid"], ["watch_events.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("event_blips", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_event_blips_event_uuid"), ["event_uuid"], unique=False)
        batch_op.create_index(batch_op.f("ix_event_blips_harvestable"), ["harvestable"], unique=False)
        batch_op.create_index(batch_op.f("ix_event_blips_kind"), ["kind"], unique=False)
        batch_op.create_index(batch_op.f("ix_event_blips_recording_uuid"), ["recording_uuid"], unique=False)
        batch_op.create_index(batch_op.f("ix_event_blips_severity"), ["severity"], unique=False)
        batch_op.create_index(batch_op.f("ix_event_blips_state"), ["state"], unique=False)

    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "review_state",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="unreviewed",
            )
        )
        batch_op.create_index(batch_op.f("ix_watch_events_review_state"), ["review_state"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_watch_events_review_state"))
        batch_op.drop_column("review_state")

    with op.batch_alter_table("event_blips", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_event_blips_state"))
        batch_op.drop_index(batch_op.f("ix_event_blips_severity"))
        batch_op.drop_index(batch_op.f("ix_event_blips_recording_uuid"))
        batch_op.drop_index(batch_op.f("ix_event_blips_kind"))
        batch_op.drop_index(batch_op.f("ix_event_blips_harvestable"))
        batch_op.drop_index(batch_op.f("ix_event_blips_event_uuid"))

    op.drop_table("event_blips")
