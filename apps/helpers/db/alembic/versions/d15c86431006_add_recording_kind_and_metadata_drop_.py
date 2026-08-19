"""add recording kind and metadata, drop watch_events.recording_uuid

Revision ID: d15c86431006
Revises: f8f31a7ddc86
Create Date: 2026-08-18 22:59:36.671177

An event can now own more than one piece of media — a clip and a timelapse — so the
single `watch_events.recording_uuid` pointer that answered "does this card have a video"
is replaced by the `recordings.event_uuid` link that already existed in the other
direction. No data migration is needed for that: the recorder has always written both
ends, so every row the dropped column named is already reachable from `event_uuid`.

Autogenerate also proposed a `bboxes.confidence` REAL→Float alteration. It is dropped
here: the two are the same SQLite type, and honouring it would rebuild a large table
carrying a computed column for no change at all.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d15c86431006"
down_revision: Union[str, None] = "f8f31a7ddc86"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("recordings", schema=None) as batch_op:
        # server_default, not just a model default: the column is NOT NULL and the table
        # has rows. Every one of them predates timelapses and is therefore a clip.
        batch_op.add_column(
            sa.Column("kind", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="clip")
        )
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f("ix_recordings_kind"), ["kind"], unique=False)

    # Dropping the column takes its foreign key with it — batch mode rebuilds the table
    # from the remaining columns, so naming the constraint (which autogenerate could
    # not, and which SQLite does not record) is unnecessary.
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_column("recording_uuid")


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("recording_uuid", sa.VARCHAR(), nullable=True))
        batch_op.create_foreign_key(
            "fk_watch_events_recording_uuid", "recordings", ["recording_uuid"], ["uuid"], ondelete="SET NULL"
        )
    # Deliberately not repopulated from `recordings.event_uuid`. One column cannot hold
    # both of an event's recordings, and inventing a rule for which one wins would make
    # a downgrade lose data silently rather than visibly.

    with op.batch_alter_table("recordings", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_recordings_kind"))
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("kind")
