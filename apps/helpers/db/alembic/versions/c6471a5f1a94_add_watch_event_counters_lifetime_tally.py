"""add watch_event_counters lifetime tally

Revision ID: c6471a5f1a94
Revises: aaebbb686af7
Create Date: 2026-09-25 19:28:43.374002

Hand-adjusted after autogenerate, in three places, the first two for the same reasons
spelled out in `9b19f7c2223f`:

* `last_seen` is written as `sa.DateTime(timezone=True)` rather than the `UtcDateTime`
  decorator autogenerate names but does not import.
* A spurious `bboxes.confidence` REAL -> Float retype was dropped — a no-op reflection
  artifact on a computed column, unrelated to this change.
* A backfill was added. Without it every counter starts at zero on deploy while Home
  Assistant holds a retained non-zero value, and the drop is read as a counter reset —
  the exact corruption this table exists to stop. Seeding from the surviving events
  reproduces the number HA already has, so the cutover is invisible to it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c6471a5f1a94"
down_revision: Union[str, None] = "aaebbb686af7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "watch_event_counters",
        sa.Column("source_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("rule_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("subject_ref", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("source_uuid", "rule_uuid", "subject_ref"),
    )
    # Matches the `COUNT(*) GROUP BY` this table replaces, including its filter on the
    # two nullable uuids — a tally keyed on a NULL has no entity to publish to. The
    # `COALESCE` is the `""`-for-NULL sentinel `WatchEventCounter` documents.
    op.execute("""
        INSERT INTO watch_event_counters (source_uuid, rule_uuid, subject_ref, count, last_seen)
        SELECT source_uuid, rule_uuid, COALESCE(subject_ref, ''), COUNT(*), MAX(started_at)
        FROM watch_events
        WHERE source_uuid IS NOT NULL AND rule_uuid IS NOT NULL
        GROUP BY source_uuid, rule_uuid, COALESCE(subject_ref, '')
        """)


def downgrade() -> None:
    op.drop_table("watch_event_counters")
