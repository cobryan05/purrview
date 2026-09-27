"""add event_granularity to interaction_rules

Revision ID: 8aafadbc3f5e
Revises: c6471a5f1a94
Create Date: 2026-09-26 20:40:23.500905

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8aafadbc3f5e"
down_revision: Union[str, None] = "c6471a5f1a94"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: Rules whose subject slot names at most one label and does not widen to descendants ran
#: as exactly one state machine before this column existed, so `rule` reproduces their
#: behaviour byte for byte. Every other rule ran as one machine *per label*, which is
#: neither setting — `subject` is the closest, and it is the one that fixes them: those are
#: precisely the rules that split one subject into several events when the detector changed
#: its mind, or when it moved from one target to the next.
#:
#: `json_valid` guards a column that is free text as far as SQLite is concerned;
#: `json_array_length` raises rather than returning NULL on a malformed value, which would
#: abort the whole migration over one bad row.
_BACKFILL = sa.text("""
    UPDATE interaction_rules
       SET event_granularity = CASE
               WHEN COALESCE(subject_a_include_descendants, 0) = 0
                AND json_valid(COALESCE(subject_a_refs_json, '[]'))
                AND json_array_length(COALESCE(subject_a_refs_json, '[]')) <= 1
               THEN 'rule'
               ELSE 'subject'
           END
    """)


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("event_granularity", sa.String(), server_default=sa.text("'subject'"), nullable=False)
        )
    op.get_bind().execute(_BACKFILL)


def downgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("event_granularity")
