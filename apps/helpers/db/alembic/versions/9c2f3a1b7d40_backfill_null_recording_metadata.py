"""backfill null recording metadata_json

Revision ID: 9c2f3a1b7d40
Revises: d15c86431006
Create Date: 2026-08-19 09:05:00.000000

`d15c86431006` added `recordings.metadata_json` as a nullable column and did not
backfill it, so every recording written before that migration holds SQL NULL rather
than `{}`. `RecordingRead` typed the field as a plain dict, so reading one raised, the
recording endpoint answered `recording: null`, and the events modal — which cannot tell
that apart from a deleted file — reported an install's whole clip history as "no longer
on disk" while every mp4 was still on the volume.

The read model now coerces NULL to `{}` as well, so this is belt and braces rather than
the only line of defence. Both are wanted: the model keeps any row that reaches it
readable whatever its provenance, and this leaves the table in the state the column's
default always claimed it was in, so a query written against it does not have to know.

Data only — the column's type and nullability are unchanged, so there is nothing to
undo and `downgrade` is deliberately empty. Restoring the NULLs would only reinstate
the fault.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c2f3a1b7d40"
down_revision: Union[str, None] = "d15c86431006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE recordings SET metadata_json = '{}' WHERE metadata_json IS NULL")


def downgrade() -> None:
    # Nothing to reverse: this migration changed no schema, and putting the NULLs back
    # would only restore the bug it exists to clear.
    pass
