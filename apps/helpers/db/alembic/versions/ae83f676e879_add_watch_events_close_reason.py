"""add watch_events close_reason

Why an event ended, split out of the `truncated` boolean.

`truncated` answered "is this span the whole story?" and was then read as "the runaway
cap fired" by the log line, the event card and the detail modal. Two different endings
set it: the cap, and a force-close when the watcher stops, pauses or restarts. Saving a
watcher's config restarts its task, so an ordinary settings edit closed every open event
and every one of them was reported as capped — including events on rules with
`max_duration_s = 0`, where the cap is disabled and the message is impossible.

Nullable with no back-fill for existing rows: which of the two it was is genuinely not
recorded anywhere, and the UI's whole job here is to stop guessing. The one exception is
back-filled below — events closed by `close_stranded_watch_events` already say so in
their metadata, so those are known to be `forced`.

Revision ID: ae83f676e879
Revises: 7799a2dd977e
Create Date: 2026-08-22 01:34:35.770991

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ae83f676e879"
down_revision: Union[str, None] = "7799a2dd977e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("close_reason", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    # The only ending old rows do record. `close_stranded_watch_events` has always
    # stamped this flag, so these are known to be forced rather than guessed at.
    op.execute(
        sa.text(
            "UPDATE watch_events SET close_reason = 'forced' "
            "WHERE json_extract(metadata_json, '$.closedBy') = 'restart'"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_column("close_reason")
