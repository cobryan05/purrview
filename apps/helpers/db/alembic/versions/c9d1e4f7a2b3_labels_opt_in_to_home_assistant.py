"""labels opt in to Home Assistant rather than out

Revision ID: c9d1e4f7a2b3
Revises: 918cd4b9a74a
Create Date: 2026-08-31 00:00:00.000000

`labels.publish_to_ha` flips from default-on to default-off, and every existing row is
set to 0.

The flag used to be a rarely-reached escape hatch: a label only got a device if some
enabled rule named it as a subject, so the vocabulary at large was already excluded and
defaulting the column on cost nothing. That gate has been removed — a label device is
now justified by the label, because `Detected now` reads off the tracker and answers
"is she in the room" with no rule involved. With the gate gone the column *is* the
bound, and a default of 1 turned a 43-label vocabulary into 43 Home Assistant devices,
most of them classes nobody automates on. Home Assistant charges per device in its UI,
so that lands where it is most visible.

Existing rows are set to 0 rather than left alone, which is the deliberate part. Leaving
them would mean the very installs that prompted this keep the wall of devices, and the
new default would only ever apply to labels created afterwards — the setting would not
mean the same thing for two labels in the same tree.

**This retracts the subject devices an install currently has**, including for labels a
rule does name. That is a retraction rather than a silent stop, so Home Assistant removes
the entities rather than leaving them stale, and re-ticking a label in Labels recreates
them with the same `unique_id` — the entity ids are derived from names that do not change
here, so history is reclaimed rather than orphaned. Re-tick the few you automate on.

`server_default` is changed as well as the data: SQLite has no ALTER COLUMN, so the
column is rewritten through batch mode. Without the matching default a row inserted by
the freshly-created schema and one inserted by the migrated schema would disagree, which
is the divergence the original migration's docstring was written to prevent.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d1e4f7a2b3"
down_revision: Union[str, None] = "918cd4b9a74a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.alter_column(
            "publish_to_ha",
            existing_type=sa.Boolean(),
            existing_nullable=False,
            server_default=sa.text("0"),
        )
    # After the default, so a row inserted between the two (there is no such row inside a
    # migration, but the ordering is the one that reads correctly) cannot arrive ticked.
    op.execute(sa.text("UPDATE labels SET publish_to_ha = 0"))


def downgrade() -> None:
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.alter_column(
            "publish_to_ha",
            existing_type=sa.Boolean(),
            existing_nullable=False,
            server_default=sa.text("1"),
        )
    # Restores the pre-upgrade *default*, not the pre-upgrade per-label choices — those
    # were overwritten and this migration did not record them. Turning every label back on
    # is the honest inverse of turning every label off; it is not a claim to have
    # remembered which ones you had.
    op.execute(sa.text("UPDATE labels SET publish_to_ha = 1"))
