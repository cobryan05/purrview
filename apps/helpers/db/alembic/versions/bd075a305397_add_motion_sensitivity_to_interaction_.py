"""add motion_sensitivity to interaction rules

Revision ID: bd075a305397
Revises: ae83f676e879
Create Date: 2026-08-22 10:39:02.860828

The `motion` relation needs two independent knobs — how big a per-pixel change counts
as movement (this column) and how much of the region must change (the existing
`threshold`) — and one number cannot express both. 0 means "use the default", so
every existing rule is backfilled with it and nothing changes behaviour.

Autogenerate also proposed a REAL→Float retype of `bboxes.confidence`. That is a
no-op rename of the same SQLite storage class on a *generated* column, and applying
it would rebuild the bboxes table for nothing. Dropped deliberately.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bd075a305397"
down_revision: Union[str, None] = "ae83f676e879"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        # server_default is required, not cosmetic: the column is NOT NULL and every
        # existing rule row needs a value at the moment it is added.
        batch_op.add_column(
            sa.Column("motion_sensitivity", sa.Float(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("motion_sensitivity")
