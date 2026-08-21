"""add per-watcher blip analysis settings

Revision ID: 0fa01c3d7d3a
Revises: 9b19f7c2223f
Create Date: 2026-08-21 09:14:02.331907

Hand-adjusted after autogenerate, the same two ways as the previous revision:

* `blip_analysis_enabled` gets a `server_default`. Autogenerate emitted it
  `nullable=False` with none, which fails against any database that already holds
  watchers. Existing profiles adopt the feature switched on, matching the behaviour
  they already had before it was configurable.
* The spurious `bboxes.confidence` REAL -> Float retype was dropped again. It is a
  reflection artifact on a *computed* column, unrelated to this change, and applying it
  rebuilds the table for no benefit.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0fa01c3d7d3a"
down_revision: Union[str, None] = "9b19f7c2223f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("blip_analysis_enabled", sa.Boolean(), nullable=False, server_default=sa.true()))
        # Nullable, like every other JSON column here: a row written before this
        # revision holds SQL NULL rather than `{}`, and the read model normalises it.
        batch_op.add_column(sa.Column("blip_thresholds_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        batch_op.drop_column("blip_thresholds_json")
        batch_op.drop_column("blip_analysis_enabled")
