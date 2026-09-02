"""add crop_path to watch_events

Revision ID: 35f138dfc9b3
Revises: 5ec93fb6593c
Create Date: 2026-09-01 22:47:51.581709

The event's second image: the same frame as the snapshot, cropped to the boxes the
event fired on. Nullable with no back-fill here — the geometry is not in this database,
it is in the clips' `.boxes.json` sidecars, so filling it is a one-off pass over the
recordings (`scripts/backfill_event_crops.py`) and not something a migration can do.

Autogenerate also proposed a `bboxes.confidence` REAL→Float alter, as it does on every
run. It is a no-op in SQLite and rebuilds the whole table with its computed column to
achieve nothing, so it is deliberately not here.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "35f138dfc9b3"
down_revision: Union[str, None] = "5ec93fb6593c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.add_column(sa.Column("crop_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.drop_column("crop_path")
