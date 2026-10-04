"""add default_manual_recording_s to watch profiles

Revision ID: d8a5323e69d8
Revises: 12092cc491f4
Create Date: 2026-10-04 08:33:24.322642

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8a5323e69d8"
down_revision: Union[str, None] = "12092cc491f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        # server_default, even though the model carries the same default: existing rows
        # have no value for a NOT NULL column, and SQLite rejects the ALTER outright
        # without one.
        batch_op.add_column(sa.Column("default_manual_recording_s", sa.Float(), nullable=False, server_default="300.0"))


def downgrade() -> None:
    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        batch_op.drop_column("default_manual_recording_s")
