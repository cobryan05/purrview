"""watch action name

A human label on a watch action, so a watcher's list of actions is readable.
Nullable and never read by the engine: an unnamed action still says what kind it is,
and every action that exists today has no name.

Revision ID: d1d84f146112
Revises: 387e8e43b380
Create Date: 2026-08-18 14:37:44.887589

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1d84f146112"
down_revision: Union[str, None] = "387e8e43b380"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.drop_column("name")
