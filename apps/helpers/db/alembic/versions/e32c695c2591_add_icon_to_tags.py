"""add icon to tags

Revision ID: e32c695c2591
Revises: 70896580cdb5
Create Date: 2026-05-26 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e32c695c2591"
down_revision: Union[str, None] = "70896580cdb5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("tags", schema=None) as batch_op:
        batch_op.add_column(sa.Column("icon", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("tags", schema=None) as batch_op:
        batch_op.drop_column("icon")
