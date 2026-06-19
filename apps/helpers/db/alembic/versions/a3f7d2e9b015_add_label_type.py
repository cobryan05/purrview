"""add label_type to labels

Revision ID: a3f7d2e9b015
Revises: 76cc8e4b7f61
Create Date: 2026-06-10 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op

revision: str = "a3f7d2e9b015"
down_revision: Union[str, None] = "76cc8e4b7f61"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "label_type",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="standard",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.drop_column("label_type")
