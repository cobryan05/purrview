"""add recording overlay_path

Revision ID: f8f31a7ddc86
Revises: 814ef041c4b1
Create Date: 2026-08-18 20:23:09.845973

Autogenerate also proposed rewriting `bboxes.confidence` from REAL to Float. That is
noise: the column is a generated one whose type SQLite reports back differently from the
way SQLModel spells it, and nothing about it changed here. Batch-altering a computed
column rewrites the whole table for no reason, so it is deliberately not in this
revision.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8f31a7ddc86"
down_revision: Union[str, None] = "814ef041c4b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("recordings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("overlay_path", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("recordings", schema=None) as batch_op:
        batch_op.drop_column("overlay_path")
