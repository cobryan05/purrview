"""add phash to images and bboxes

Revision ID: 53e1fcd8be40
Revises: b1d4e8f2a093
Create Date: 2026-06-08 05:43:36.531402

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '53e1fcd8be40'
down_revision: Union[str, None] = 'b1d4e8f2a093'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('bboxes', schema=None) as batch_op:
        batch_op.add_column(sa.Column('phash', sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.add_column(sa.Column('phash', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('images', schema=None) as batch_op:
        batch_op.drop_column('phash')

    with op.batch_alter_table('bboxes', schema=None) as batch_op:
        batch_op.drop_column('phash')
