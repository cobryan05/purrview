"""add stationary_ignore_after_s to interaction rules

Revision ID: bbd84526644c
Revises: 787060497968
Create Date: 2026-11-21 12:51:58.707505

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bbd84526644c"
down_revision: Union[str, None] = "787060497968"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("stationary_ignore_after_s", sa.Float(), server_default=sa.text("0"), nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("stationary_ignore_after_s")
