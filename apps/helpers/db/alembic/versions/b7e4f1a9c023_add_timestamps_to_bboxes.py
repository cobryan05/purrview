"""add timestamps to bboxes

Revision ID: b7e4f1a9c023
Revises: a1b2c3d4e5f6
Create Date: 2026-05-29 00:00:00.000000

Bboxes now carry created_at and updated_at so that bbox-age filters can
query directly on the bbox table rather than relying solely on the parent
image's updated_at (which is touched on every bbox write via _touch_image).
Existing rows are backfilled with CURRENT_TIMESTAMP.
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7e4f1a9c023"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("created_at")
