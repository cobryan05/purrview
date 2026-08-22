"""add images.source_uuid

Revision ID: 7799a2dd977e
Revises: 0fa01c3d7d3a
Create Date: 2026-08-21 21:53:58.855791

Records which source (camera / stream / file) produced an image, so search can
filter by origin. NULL for manual uploads and for everything collected before
this column existed — there is nothing to backfill from, since the only trace
of a source on older rows is a sanitised prefix in the filename, which is
lossy (two sources whose names sanitise alike are indistinguishable).

Autogenerate additionally proposed a bboxes.confidence REAL -> Float column
type change. That is a spelling difference between SQLite's reflected type and
SQLAlchemy's generic one, not a schema change, and acting on it would rebuild
the bboxes table (and its generated column) for nothing. Deliberately dropped.
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7799a2dd977e"
down_revision: Union[str, None] = "0fa01c3d7d3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Named explicitly: SQLite has no ALTER for constraints, so alembic emulates a
# drop by rebuilding the table, and it cannot rebuild what it cannot name.
_FK_NAME = "fk_images_source_uuid_sources"


def upgrade() -> None:
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
        batch_op.create_index(batch_op.f("ix_images_source_uuid"), ["source_uuid"], unique=False)
        batch_op.create_foreign_key(_FK_NAME, "sources", ["source_uuid"], ["uuid"], ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.drop_constraint(_FK_NAME, type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_images_source_uuid"))
        batch_op.drop_column("source_uuid")
