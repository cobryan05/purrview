"""add composite index on bboxes(image_uuid, label_uuid)

Revision ID: b1d4e8f2a093
Revises: a8f3d2c9e1b5
Create Date: 2026-06-08 00:00:00.000000

The dataset label filter added in the camelCase API boundary commit uses a
correlated EXISTS subquery that filters on both BBox.image_uuid and
BBox.label_uuid simultaneously:

    EXISTS (
        SELECT 1 FROM bboxes
        WHERE bboxes.image_uuid = images.uuid
          AND bboxes.label_uuid IN (...)
    )

The prior migration (a8f3d2c9e1b5) added separate single-column indexes on
image_uuid and label_uuid, but SQLite can only use one index per table node
in a query plan. A composite index covering both columns lets SQLite satisfy
the entire EXISTS condition with a single range seek per image UUID instead of
scanning all bboxes for that image and then checking the label.

This index also benefits the tag-filter JOIN that reads
    bboxes.image_uuid AND bboxes.label_uuid
in the same pattern.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1d4e8f2a093"
down_revision: Union[str, None] = "a8f3d2c9e1b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.create_index(
            "ix_bboxes_image_uuid_label_uuid",
            ["image_uuid", "label_uuid"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.drop_index("ix_bboxes_image_uuid_label_uuid")
