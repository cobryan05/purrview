"""add performance indexes for dataset search

Revision ID: a8f3d2c9e1b5
Revises: c4a7b2e9f016
Create Date: 2026-06-07 00:00:00.000000

Without these indexes every EXISTS subquery used by the dataset-preview and
image-search endpoints performs a full table scan of the bboxes table.
On any non-trivial dataset this makes the preview endpoint take many seconds
and holds the SQLite write lock, blocking all other db_server requests.

Indexes added:
  bboxes.image_uuid  — used by every per-image EXISTS subquery
  bboxes.label_uuid  — used by the dataset label filter EXISTS subquery
  images.updated_at  — used for ORDER BY in the default (updated_at desc) sort
  images.created_at  — used for ORDER BY in the alternative (created_at desc) sort
  bbox_tags.tag_uuid — allows reverse tag→bbox lookups without scanning bbox_tags
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a8f3d2c9e1b5"
down_revision: Union[str, None] = "c4a7b2e9f016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.create_index("ix_bboxes_image_uuid", ["image_uuid"], unique=False)
        batch_op.create_index("ix_bboxes_label_uuid", ["label_uuid"], unique=False)

    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.create_index("ix_images_updated_at", ["updated_at"], unique=False)
        batch_op.create_index("ix_images_created_at", ["created_at"], unique=False)

    with op.batch_alter_table("bbox_tags", schema=None) as batch_op:
        batch_op.create_index("ix_bbox_tags_tag_uuid", ["tag_uuid"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("bbox_tags", schema=None) as batch_op:
        batch_op.drop_index("ix_bbox_tags_tag_uuid")

    with op.batch_alter_table("images", schema=None) as batch_op:
        batch_op.drop_index("ix_images_created_at")
        batch_op.drop_index("ix_images_updated_at")

    with op.batch_alter_table("bboxes", schema=None) as batch_op:
        batch_op.drop_index("ix_bboxes_label_uuid")
        batch_op.drop_index("ix_bboxes_image_uuid")
