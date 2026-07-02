"""optimize label reverse-lookup indexes

Revision ID: d8e2f1a4b6c9
Revises: 4a5518992974
Create Date: 2026-07-02 00:00:00.000000

Two index changes for the image-search / dataset-assembly hot paths:

1. bboxes: replace the single-column ix_bboxes_label_uuid with a composite
   (label_uuid, image_uuid). The dataset/label filters do
   `SELECT [DISTINCT] image_uuid FROM bboxes WHERE label_uuid = ?`; the
   composite makes this a COVERING index seek and satisfies the DISTINCT
   without a temp b-tree (EXPLAIN: `SEARCH ... USING COVERING INDEX
   ix_bboxes_label_uuid_image_uuid (label_uuid=?)`). Its leading column also
   covers every plain label_uuid lookup the dropped single-column index served,
   so dropping that index cannot regress any query and keeps write amplification
   flat. The existing (image_uuid, label_uuid) composite is kept — it serves the
   correlated-EXISTS image_label probes.

2. image_labels: add ix_image_labels_label_uuid. The composite PK leads with
   image_uuid, so reverse lookups by label_uuid alone (list_images_for_label /
   dataset assembly) full-scanned the table.

Uses direct create_index/drop_index (not batch_alter_table): SQLite supports
index DDL natively, so there is no table-copy — important on a large bboxes
table where a batch recreate would rewrite every row under a write lock.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8e2f1a4b6c9"
down_revision: Union[str, None] = "4a5518992974"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_bboxes_label_uuid", table_name="bboxes")
    op.create_index(
        "ix_bboxes_label_uuid_image_uuid",
        "bboxes",
        ["label_uuid", "image_uuid"],
        unique=False,
    )
    op.create_index(
        "ix_image_labels_label_uuid",
        "image_labels",
        ["label_uuid"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_image_labels_label_uuid", table_name="image_labels")
    op.drop_index("ix_bboxes_label_uuid_image_uuid", table_name="bboxes")
    op.create_index("ix_bboxes_label_uuid", "bboxes", ["label_uuid"], unique=False)
