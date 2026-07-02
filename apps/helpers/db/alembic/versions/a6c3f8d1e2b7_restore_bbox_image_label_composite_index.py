"""restore bboxes (image_uuid, label_uuid) composite index

Revision ID: a6c3f8d1e2b7
Revises: d8e2f1a4b6c9
Create Date: 2026-07-02 00:00:00.000000

Migration b1d4e8f2a093 added ix_bboxes_image_uuid_label_uuid for the correlated
EXISTS per-image probes (image_uuid = images.uuid AND label_uuid IN (...)), but
the later phash migration's batch_alter_table recreated the bboxes table and
silently dropped it — SQLite batch mode loses indexes it doesn't know about.
This restores it and drops the now-redundant single-column ix_bboxes_image_uuid:
the composite's leading column covers every plain image_uuid lookup (including
the ondelete=CASCADE child scan), so dropping it cannot regress any query and
keeps write amplification flat. Mirrors what d8e2f1a4b6c9 did on the label side.

Uses direct create_index/drop_index (not batch_alter_table): SQLite supports
index DDL natively, so there is no table-copy — and batch mode is exactly what
lost this index in the first place.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a6c3f8d1e2b7"
down_revision: Union[str, None] = "d8e2f1a4b6c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_bboxes_image_uuid_label_uuid",
        "bboxes",
        ["image_uuid", "label_uuid"],
        unique=False,
    )
    op.drop_index("ix_bboxes_image_uuid", table_name="bboxes")


def downgrade() -> None:
    op.create_index("ix_bboxes_image_uuid", "bboxes", ["image_uuid"], unique=False)
    op.drop_index("ix_bboxes_image_uuid_label_uuid", table_name="bboxes")
