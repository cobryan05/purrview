"""add images sort-order indexes for keyset pagination

Revision ID: c7e4a9b2f6d1
Revises: a6c3f8d1e2b7
Create Date: 2026-07-02 00:00:00.000000

Adds (created_at, uuid) and (updated_at, uuid) composites on images. The
image-search list query orders by (sort_col, uuid) with LIMIT; when the bbox
tag/label filters are dispatched to correlated EXISTS for common predicates,
SQLite scans one of these indexes in ORDER BY direction and stops as soon as
LIMIT rows pass the filter (measured 916ms → 0.3ms on a 100k-image DB for the
default review view). Without them the EXISTS strategy always degrades to a
full scan + temp b-tree sort, so the selectivity dispatch depends on these.

Uses direct create_index (not batch_alter_table): SQLite supports index DDL
natively, so there is no table copy, and batch mode silently drops indexes it
doesn't know about.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e4a9b2f6d1"
down_revision: Union[str, None] = "a6c3f8d1e2b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_images_created_at_uuid", "images", ["created_at", "uuid"], unique=False)
    op.create_index("ix_images_updated_at_uuid", "images", ["updated_at", "uuid"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_images_updated_at_uuid", table_name="images")
    op.drop_index("ix_images_created_at_uuid", table_name="images")
