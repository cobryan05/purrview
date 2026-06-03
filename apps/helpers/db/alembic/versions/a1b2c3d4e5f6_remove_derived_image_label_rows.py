"""remove derived image_label rows (uncertain and false-pos absent)

Revision ID: a1b2c3d4e5f6
Revises: e32c695c2591
Create Date: 2026-05-27 00:00:00.000000

image_labels now stores only:
  - PRESENT rows  (derived from verified bboxes, rebuilt automatically)
  - ABSENT  rows  (explicit user marks, never touched by recompute)

UNCERTAIN rows were never meaningful to store (implicit default = no row).
Derived ABSENT rows (all-false-positive bboxes) are no longer written.
This migration removes all non-PRESENT rows left from the old approach.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "e32c695c2591"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM image_labels WHERE status != 'present'")


def downgrade() -> None:
    pass  # no way to recreate derived rows; downgrade is a no-op
