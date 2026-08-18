"""rule label slots can include descendants

Revision ID: 48185ebb37cc
Revises: b0811d8fbe8d
Create Date: 2026-08-17 20:33:52.742008

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "48185ebb37cc"
down_revision: Union[str, None] = "b0811d8fbe8d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Autogenerate also proposed a REAL -> Float retype of `bboxes.confidence`, as it did
    # on the previous revision. Dropped by hand for the same reason: it is a no-op on
    # SQLite (both are NUMERIC affinity) and applying it rewrites the whole bboxes table
    # just to restate a generated column's long expression.
    #
    # `server_default` is required, not decorative: these are NOT NULL columns being
    # added to a table that already has rows, and without a default the batch rewrite has
    # nothing to put in them. False is also the right semantic default — an existing rule
    # meant exactly the labels it named.
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "subject_a_include_descendants",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.add_column(
            sa.Column(
                "subject_b_include_descendants",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("subject_b_include_descendants")
        batch_op.drop_column("subject_a_include_descendants")
