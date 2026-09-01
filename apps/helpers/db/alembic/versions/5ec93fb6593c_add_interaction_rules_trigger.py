"""add interaction_rules trigger

Revision ID: 5ec93fb6593c
Revises: c9d1e4f7a2b3
Create Date: 2026-09-01 00:21:38.584445

Whether a rule's event spans its predicate or is an impulse at one of its edges
(`while` / `on_enter` / `on_exit`) — see `RuleTrigger`.

`server_default 'while'` rather than a nullable column: every rule that exists when
this runs was written before edges did, and `while` is the only behaviour it has ever
had. A NULL would mean "unset" to a reader that has no such state.

Autogenerate also proposed a REAL -> Float alteration on `bboxes.confidence`. That is
a no-op spelling difference on a computed column, and rebuilding that table to apply it
would rewrite the generated expression for nothing. Dropped deliberately; if it shows
up in the next autogenerate too, it is still noise.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5ec93fb6593c"
down_revision: Union[str, None] = "c9d1e4f7a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("trigger", sa.String(), server_default=sa.text("'while'"), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("trigger")
