"""add single-model selection to watch profiles and sources

A watch source used to be able to name only a Model Stack, so running one model
meant creating a one-member stack for it. These two columns let a source (or the
profile default) name a bare model instead. Exactly one of the pair is set —
enforced in WatchProfileSourceWrite / WatchProfileCreate, not by a DB constraint,
matching how ModelGroupMember already handles its own model_name/child_group_uuid
pair.

Autogenerate also proposed a bboxes.confidence REAL→Float alter; it is dropped
here because the two are the same SQLite type, and a batch_alter_table on bboxes
would rebuild the whole table — and its generated column — for no change at all.

Revision ID: 293b86a081d3
Revises: c4e7a1b93f52
Create Date: 2026-08-16 11:52:13.285949

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "293b86a081d3"
down_revision: Union[str, None] = "c4e7a1b93f52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("model_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        batch_op.add_column(sa.Column("default_model_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("watch_profiles", schema=None) as batch_op:
        batch_op.drop_column("default_model_name")

    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.drop_column("model_name")
