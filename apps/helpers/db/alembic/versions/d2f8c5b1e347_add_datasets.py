"""add datasets

Revision ID: d2f8c5b1e347
Revises: b7e4f1a9c023
Create Date: 2026-05-29 00:00:00.000000

Adds the datasets table (name, description, filter_json, timestamps) and
dataset_label_links join table (dataset_uuid → datasets, label_uuid → labels,
both with CASCADE deletes).
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2f8c5b1e347"
down_revision: Union[str, None] = "b7e4f1a9c023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("filter_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_datasets_name"), ["name"], unique=True)

    op.create_table(
        "dataset_label_links",
        sa.Column("dataset_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("label_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_uuid"], ["datasets.uuid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["label_uuid"], ["labels.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("dataset_uuid", "label_uuid"),
    )


def downgrade() -> None:
    op.drop_table("dataset_label_links")
    with op.batch_alter_table("datasets", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_datasets_name"))
    op.drop_table("datasets")
