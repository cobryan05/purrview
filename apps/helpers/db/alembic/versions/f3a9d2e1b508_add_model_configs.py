"""add model configs

Revision ID: f3a9d2e1b508
Revises: d2f8c5b1e347
Create Date: 2026-06-01 00:00:00.000000

Adds model_configs (named configuration containers) and model_config_nodes
(one row per model slot in the hierarchy, FK→model_configs CASCADE,
FK→labels CASCADE nullable).
"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel.sql.sqltypes

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a9d2e1b508"
down_revision: Union[str, None] = "d2f8c5b1e347"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_configs",
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
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
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("model_configs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_model_configs_name"), ["name"], unique=True)

    op.create_table(
        "model_config_nodes",
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("label_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("model_typename", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_config_uuid"], ["model_configs.uuid"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["label_uuid"], ["labels.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("model_config_nodes", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_model_config_nodes_model_config_uuid"),
            ["model_config_uuid"],
            unique=False,
        )
        batch_op.create_unique_constraint(
            "uq_model_config_nodes_config_label",
            ["model_config_uuid", "label_uuid"],
        )


def downgrade() -> None:
    with op.batch_alter_table("model_config_nodes", schema=None) as batch_op:
        batch_op.drop_constraint("uq_model_config_nodes_config_label", type_="unique")
        batch_op.drop_index(batch_op.f("ix_model_config_nodes_model_config_uuid"))
    op.drop_table("model_config_nodes")
    with op.batch_alter_table("model_configs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_configs_name"))
    op.drop_table("model_configs")
