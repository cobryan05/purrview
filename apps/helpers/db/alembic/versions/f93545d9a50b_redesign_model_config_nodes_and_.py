"""redesign model config nodes and training configs for type arch split and bundle support

Revision ID: f93545d9a50b
Revises: e1a2b3c4d5e6
Create Date: 2026-06-19 04:26:34.989423

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f93545d9a50b"
down_revision: Union[str, None] = "e1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This migration was previously broken (drop_constraint(None) raised ValueError
    # inside the training_configs batch block).  SQLite's non-transactional DDL
    # committed the first three steps before the failure, leaving some databases
    # in a partially-migrated state (model_bundles created, training_config_label_links
    # dropped, model_config_nodes altered) while still stamped at the prior revision.
    # Each step below is guarded so the migration is safe to re-run.

    conn = op.get_bind()

    def _tables():
        return {r[0] for r in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    def _cols(table):
        return {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}

    # Step 1: create model_bundles (may already exist from a prior partial run)
    if "model_bundles" not in _tables():
        op.create_table(
            "model_bundles",
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column("model_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("bundle_path", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
            sa.Column("manifest_json", sa.JSON(), nullable=True),
            sa.Column("training_config_uuids", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["model_config_uuid"], ["model_configs.uuid"], ondelete="RESTRICT"),
            sa.PrimaryKeyConstraint("uuid"),
        )
        with op.batch_alter_table("model_bundles", schema=None) as batch_op:
            batch_op.create_index(batch_op.f("ix_model_bundles_name"), ["name"], unique=True)

    # Step 2: drop training_config_label_links (may already be gone)
    if "training_config_label_links" in _tables():
        op.drop_table("training_config_label_links")

    # Step 3: alter model_config_nodes.
    # Alembic batch mode uses _alembic_tmp_<table> as a scratch table.  A prior
    # failed run may have left it behind in one of two states:
    #   (a) temp exists, original also exists  → drop the stale temp, redo the alter
    #   (b) temp exists, original was already dropped  → just complete the rename
    _TMP_MCN = "_alembic_tmp_model_config_nodes"
    if "model_config_nodes" not in _tables():
        # Case (b): original was dropped but rename never happened
        if _TMP_MCN in _tables():
            conn.exec_driver_sql(f'ALTER TABLE "{_TMP_MCN}" RENAME TO model_config_nodes')
    elif "model_typename" in _cols("model_config_nodes"):
        # Case (a) or clean path: needs the batch alter
        if _TMP_MCN in _tables():
            conn.exec_driver_sql(f'DROP TABLE "{_TMP_MCN}"')
        with op.batch_alter_table("model_config_nodes", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "model_type",
                    sqlmodel.sql.sqltypes.AutoString(),
                    nullable=False,
                    server_default="object_detection",
                )
            )
            batch_op.add_column(
                sa.Column(
                    "architecture",
                    sqlmodel.sql.sqltypes.AutoString(),
                    nullable=False,
                    server_default="yolo_v8_n",
                )
            )
            batch_op.drop_column("model_typename")
    # else: already fully migrated, nothing to do

    # Step 4: alter training_configs.  Same temp-table guard as step 3.
    _TMP_TC = "_alembic_tmp_training_configs"
    if "training_configs" not in _tables():
        # Case (b): original dropped but rename pending
        if _TMP_TC in _tables():
            conn.exec_driver_sql(f'ALTER TABLE "{_TMP_TC}" RENAME TO training_configs')
    elif "model_config_node_uuid" not in _cols("training_configs"):
        # Case (a) or clean path: needs the batch alter
        if _TMP_TC in _tables():
            conn.exec_driver_sql(f'DROP TABLE "{_TMP_TC}"')
        with op.batch_alter_table("training_configs", schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "model_config_node_uuid",
                    sqlmodel.sql.sqltypes.AutoString(),
                    nullable=True,
                )
            )
            batch_op.create_foreign_key(
                "fk_training_configs_model_config_node_uuid",
                "model_config_nodes",
                ["model_config_node_uuid"],
                ["uuid"],
                ondelete="SET NULL",
            )
            batch_op.drop_column("trainer_type")
            batch_op.drop_column("model_config_uuid")
    # else: already fully migrated, nothing to do


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("training_configs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("model_config_uuid", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("trainer_type", sa.VARCHAR(), nullable=False))
        batch_op.create_foreign_key(
            "fk_training_configs_model_config_uuid",
            "model_configs",
            ["model_config_uuid"],
            ["uuid"],
            ondelete="SET NULL",
        )
        batch_op.drop_column("model_config_node_uuid")

    with op.batch_alter_table("model_config_nodes", schema=None) as batch_op:
        batch_op.add_column(sa.Column("model_typename", sa.VARCHAR(), nullable=False))
        batch_op.drop_column("architecture")
        batch_op.drop_column("model_type")

    op.create_table(
        "training_config_label_links",
        sa.Column("training_config_uuid", sa.VARCHAR(), nullable=False),
        sa.Column("label_uuid", sa.VARCHAR(), nullable=False),
        sa.ForeignKeyConstraint(["label_uuid"], ["labels.uuid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["training_config_uuid"], ["training_configs.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_config_uuid", "label_uuid"),
    )
    with op.batch_alter_table("model_bundles", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_bundles_name"))

    op.drop_table("model_bundles")
    # ### end Alembic commands ###
