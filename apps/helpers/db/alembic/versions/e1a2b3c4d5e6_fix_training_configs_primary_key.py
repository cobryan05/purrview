"""fix training_configs primary key lost by batch alter

Revision ID: e1a2b3c4d5e6
Revises: 307a345162a9
Create Date: 2026-06-18 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1a2b3c4d5e6"
down_revision: Union[str, None] = "307a345162a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rebuild training_configs with its PRIMARY KEY intact.

    The batch_alter_table in f09f3cfe5922 (drop dataset_uuid column) silently
    rebuilt the table without the PRIMARY KEY constraint, causing SQLite to
    report "foreign key mismatch" on any DML against training_config_*_links.
    """
    conn = op.get_bind()

    # Snapshot all data before dropping anything
    tc_rows = conn.execute(
        sa.text(
            "SELECT created_at, updated_at, uuid, name, description, "
            "trainer_type, model_config_uuid, hyperparams_json FROM training_configs"
        )
    ).fetchall()

    ll_rows = conn.execute(
        sa.text("SELECT training_config_uuid, label_uuid FROM training_config_label_links")
    ).fetchall()

    dl_rows = conn.execute(
        sa.text("SELECT training_config_uuid, dataset_uuid FROM training_config_dataset_links")
    ).fetchall()

    # Drop child tables first (avoids any FK-related surprises on recreation)
    op.drop_table("training_config_dataset_links")
    op.drop_table("training_config_label_links")
    op.drop_table("training_configs")

    # Recreate training_configs with explicit PRIMARY KEY
    op.create_table(
        "training_configs",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("trainer_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("hyperparams_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["model_config_uuid"], ["model_configs.uuid"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("training_configs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_training_configs_name"), ["name"], unique=True)

    # Recreate child tables
    op.create_table(
        "training_config_label_links",
        sa.Column("training_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("label_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["label_uuid"], ["labels.uuid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["training_config_uuid"], ["training_configs.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_config_uuid", "label_uuid"),
    )
    op.create_table(
        "training_config_dataset_links",
        sa.Column("training_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dataset_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_uuid"], ["datasets.uuid"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["training_config_uuid"], ["training_configs.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_config_uuid", "dataset_uuid"),
    )

    # Re-insert data
    if tc_rows:
        conn.execute(
            sa.text(
                "INSERT INTO training_configs "
                "(created_at, updated_at, uuid, name, description, trainer_type, "
                "model_config_uuid, hyperparams_json) "
                "VALUES (:created_at, :updated_at, :uuid, :name, :description, "
                ":trainer_type, :model_config_uuid, :hyperparams_json)"
            ),
            [
                {
                    "created_at": r[0],
                    "updated_at": r[1],
                    "uuid": r[2],
                    "name": r[3],
                    "description": r[4],
                    "trainer_type": r[5],
                    "model_config_uuid": r[6],
                    "hyperparams_json": r[7],
                }
                for r in tc_rows
            ],
        )
    if ll_rows:
        conn.execute(
            sa.text(
                "INSERT INTO training_config_label_links (training_config_uuid, label_uuid) "
                "VALUES (:tc_uuid, :l_uuid)"
            ),
            [{"tc_uuid": r[0], "l_uuid": r[1]} for r in ll_rows],
        )
    if dl_rows:
        conn.execute(
            sa.text(
                "INSERT INTO training_config_dataset_links (training_config_uuid, dataset_uuid) "
                "VALUES (:tc_uuid, :ds_uuid)"
            ),
            [{"tc_uuid": r[0], "ds_uuid": r[1]} for r in dl_rows],
        )


def downgrade() -> None:
    # Downgrade isn't meaningful here — the previous state was broken.
    # Mirror upgrade to leave things in the correct shape.
    pass
