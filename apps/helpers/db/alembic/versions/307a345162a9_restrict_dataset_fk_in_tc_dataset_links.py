"""restrict dataset_uuid FK in training_config_dataset_links

Revision ID: 307a345162a9
Revises: f09f3cfe5922
Create Date: 2026-06-18 15:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "307a345162a9"
down_revision: Union[str, None] = "f09f3cfe5922"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite cannot ALTER foreign key constraints; drop and recreate the table
    # with ON DELETE RESTRICT on dataset_uuid so the DB agrees with the app-level
    # check in delete_dataset (which raises a 409 when a training config still
    # references the dataset).
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT training_config_uuid, dataset_uuid FROM training_config_dataset_links")
    ).fetchall()

    op.drop_table("training_config_dataset_links")
    op.create_table(
        "training_config_dataset_links",
        sa.Column("training_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dataset_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_uuid"], ["datasets.uuid"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["training_config_uuid"], ["training_configs.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_config_uuid", "dataset_uuid"),
    )

    if rows:
        conn.execute(
            sa.text(
                "INSERT INTO training_config_dataset_links (training_config_uuid, dataset_uuid) "
                "VALUES (:tc_uuid, :ds_uuid)"
            ),
            [{"tc_uuid": r[0], "ds_uuid": r[1]} for r in rows],
        )


def downgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT training_config_uuid, dataset_uuid FROM training_config_dataset_links")
    ).fetchall()

    op.drop_table("training_config_dataset_links")
    op.create_table(
        "training_config_dataset_links",
        sa.Column("training_config_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dataset_uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_uuid"], ["datasets.uuid"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["training_config_uuid"], ["training_configs.uuid"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_config_uuid", "dataset_uuid"),
    )

    if rows:
        conn.execute(
            sa.text(
                "INSERT INTO training_config_dataset_links (training_config_uuid, dataset_uuid) "
                "VALUES (:tc_uuid, :ds_uuid)"
            ),
            [{"tc_uuid": r[0], "ds_uuid": r[1]} for r in rows],
        )
