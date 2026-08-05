"""drop model_bundles

Revision ID: 9c62f3502b03
Revises: c1e5a83f7d92
Create Date: 2026-08-05 18:38:25.690876

The ModelBundle concept is gone: TrainBundleTask and the "Train all → bundle"
UI were removed, leaving this table with no writer and no reader. A training
run's image manifest — the one thing manifest_json held that wasn't derivable —
is recorded per trained model in the sidecar JSON beside its ONNX
(train_image_uuids / val_image_uuids), so nothing is lost by dropping it.

Guarded the same way f93545d9a50b is: SQLite's DDL is non-transactional, so a
half-applied migration can leave the table already gone, and re-running must
not raise.

"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9c62f3502b03"
down_revision: Union[str, None] = "c1e5a83f7d92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def upgrade() -> None:
    conn = op.get_bind()
    # Dropping the table drops ix_model_bundles_name with it — no separate
    # drop_index step (which would raise on a DB where the index never existed).
    if "model_bundles" in _tables(conn):
        op.drop_table("model_bundles")


def downgrade() -> None:
    conn = op.get_bind()
    if "model_bundles" in _tables(conn):
        return
    # Mirrors f93545d9a50b's original CREATE, including its column types.
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
