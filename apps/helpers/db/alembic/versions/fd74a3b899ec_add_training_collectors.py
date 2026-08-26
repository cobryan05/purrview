"""add training_collectors, and move collect params out of task configs

Revision ID: fd74a3b899ec
Revises: c31a70f4be82
Create Date: 2026-08-25 21:28:39.841239

The collect parameter set becomes a reusable entity so a watcher can drive one
without owning thirty knobs. Existing `CollectTrainingImagesTask` configs carry
those knobs inline, so each mints one collector row and keeps only the params
that belong to the loop it drives.

Supporting both shapes was rejected: "params may be inline or by reference" is a
fork that would have to be honoured in the spec builder, the UI and every future
knob, forever.
"""

import json
from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
import sqlmodel
from alembic import op

from apps.helpers.db.types import UtcDateTime

# revision identifiers, used by Alembic.
revision: str = "fd74a3b899ec"
down_revision: Union[str, None] = "c31a70f4be82"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TYPENAME = "CollectTrainingImagesTask"
# The params that belong to whoever rotates providers and paces the fetch, not to
# the collector. These stay on the task config; everything else moves.
LOOP_KEYS = ("source_uuids", "snapshot_interval", "stream_interval")


def _split_params(params: dict) -> tuple[dict, dict]:
    """(collector params, params that stay on the config).

    Underscore-prefixed keys are task-machinery reserved keys (`_persistent`,
    `_restart_info`, `_config_name`) — they are not collector configuration and
    the manager still reads them off the config.
    """
    collector, loop = {}, {}
    for key, value in params.items():
        if key in LOOP_KEYS or key.startswith("_"):
            loop[key] = value
        else:
            collector[key] = value
    return collector, loop


def upgrade() -> None:
    op.create_table(
        "training_collectors",
        sa.Column("created_at", UtcDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", UtcDateTime(timezone=True), nullable=False),
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("params_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("uuid"),
    )
    with op.batch_alter_table("training_collectors", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_training_collectors_name"), ["name"], unique=True)

    conn = op.get_bind()
    configs = conn.execute(
        sa.text("SELECT uuid, name, params_json FROM task_configs WHERE typename = :t"),
        {"t": TYPENAME},
    ).fetchall()

    taken: set[str] = set()
    for config_uuid, config_name, raw_params in configs:
        params = json.loads(raw_params) if isinstance(raw_params, str) else (raw_params or {})
        collector_params, loop_params = _split_params(params)

        base_name = config_name or "Collector"
        name = base_name
        suffix = 2
        while name in taken:
            name = f"{base_name} ({suffix})"
            suffix += 1
        taken.add(name)

        collector_uuid = str(uuid4())
        conn.execute(
            sa.text(
                "INSERT INTO training_collectors "
                "(uuid, name, description, enabled, params_json, created_at, updated_at) "
                "VALUES (:u, :n, :d, 1, :p, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {
                "u": collector_uuid,
                "n": name,
                "d": f"Migrated from job config {config_name!r}",
                "p": json.dumps(collector_params),
            },
        )
        loop_params["collector_uuid"] = collector_uuid
        conn.execute(
            sa.text("UPDATE task_configs SET params_json = :p WHERE uuid = :u"),
            {"p": json.dumps(loop_params), "u": config_uuid},
        )


def downgrade() -> None:
    conn = op.get_bind()
    configs = conn.execute(
        sa.text("SELECT uuid, params_json FROM task_configs WHERE typename = :t"),
        {"t": TYPENAME},
    ).fetchall()

    for config_uuid, raw_params in configs:
        params = json.loads(raw_params) if isinstance(raw_params, str) else (raw_params or {})
        collector_uuid = params.pop("collector_uuid", None)
        if not collector_uuid:
            continue
        row = conn.execute(
            sa.text("SELECT params_json FROM training_collectors WHERE uuid = :u"),
            {"u": collector_uuid},
        ).fetchone()
        if row is not None:
            stored = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
            params = {**stored, **params}
        conn.execute(
            sa.text("UPDATE task_configs SET params_json = :p WHERE uuid = :u"),
            {"p": json.dumps(params), "u": config_uuid},
        )

    with op.batch_alter_table("training_collectors", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_training_collectors_name"))
    op.drop_table("training_collectors")
