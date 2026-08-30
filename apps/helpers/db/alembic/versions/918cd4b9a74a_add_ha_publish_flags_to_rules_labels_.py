"""add HA publish flags to rules, labels and watched sources

Revision ID: 918cd4b9a74a
Revises: 7efcbb5f6ba0
Create Date: 2026-08-30 10:59:58.081706

Four booleans that decide what the MQTT bridge represents in Home Assistant. None of
them changes what is watched, evaluated or recorded — they gate the HA projection only.

`server_default` is not decoration here: these are NOT NULL columns added to tables that
already have rows, which SQLite rejects outright without one. The defaults are also the
migration's answer for existing installs, so they are chosen to preserve current
behaviour everywhere except the one place the change is deliberate:

* `interaction_rules.publish_to_ha`     -> 1, every existing rule keeps its entities.
* `labels.publish_to_ha`                -> 1, every existing subject keeps its device.
* `watch_profile_sources.publish_to_ha` -> 1, every watched camera keeps its device.
* `labels.publish_per_source`           -> 0, the deliberate change. This is the
  (label x rule x source) counter tier, the only part of the entity count that
  multiplies all three axes; it becomes opt-in per label. Existing installs lose those
  counters on the next reconcile, which is a retraction, not a silent stop.

Autogenerate also proposed a `bboxes.confidence` REAL -> Float alteration. That is a
no-op type-affinity difference on a *computed* column, and applying it would rewrite the
whole bboxes table (the largest in the database) to change nothing. Dropped deliberately
— do not reinstate it if a later autogenerate offers it again.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "918cd4b9a74a"
down_revision: Union[str, None] = "7efcbb5f6ba0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("publish_to_ha", sa.Boolean(), nullable=False, server_default=sa.text("1")))

    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.add_column(sa.Column("publish_to_ha", sa.Boolean(), nullable=False, server_default=sa.text("1")))
        batch_op.add_column(sa.Column("publish_per_source", sa.Boolean(), nullable=False, server_default=sa.text("0")))

    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("publish_to_ha", sa.Boolean(), nullable=False, server_default=sa.text("1")))


def downgrade() -> None:
    with op.batch_alter_table("watch_profile_sources", schema=None) as batch_op:
        batch_op.drop_column("publish_to_ha")

    with op.batch_alter_table("labels", schema=None) as batch_op:
        batch_op.drop_column("publish_per_source")
        batch_op.drop_column("publish_to_ha")

    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("publish_to_ha")
