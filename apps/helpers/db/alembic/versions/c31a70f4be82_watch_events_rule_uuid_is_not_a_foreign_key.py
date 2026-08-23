"""watch_events.rule_uuid is history, not a foreign key

Revision ID: c31a70f4be82
Revises: bd075a305397
Create Date: 2026-08-23 18:05:00.000000

`watch_events.rule_uuid` carried `ON DELETE SET NULL` to `interaction_rules.uuid`.
Saving a watch profile replaces its nested lists wholesale — `update_watch_profile`
deletes every `InteractionRule` for the profile and re-inserts them carrying their
original uuids — so each save fired SET NULL across the whole event history. The rules
came back with the same uuids; the events did not get their pointers back.

Symptom: the NVR Event type filter only ever matched events fired since the last profile
save. Everything older had `rule_uuid IS NULL`, which the facets query excludes outright,
so those events were unreachable by that filter and invisible in its dropdown counts.

Two parts:

1. Drop the constraint. The column is a historical reference like `rule_name` beside it,
   not a live link — the facets endpoint reads it *because* it wants rules that no longer
   exist to stay filterable, which SET NULL made impossible.
2. Backfill the rows already NULLed, matching `rule_name` back to a rule of that name in
   the same profile. Only where the name resolves to exactly one rule: a profile with two
   rules sharing a name cannot be disambiguated after the fact, and guessing would
   attribute events to the wrong rule rather than leave them unattributed.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c31a70f4be82"
down_revision: Union[str, None] = "bd075a305397"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Scoped to the event's own profile, because rule names are only unique within one.
# `HAVING count(*) = 1` is what keeps an ambiguous name unattributed instead of wrong.
_BACKFILL = sa.text("""
    UPDATE watch_events
       SET rule_uuid = (
           SELECT r.uuid
             FROM interaction_rules r
            WHERE r.profile_uuid = watch_events.profile_uuid
              AND r.name = watch_events.rule_name
            GROUP BY r.name
           HAVING count(*) = 1
       )
     WHERE rule_uuid IS NULL
       AND rule_name IS NOT NULL
    """)


def upgrade() -> None:
    # naming_convention so batch mode can address the unnamed FK SQLite reflected from
    # the original CREATE TABLE; without it `drop_constraint` has no handle to drop.
    with op.batch_alter_table(
        "watch_events",
        schema=None,
        naming_convention={"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"},
    ) as batch_op:
        batch_op.drop_constraint("fk_watch_events_rule_uuid_interaction_rules", type_="foreignkey")

    op.execute(_BACKFILL)


def downgrade() -> None:
    # Restoring the constraint requires every existing rule_uuid to resolve, and the
    # backfill above may have attributed events to rules deleted since. Clear the
    # danglers first, which is exactly what the constraint would have done to them.
    op.execute(sa.text("""
            UPDATE watch_events
               SET rule_uuid = NULL
             WHERE rule_uuid IS NOT NULL
               AND rule_uuid NOT IN (SELECT uuid FROM interaction_rules)
            """))
    with op.batch_alter_table("watch_events", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_watch_events_rule_uuid_interaction_rules",
            "interaction_rules",
            ["rule_uuid"],
            ["uuid"],
            ondelete="SET NULL",
        )
