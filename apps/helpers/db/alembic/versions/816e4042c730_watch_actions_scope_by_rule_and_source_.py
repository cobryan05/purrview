"""watch actions scope by rule and source lists

Replaces `watch_actions.rule_uuid` (a nullable FK, NULL meaning "every rule") with two
JSON lists, `rule_uuids_json` and `source_uuids_json`, both empty-meaning-all.

The source axis is the point of the change: it lets one action be narrowed to one
camera, which is what removes the reason to duplicate a whole watcher — rules and all —
just to record on the one camera pointed at the litter box.

The rule axis moves to a list at the same time rather than later. A lopsided pair (one
scalar, one list) would need its own migration to fix and would read as an accident in
every piece of code that touches both.

**Data is preserved.** An action scoped to a rule keeps that scope as a one-element
list, and an unscoped action becomes the empty list, which means the same thing it
always did. Nothing changes behaviour for an existing watcher.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "816e4042c730"
down_revision: Union[str, None] = "48185ebb37cc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Added nullable and backfilled rather than declared NOT NULL: SQLite cannot add a
    # NOT NULL column to a populated table without a default, and a JSON default that
    # round-trips correctly is easier to write as an UPDATE than as a server_default.
    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rule_uuids_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("source_uuids_json", sa.JSON(), nullable=True))

    # Carry the old scope across before the column holding it goes away. An action that
    # named a rule keeps naming it; one that named none stays unscoped.
    op.execute("""
        UPDATE watch_actions
           SET rule_uuids_json = CASE
                   WHEN rule_uuid IS NULL OR rule_uuid = '' THEN '[]'
                   ELSE json_array(rule_uuid)
               END,
               source_uuids_json = '[]'
        """)

    # No explicit drop_constraint: batch mode rebuilds the table to drop the column, and
    # the old foreign key goes with it. Naming it would be guesswork — SQLite does not
    # record a name for an inline REFERENCES clause, which is why autogenerate emitted
    # `drop_constraint(None, ...)`, a call that raises.
    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.drop_column("rule_uuid")


def downgrade() -> None:
    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("rule_uuid", sa.VARCHAR(), nullable=True))

    # Lossy in one direction only, and unavoidably so: a scalar column cannot hold an
    # action scoped to two rules, so it keeps the first. The source scope has nowhere to
    # go at all and is dropped, which is what downgrading past the feature means.
    op.execute("""
        UPDATE watch_actions
           SET rule_uuid = json_extract(rule_uuids_json, '$[0]')
         WHERE rule_uuids_json IS NOT NULL
           AND json_array_length(rule_uuids_json) > 0
        """)

    with op.batch_alter_table("watch_actions", schema=None) as batch_op:
        batch_op.drop_column("source_uuids_json")
        batch_op.drop_column("rule_uuids_json")
