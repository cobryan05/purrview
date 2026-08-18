"""interaction rules scope by source

Adds `interaction_rules.source_uuids_json`, an empty-meaning-all list mirroring the pair
already on `watch_actions`.

Scoping the *action* was only ever half the answer. It narrows the side effect but
leaves the rule evaluating on every camera, so "a cat is present" — true in every room,
all day — still opens an event per room and fills the event browser with rows the user
said they did not care about. Scoping the rule suppresses the event itself.

**No behaviour changes for an existing watcher.** Every rule is backfilled to `[]`,
which means what it has always meant: evaluate on every source in the profile.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f5c1a9b7e20"
down_revision: Union[str, None] = "d1d84f146112"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Nullable + backfill, as `816e4042c730` did for the action scopes: SQLite cannot add
    # a NOT NULL column to a populated table without a default, and a JSON default is
    # easier to get right as an UPDATE than as a server_default.
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_uuids_json", sa.JSON(), nullable=True))

    op.execute("UPDATE interaction_rules SET source_uuids_json = '[]'")


def downgrade() -> None:
    # Lossy, and unavoidably so: a rule narrowed to one camera goes back to running on
    # all of them, which is what downgrading past the feature means. It fails open — the
    # rule fires more than it used to, never less.
    with op.batch_alter_table("interaction_rules", schema=None) as batch_op:
        batch_op.drop_column("source_uuids_json")
