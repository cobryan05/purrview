"""remove the False Positive verification tag

False positives are not a concept in this application — the tag was a user tag
that got promoted into the system verification group, giving bbox verification a
third state that nothing else models. Verification is now two-state:
Unverified / Verified.

Any bbox_tags links to the tag are deleted along with it. On the dev database
there were none (12,264 Unverified / 5,830 Verified / 0 False Positive), so this
destroys no annotation; on a database that did use it, the affected bboxes fall
back to having no verification tag, which reads as Unverified — the same state
they would have had before someone marked them.

Revision ID: c1e5a83f7d92
Revises: b7d31c904ae2
Create Date: 2026-08-05
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c1e5a83f7d92"
down_revision = "b7d31c904ae2"
branch_labels = None
depends_on = None

_FALSE_POS_TAG_UUID = "00000000-0000-0000-0000-000000000003"


def upgrade() -> None:
    conn = op.get_bind()
    # Links first — the tag row is the FK target.
    conn.execute(
        sa.text("DELETE FROM bbox_tags WHERE tag_uuid = :uuid"),
        {"uuid": _FALSE_POS_TAG_UUID},
    )
    conn.execute(
        sa.text("DELETE FROM tags WHERE uuid = :uuid"),
        {"uuid": _FALSE_POS_TAG_UUID},
    )


def downgrade() -> None:
    # One-way: re-seeding the tag would restore the row but not the links, and
    # nothing in the app can produce the state again.
    pass
