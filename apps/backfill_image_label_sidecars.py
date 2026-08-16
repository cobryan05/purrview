"""One-shot backfill: write existing image_labels rows into JSON sidecars.

Sidecars only started carrying whole-image label assessments (present / absent)
after they had already been in use for a long time, so every assessment made
before that lives solely in the database — the one place the sidecars exist to
be independent of. This walks the assessments already in the DB and writes them
into the sidecar next to each image.

Idempotent: rerunning rewrites identical content, so an interrupted run can
simply be repeated. An image with no sidecar yet gets a complete one (uuid,
filename, its boxes from the DB, then the assessments) — a boxless image never
had a sidecar to begin with, and it is exactly the kind that carries an
"absent" assessment. An image with nothing to record — no assessments and no
boxes — is left without a file rather than given an empty one.

Run inside the container:

    python -m apps.backfill_image_label_sidecars
"""

import asyncio
import logging
import os
from pathlib import Path

from apps import DB_DIR
from apps.db_server.manager import Manager
from apps.helpers.db.db_client import DbClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


async def main() -> None:
    files_root = Path(os.environ.get("FILES_ROOT", "/app/image_datasets"))
    db_client = DbClient(DB_DIR, files_root=files_root)
    # Deliberately not init_db(): this is a repair pass over an existing
    # database, and running migrations as a side effect of a backfill is a
    # surprise nobody asked for.
    manager = Manager(db_client=db_client, files_root=files_root)
    try:
        stats = await manager.backfill_image_label_sidecars()
    finally:
        await db_client.close()

    print(
        f"updated {stats['written']} sidecars, created {stats['created']}; "
        f"skipped {stats['skipped_no_data']} with nothing to record, "
        f"{stats['skipped_no_path']} with no resolvable path"
    )


if __name__ == "__main__":
    asyncio.run(main())
