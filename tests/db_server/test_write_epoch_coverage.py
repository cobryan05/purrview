"""The review-count cache's write epoch only invalidates for writes that bump it.

`_bump_write_epoch()` is called by hand after every Core-statement (non-ORM) write
to a count-relevant table, because the `after_flush` ORM listener never sees a
`session.exec(delete(...))`/`sqlite_insert(...)` statement — see the write-epoch
comment block in `db_client.py`. That's a list maintained by hand at each call
site, which is exactly the kind of list that silently goes stale: two Core-delete
call sites (`delete_image_labels`, `remove_image_tags`) were added without the
matching bump, so removing a label or a tag from an image left a cached review
count describing rows that no longer existed, for up to the cache's 60s TTL.
"""

from pathlib import Path

from apps.helpers.db.types import ImageItemStatus, Label, Tag


async def _image_uuid(db_client, filename: str) -> str:
    record = await db_client.add_image(Path(filename))
    return record.uuid


async def _label_uuid(db_client, name: str) -> str:
    async with db_client._async_session_maker() as session:
        label = Label(name=name)
        session.add(label)
        await session.commit()
        await session.refresh(label)
        return label.uuid


async def _tag_uuid(db_client, name: str) -> str:
    async with db_client._async_session_maker() as session:
        tag = Tag(name=name)
        session.add(tag)
        await session.commit()
        await session.refresh(tag)
        return tag.uuid


class TestWriteEpochCoversCoreDeletes:
    async def test_delete_image_labels_bumps_the_write_epoch(self, db_client):
        image_uuid = await _image_uuid(db_client, "a.png")
        label_uuid = await _label_uuid(db_client, "Cat")
        await db_client.set_image_labels_status(image_uuid, [label_uuid], ImageItemStatus.PRESENT)
        epoch_before = db_client.write_epoch

        await db_client.delete_image_labels(image_uuid, [label_uuid])

        assert db_client.write_epoch > epoch_before, (
            "delete_image_labels is a Core statement, not an ORM delete, so the after_flush "
            "listener never sees it — it must bump the epoch itself or a cached review count "
            "survives the delete it should have retired."
        )

    async def test_remove_image_tags_bumps_the_write_epoch(self, db_client):
        image_uuid = await _image_uuid(db_client, "b.png")
        tag_uuid = await _tag_uuid(db_client, "Reviewed")
        await db_client.add_image_tags(image_uuid, [tag_uuid])
        epoch_before = db_client.write_epoch

        await db_client.remove_image_tags(image_uuid, [tag_uuid])

        assert db_client.write_epoch > epoch_before, (
            "remove_image_tags is a Core statement, not an ORM delete, so the after_flush "
            "listener never sees it — it must bump the epoch itself or a cached review count "
            "survives the delete it should have retired."
        )
