"""Rows written before a column existed must still read back.

A migration that adds a nullable column leaves every pre-existing row holding SQL NULL,
and a read model that types the field as non-optional then fails validation on all of
them. The endpoint catches the error and returns `recording: null`, which the events
modal cannot tell apart from a deleted file — so the whole of an install's clip history
reported itself as "no longer on disk" while every mp4 sat untouched on the volume.

Nothing caught this in the ordinary suite because tests build rows through the model,
where the field defaults to `{}`. Only a row that reached the table without going
through Python has the NULL, which is exactly what a migration produces.
"""

import pytest
from sqlalchemy import text

from apps.helpers.db.types import Recording, RecordingRead


async def _null_out(db_client, uuid: str, column: str) -> None:
    """Set a column to SQL NULL behind the ORM's back, as a migration would."""
    async with db_client._async_session_maker() as session:
        await session.exec(text(f"UPDATE recordings SET {column} = NULL WHERE uuid = :u").bindparams(u=uuid))
        await session.commit()


class TestARecordingWithNoMetadata:
    async def test_the_read_model_accepts_a_null(self):
        """The narrowest statement of the bug: no database, no endpoint."""
        assert RecordingRead.model_validate(Recording(metadata_json=None)).metadata_json == {}

    async def test_it_still_reads_back_through_the_db_client(self, db_client):
        recording = await db_client.create_recording(Recording(file_path="/tmp/a.mp4", status="complete"))
        await _null_out(db_client, recording.uuid, "metadata_json")

        fetched = await db_client.get_recording(recording.uuid)

        assert fetched is not None, "a row predating metadata_json must not read back as missing"
        assert fetched.metadata_json == {}
        assert fetched.file_path == "/tmp/a.mp4"

    async def test_the_endpoint_still_serves_it(self, http_client, db_client):
        """What the events modal actually calls. A failure here renders as "the clip for
        this event is no longer on disk" over a file that exists."""
        recording = await db_client.create_recording(Recording(file_path="/tmp/a.mp4", status="complete"))
        await _null_out(db_client, recording.uuid, "metadata_json")

        body = (await http_client.get(f"/api/recordings/{recording.uuid}")).json()

        assert body["status"] == "success", body.get("message")
        assert body["recording"]["filePath"] == "/tmp/a.mp4"

    async def test_it_still_reaches_the_event_that_owns_it(self, http_client, db_client):
        """The card badges off this list, so a NULL must not cost an event its badge."""
        from apps.helpers.db.types import WatchEvent, WatchProfile

        profile = await db_client.create_watch_profile(WatchProfile(name="House cams"))
        event = await db_client.create_watch_event(WatchEvent(profile_uuid=profile.uuid))
        recording = await db_client.create_recording(
            Recording(event_uuid=event.uuid, file_path="/tmp/a.mp4", status="complete")
        )
        await _null_out(db_client, recording.uuid, "metadata_json")

        listed = (await http_client.get("/api/watch-events")).json()

        assert [r["kind"] for r in listed["events"][0]["recordings"]] == ["clip"]

    async def test_the_badge_survived_even_while_the_modal_did_not(self, http_client, db_client):
        """Documents why this presented so confusingly: the events list builds its
        summaries from selected columns and never touches `metadata_json`, so the card
        kept its video badge while opening it reported the file gone. Anyone debugging
        from the card alone would have concluded the file really had been deleted.

        Kept as a test because that asymmetry is a property worth not losing: the list
        must stay cheap enough not to hydrate the full row per card.
        """
        from apps.helpers.db.types import WatchEvent, WatchProfile

        profile = await db_client.create_watch_profile(WatchProfile(name="Hall cams"))
        event = await db_client.create_watch_event(WatchEvent(profile_uuid=profile.uuid))
        await db_client.create_recording(Recording(event_uuid=event.uuid, file_path="/tmp/a.mp4", status="complete"))
        listed = (await http_client.get("/api/watch-events")).json()
        assert listed["events"][0]["recordings"][0]["kind"] == "clip"
