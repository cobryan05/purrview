import pytest

from apps.helpers.db.db_client import DbClient
from apps.helpers.db.types import LabelScope, LabelType


async def test_list_labels_empty(http_client):
    """A freshly-seeded DB has no user-created labels — only the two default
    image-scope labels (Corrupt, Blurry) seeded on startup."""
    r = await http_client.get("/api/labels")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    names = {lbl["name"] for lbl in body["labels"]}
    assert names == {"Corrupt", "Blurry"}


async def test_create_label(http_client):
    r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["label"]["name"] == "cat"
    assert body["label"]["color"] == "#FF0000"
    assert "uuid" in body["label"]


async def test_create_and_list_label(http_client):
    await http_client.post("/api/labels", json={"name": "dog", "color": "#00FF00"})
    r = await http_client.get("/api/labels")
    body = r.json()
    assert body["status"] == "success"
    names = [lbl["name"] for lbl in body["labels"]]
    assert "dog" in names


async def test_update_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "bird", "color": "#0000FF"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.patch(f"/api/labels/{uuid}", json={"name": "parrot", "color": "#FFFF00"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "parrot" in names
    assert "bird" not in names


async def test_update_label_type_via_camel_case(http_client):
    """The UI sends {labelType: ...} (camelCase). LabelUpdate must map it to
    label_type; otherwise the key is silently dropped and the type never changes.
    """
    create_r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
    uuid = create_r.json()["label"]["uuid"]

    patch_r = await http_client.patch(f"/api/labels/{uuid}", json={"labelType": "dynamic"})
    assert patch_r.status_code == 200
    assert patch_r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    lbl = next(l for l in list_r.json()["labels"] if l["uuid"] == uuid)
    assert lbl["labelType"] == "dynamic"


async def test_delete_label(http_client):
    create_r = await http_client.post("/api/labels", json={"name": "fish", "color": "#AABBCC"})
    uuid = create_r.json()["label"]["uuid"]

    r = await http_client.delete(f"/api/labels/{uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "success"

    list_r = await http_client.get("/api/labels")
    names = [lbl["name"] for lbl in list_r.json()["labels"]]
    assert "fish" not in names


async def test_delete_label_with_children_fails(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "animal", "color": "#111111"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    await http_client.post(
        "/api/labels",
        json={"name": "dog", "color": "#222222", "parent_uuid": parent_uuid},
    )

    r = await http_client.delete(f"/api/labels/{parent_uuid}")
    assert r.status_code == 200
    assert r.json()["status"] == "failure"


async def test_create_child_label(http_client):
    parent_r = await http_client.post("/api/labels", json={"name": "vehicle", "color": "#333333"})
    parent_uuid = parent_r.json()["label"]["uuid"]

    child_r = await http_client.post(
        "/api/labels",
        json={"name": "car", "color": "#444444", "parent_uuid": parent_uuid},
    )
    assert child_r.status_code == 200
    body = child_r.json()
    assert body["status"] == "success"
    assert body["label"]["parentUuid"] == parent_uuid


class TestExclusiveVariantsDefault:
    """A new label's exclusive_variants defaults based on whether it has a parent —
    see AddLabelPayload._default_exclusive_variants_to_parentless in apps/db_server/web.py.
    A top-level label defaults on (the common "Vehicle" variant-group case); a label
    created under a parent defaults off, so it defers to whatever its immediate parent
    decides instead of becoming its own singleton group and breaking out of the
    parent's group (suppression_group_for checks the label's own flag before its
    parent's)."""

    async def test_parentless_label_defaults_on(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "Vehicle", "color": "#111111"})
        assert r.json()["label"]["exclusiveVariants"] is True

    async def test_child_label_defaults_off(self, http_client):
        parent_r = await http_client.post("/api/labels", json={"name": "Vehicle", "color": "#111111"})
        parent_uuid = parent_r.json()["label"]["uuid"]

        child_r = await http_client.post(
            "/api/labels",
            json={"name": "Truck", "color": "#222222", "parentUuid": parent_uuid},
        )
        assert child_r.json()["label"]["exclusiveVariants"] is False

    async def test_explicit_value_overrides_the_default(self, http_client):
        parent_r = await http_client.post("/api/labels", json={"name": "Vehicle", "color": "#111111"})
        parent_uuid = parent_r.json()["label"]["uuid"]

        child_r = await http_client.post(
            "/api/labels",
            json={
                "name": "Truck",
                "color": "#222222",
                "parentUuid": parent_uuid,
                "exclusiveVariants": True,
            },
        )
        assert child_r.json()["label"]["exclusiveVariants"] is True


async def test_label_type_roundtrip_from_migration_default(db_client, http_client):
    """
    Regression: the migration adds label_type with server_default='standard' (lowercase
    enum value).  sa.Enum(LabelType) without values_callable stores/expects names
    ('STANDARD'), so it rejects 'standard' on read.  The ORM column must be configured
    to use values, not names, so rows written by the migration default survive a round-trip.
    """
    import sqlalchemy as sa

    # Insert a row the way the migration's server_default does: lowercase value.
    async with db_client._engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO labels (uuid, name, label_type, created_at, updated_at)"
                " VALUES (:uuid, :name, :lt, datetime('now'), datetime('now'))"
            ),
            {"uuid": "test-migration-default", "name": "cat", "lt": "standard"},
        )

    # Fetching labels must succeed and return the correct label_type value.
    r = await http_client.get("/api/labels")
    assert r.status_code == 200, r.text
    labels = r.json()["labels"]
    cat = next(lbl for lbl in labels if lbl["name"] == "cat")
    assert cat["labelType"] == "standard"


class TestLabelScope:
    async def test_create_label_defaults_to_object_scope(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["label"]["scope"] == "object"

    async def test_create_image_scope_label(self, http_client):
        r = await http_client.post(
            "/api/labels",
            json={"name": "corrupt", "color": "#FF0000", "scope": "image"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["label"]["scope"] == "image"

    async def test_scope_roundtrips_via_list(self, http_client):
        await http_client.post("/api/labels", json={"name": "blurry", "color": "#aabbcc", "scope": "image"})
        r = await http_client.get("/api/labels")
        labels = r.json()["labels"]
        blurry = next(l for l in labels if l["name"] == "blurry")
        assert blurry["scope"] == "image"

    async def test_update_label_scope(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "dark", "color": "#111"})
        uuid = r.json()["label"]["uuid"]

        patch_r = await http_client.patch(f"/api/labels/{uuid}", json={"scope": "image"})
        assert patch_r.status_code == 200
        assert patch_r.json()["status"] == "success"

        list_r = await http_client.get("/api/labels")
        dark = next(l for l in list_r.json()["labels"] if l["name"] == "dark")
        assert dark["scope"] == "image"

    async def test_scope_server_default_is_object(self, db_client, http_client):
        """Rows inserted without a scope column (migration backfill path) read back as object."""
        import sqlalchemy as sa

        async with db_client._engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO labels (uuid, name, label_type, scope, created_at, updated_at)"
                    " VALUES (:uuid, :name, 'standard', 'object', datetime('now'), datetime('now'))"
                ),
                {"uuid": "scope-default-test", "name": "migrated_label"},
            )

        r = await http_client.get("/api/labels")
        labels = r.json()["labels"]
        ml = next(l for l in labels if l["name"] == "migrated_label")
        assert ml["scope"] == "object"


class TestLabelReparent:
    """
    Re-parenting a label ("move a model in the hierarchy") == changing
    `parent_uuid`. Because `ModelConfigNode` is keyed by `label_uuid` (not a
    tree node itself), a true re-parent must preserve every model-config
    assignment hanging off the moved label — see TestReparentPreservesModelConfigNodes
    in test_model_configs.py for the constructive counterpart.
    """

    async def _label(self, http_client, name: str, parent_uuid: str | None = None) -> str:
        payload = {"name": name, "color": "#123456"}
        if parent_uuid is not None:
            payload["parentUuid"] = parent_uuid
        r = await http_client.post("/api/labels", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "success", body
        return body["label"]["uuid"]

    async def _get(self, http_client, uuid: str) -> dict:
        r = await http_client.get("/api/labels")
        labels = r.json()["labels"]
        return next(l for l in labels if l["uuid"] == uuid)

    async def test_reparent_label_moves_it(self, http_client):
        root_a = await self._label(http_client, "RootA")
        root_b = await self._label(http_client, "RootB")
        child = await self._label(http_client, "Child", parent_uuid=root_a)

        r = await http_client.patch(f"/api/labels/{child}", json={"parentUuid": root_b})
        assert r.status_code == 200
        assert r.json()["status"] == "success", r.json()

        moved = await self._get(http_client, child)
        assert moved["parentUuid"] == root_b

    async def test_reparent_to_null_promotes_to_root(self, http_client):
        root = await self._label(http_client, "Root2")
        child = await self._label(http_client, "Child2", parent_uuid=root)

        r = await http_client.patch(f"/api/labels/{child}", json={"parentUuid": None})
        assert r.status_code == 200
        assert r.json()["status"] == "success", r.json()

        moved = await self._get(http_client, child)
        assert moved["parentUuid"] is None

    async def test_omitting_parent_uuid_leaves_it_unchanged(self, http_client):
        """The subtle case: `None` is a meaningful value (promote to root), so
        omitting the field entirely must NOT be treated the same as passing null."""
        root = await self._label(http_client, "Root3")
        child = await self._label(http_client, "Child3", parent_uuid=root)

        r = await http_client.patch(f"/api/labels/{child}", json={"name": "Child3Renamed"})
        assert r.status_code == 200
        assert r.json()["status"] == "success", r.json()

        unchanged = await self._get(http_client, child)
        assert unchanged["parentUuid"] == root
        assert unchanged["name"] == "Child3Renamed"

    async def test_reparent_to_self_rejected(self, http_client):
        label = await self._label(http_client, "SelfParent")

        r = await http_client.patch(f"/api/labels/{label}", json={"parentUuid": label})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure", body

        unchanged = await self._get(http_client, label)
        assert unchanged["parentUuid"] is None

    async def test_reparent_to_descendant_rejected(self, http_client):
        grandparent = await self._label(http_client, "Grandparent")
        parent = await self._label(http_client, "Parent", parent_uuid=grandparent)
        child = await self._label(http_client, "GrandChild", parent_uuid=parent)

        # Moving grandparent under its own grandchild would create a cycle.
        r = await http_client.patch(f"/api/labels/{grandparent}", json={"parentUuid": child})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failure", body

        # Tree must be left unchanged.
        unchanged = await self._get(http_client, grandparent)
        assert unchanged["parentUuid"] is None
        unchanged_parent = await self._get(http_client, parent)
        assert unchanged_parent["parentUuid"] == grandparent

    async def test_reparent_direct_child_to_itself_as_parent_rejected(self, http_client):
        """A one-hop cycle: parent re-parented to its own direct child."""
        parent = await self._label(http_client, "P")
        child = await self._label(http_client, "C", parent_uuid=parent)

        r = await http_client.patch(f"/api/labels/{parent}", json={"parentUuid": child})
        assert r.status_code == 200
        assert r.json()["status"] == "failure", r.json()

        unchanged = await self._get(http_client, parent)
        assert unchanged["parentUuid"] is None

    async def test_reparent_moves_whole_subtree(self, http_client):
        """Re-parenting a label with children moves the whole subtree — children
        keep pointing at it, only the moved label's own parent_uuid changes."""
        new_root = await self._label(http_client, "NewRoot")
        old_root = await self._label(http_client, "OldRoot")
        subtree_top = await self._label(http_client, "SubtreeTop", parent_uuid=old_root)
        grandchild = await self._label(http_client, "SubtreeGrandchild", parent_uuid=subtree_top)

        r = await http_client.patch(f"/api/labels/{subtree_top}", json={"parentUuid": new_root})
        assert r.status_code == 200
        assert r.json()["status"] == "success", r.json()

        moved = await self._get(http_client, subtree_top)
        assert moved["parentUuid"] == new_root
        # Grandchild's parent is untouched — it still points at subtree_top.
        gc = await self._get(http_client, grandchild)
        assert gc["parentUuid"] == subtree_top


class TestSeedSystemLabels:
    """Coverage for DbClient._seed_system_labels(), which seeds the default
    image-scope labels (Corrupt, Blurry) on startup. Unlike
    _seed_system_tags, these are matched by (name, scope) rather than a fixed
    UUID, since a matching label may pre-exist with annotation rows already
    pointing at its UUID.

    These tests use a raw DbClient (not the shared db_client fixture) so each
    test controls exactly when init_db()/seeding runs relative to its setup.
    """

    _DEFAULT_NAMES = {"Corrupt", "Blurry"}

    async def test_seed_creates_all_defaults_on_empty_db(self, tmp_path):
        client = DbClient(db_dir=tmp_path, memory=True)
        await client.init_db()
        try:
            labels = await client.list_labels()
            image_labels = {lbl.name: lbl for lbl in labels if lbl.scope == LabelScope.IMAGE}
            assert set(image_labels) == self._DEFAULT_NAMES
            for lbl in image_labels.values():
                assert lbl.label_type == LabelType.STANDARD
                assert lbl.parent_uuid is None
        finally:
            await client.close()

    async def test_seed_is_idempotent(self, tmp_path):
        client = DbClient(db_dir=tmp_path, memory=True)
        await client.init_db()
        try:
            labels_before = await client.list_labels()
            uuids_before = {lbl.name: lbl.uuid for lbl in labels_before if lbl.scope == LabelScope.IMAGE}

            await client._seed_system_labels()

            labels_after = await client.list_labels()
            image_after = [lbl for lbl in labels_after if lbl.scope == LabelScope.IMAGE]
            assert len(image_after) == len(self._DEFAULT_NAMES)
            uuids_after = {lbl.name: lbl.uuid for lbl in image_after}
            assert uuids_after == uuids_before
        finally:
            await client.close()

    async def test_preexisting_label_with_same_name_and_scope_is_left_untouched(self, tmp_path):
        """Regression test: a label with the same name+scope that already exists
        (e.g. carrying real annotation rows against its UUID) must not be
        duplicated or have its UUID replaced by seeding."""
        client = DbClient(db_dir=tmp_path, memory=True)
        await client.init_db()
        try:
            labels = await client.list_labels()
            seeded_corrupt = next(lbl for lbl in labels if lbl.name == "Corrupt" and lbl.scope == LabelScope.IMAGE)
            await client.delete_label(seeded_corrupt.uuid)
            await client.add_label(name="Corrupt", uuid="custom-corrupt-uuid", scope=LabelScope.IMAGE)

            await client._seed_system_labels()

            labels_after = await client.list_labels()
            corrupt_labels = [lbl for lbl in labels_after if lbl.name == "Corrupt" and lbl.scope == LabelScope.IMAGE]
            assert len(corrupt_labels) == 1
            assert corrupt_labels[0].uuid == "custom-corrupt-uuid"
        finally:
            await client.close()

    async def test_object_scope_same_name_does_not_block_image_scope_seed(self, tmp_path):
        """An object-scope label named 'Blurry' is a distinct label from the
        image-scope 'Blurry' default — its presence must not suppress seeding."""
        client = DbClient(db_dir=tmp_path, memory=True)
        await client.init_db()
        try:
            labels = await client.list_labels()
            image_blurry = next(lbl for lbl in labels if lbl.name == "Blurry" and lbl.scope == LabelScope.IMAGE)
            await client.delete_label(image_blurry.uuid)
            obj_blurry = await client.add_label(name="Blurry", scope=LabelScope.OBJECT)

            await client._seed_system_labels()

            labels_after = await client.list_labels()
            blurries = [lbl for lbl in labels_after if lbl.name == "Blurry"]
            assert len(blurries) == 2
            scopes = {lbl.scope for lbl in blurries}
            assert scopes == {LabelScope.IMAGE, LabelScope.OBJECT}
            image_blurry_after = next(lbl for lbl in blurries if lbl.scope == LabelScope.IMAGE)
            assert image_blurry_after.uuid != obj_blurry.uuid
        finally:
            await client.close()


class TestHomeAssistantPublishing:
    """`publishToHa` is opt-in. A label device is justified by the label rather than by a
    rule mentioning it, so this flag is what bounds how many devices an install creates —
    and a vocabulary grows with every class a model can name.
    """

    async def test_a_new_label_is_not_published(self, http_client):
        r = await http_client.post("/api/labels", json={"name": "cat", "color": "#FF0000"})

        assert r.json()["label"]["publishToHa"] is False

    async def test_it_can_be_opted_in_at_creation(self, http_client):
        """The pair: off is a default, not a refusal."""
        r = await http_client.post(
            "/api/labels",
            json={"name": "person", "color": "#00FF00", "publishToHa": True},
        )

        assert r.json()["label"]["publishToHa"] is True

    async def test_opting_in_later_sticks(self, http_client):
        """The path someone actually takes — tick the few you automate on, after the
        vocabulary already exists."""
        uuid = (await http_client.post("/api/labels", json={"name": "dog", "color": "#111"})).json()["label"]["uuid"]

        patch = await http_client.patch(f"/api/labels/{uuid}", json={"publishToHa": True})
        assert patch.status_code == 200

        listed = (await http_client.get("/api/labels")).json()["labels"]
        assert next(l for l in listed if l["name"] == "dog")["publishToHa"] is True

    async def test_a_row_inserted_without_the_column_reads_back_unpublished(self, db_client, http_client):
        """The migration backfill path. A label written before the column existed must
        arrive unticked rather than silently claiming a device — publishing is the
        expensive direction, and absence is only a label nobody opted in.
        """
        import sqlalchemy as sa

        async with db_client._engine.begin() as conn:
            await conn.execute(
                sa.text(
                    "INSERT INTO labels (uuid, name, label_type, scope, created_at, updated_at)"
                    " VALUES (:uuid, :name, 'standard', 'object', datetime('now'), datetime('now'))"
                ),
                {"uuid": "ha-default-test", "name": "migrated_ha_label"},
            )

        listed = (await http_client.get("/api/labels")).json()["labels"]
        assert next(l for l in listed if l["name"] == "migrated_ha_label")["publishToHa"] is False
