import pytest


async def test_list_labels_empty(http_client):
    r = await http_client.get("/api/labels")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["labels"] == []


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
    assert len(labels) == 1
    assert labels[0]["labelType"] == "standard"


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
