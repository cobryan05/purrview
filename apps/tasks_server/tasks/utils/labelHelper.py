import asyncio


async def build_label_ancestry_map(labels_api) -> dict[str, set[str]]:
    """Fetch all labels and return {uuid: set_of_ancestor_uuids}.

    If label B has parent A (A is an ancestor of B), then:
        result[B.uuid] = {A.uuid, ...grandparents...}

    Labels with no parent map to an empty set.  Unknown UUIDs (not in the DB)
    are also absent from the map, so callers should use .get(uuid, ()).
    """
    resp = await asyncio.to_thread(labels_api.list_labels)
    parent_map: dict[str, str | None] = {lbl.uuid: lbl.parent_uuid for lbl in (resp.labels or [])}

    ancestry: dict[str, set[str]] = {}

    def _get_ancestors(uuid: str) -> set[str]:
        if uuid in ancestry:
            return ancestry[uuid]
        p = parent_map.get(uuid)
        if p is None:
            ancestry[uuid] = set()
        else:
            ancestry[uuid] = set()  # cycle guard: if uuid is re-encountered before we finish, return empty
            ancestry[uuid] = {p} | _get_ancestors(p)
        return ancestry[uuid]

    for uuid in parent_map:
        _get_ancestors(uuid)

    return ancestry
