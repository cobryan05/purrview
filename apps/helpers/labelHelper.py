import asyncio

from .suppressionGroups import suppression_group_for


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


async def build_label_metadata_map(labels_api) -> dict[str, dict[str, "str | None"]]:
    """Fetch all labels and return {uuid: {"text": name, "color": color}}.

    Used to resolve a bare label_uuid into display-ready name/color once (e.g. at
    task start), so hot paths that only carry a label_uuid — like a saved
    detection's bbox — can build a UI-ready label descriptor without any further
    DB access. Unknown UUIDs are absent from the map; callers should use
    .get(uuid, {}).
    """
    resp = await asyncio.to_thread(labels_api.list_labels)
    return {lbl.uuid: {"text": lbl.name, "color": lbl.color} for lbl in (resp.labels or [])}


async def build_label_group_map(labels_api) -> dict[str, str]:
    """Fetch all labels and return {uuid: exclusive_variants group key}.

    A label that is not part of any group maps to itself, so a consumer can compare two
    group keys for equality without a second "was this grouped at all" branch.

    The grouping decision itself is `suppression_group_for`'s and is not re-derived
    here — it is already the single source of truth for NMS, the backfill script and the
    inference server's preview endpoint, and a fourth opinion about which labels are
    alternate guesses for one object is exactly the drift that module exists to prevent.
    """
    resp = await asyncio.to_thread(labels_api.list_labels)
    labels = resp.labels or []
    parent_of = {lbl.uuid: lbl.parent_uuid for lbl in labels}
    exclusive_of = {lbl.uuid: bool(getattr(lbl, "exclusive_variants", False)) for lbl in labels}
    threshold_of = {lbl.uuid: getattr(lbl, "exclusive_variants_iou_threshold", None) for lbl in labels}
    return {lbl.uuid: suppression_group_for(lbl.uuid, parent_of, exclusive_of, threshold_of)[0] for lbl in labels}
