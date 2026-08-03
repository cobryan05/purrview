"""Translate the pre-1.4b flat ImageSearchParams kwargs into the terms-shaped
request body PR 1.4b requires.

PR 1.4b deleted ImageSearchParams' flat filter fields (tag_uuids, label_uuids,
image_label_filters, filename_search, date_from, ...) in favour of the
discriminated-union `terms` list. A large number of existing tests build their
request bodies from those flat kwargs; rather than rewrite every call site
(which would also make diffing against pre-1.4b test intent harder), this
module centralises the flat-kwargs -> terms translation so test call sites and
their assertions can stay unchanged.

New tests should just build `terms` directly (see test_filter_terms.py).
"""

from typing import Any, Dict, List, Optional


def legacy_params_to_body(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with legacy flat-filter kwargs replaced by `terms`.

    Non-filter kwargs (show_all_bboxes, require_bboxes, limit, sort_by,
    sort_order, after_uuid, after_sort_value, terms) pass through unchanged.
    """
    params = dict(params)
    terms: List[Dict[str, Any]] = list(params.pop("terms", None) or [])

    tag_uuids = params.pop("tag_uuids", None)
    tag_match_mode = params.pop("tag_match_mode", "any")
    if tag_uuids:
        terms.append({"kind": "bboxTag", "tagUuids": tag_uuids, "matchMode": tag_match_mode})

    label_uuids = params.pop("label_uuids", None)
    label_match_mode = params.pop("label_match_mode", "any")
    label_include_children = params.pop("label_include_children", False)
    if label_uuids:
        terms.append(
            {
                "kind": "objectLabel",
                "labelUuids": label_uuids,
                "matchMode": label_match_mode,
                "rollUp": label_include_children,
            }
        )

    label_uuid = params.pop("label_uuid", None)
    if label_uuid and (not label_uuids or label_uuid not in label_uuids):
        terms.append({"kind": "objectLabel", "labelUuids": [label_uuid]})

    for excluded_uuid in params.pop("label_exclusions", None) or []:
        # roll_up=False: the flat label_exclusions field never child-expanded.
        terms.append({"kind": "objectLabel", "labelUuids": [excluded_uuid], "negate": True, "rollUp": False})

    filename_search = params.pop("filename_search", None)
    if filename_search:
        terms.append({"kind": "filename", "patterns": [filename_search]})
    filename_searches = params.pop("filename_searches", None) or []
    if filename_searches:
        terms.append({"kind": "filename", "patterns": list(filename_searches)})

    date_from = params.pop("date_from", None)
    date_to = params.pop("date_to", None)
    if date_from or date_to:
        term: Dict[str, Any] = {"kind": "date", "field": "created"}
        if date_from:
            term["from"] = date_from
        if date_to:
            term["to"] = date_to
        terms.append(term)

    updated_date_from = params.pop("updated_date_from", None)
    updated_date_to = params.pop("updated_date_to", None)
    if updated_date_from or updated_date_to:
        term = {"kind": "date", "field": "updated"}
        if updated_date_from:
            term["from"] = updated_date_from
        if updated_date_to:
            term["to"] = updated_date_to
        terms.append(term)

    for tf in params.pop("tag_presence_filters", None) or []:
        terms.append(
            {
                "kind": "bboxTag",
                "tagUuids": [tf["tag_uuid"]],
                "negate": bool(tf.get("exclude", False)),
            }
        )

    for f in params.pop("image_label_filters", None) or []:
        state, negate = _image_item_state(f)
        terms.append(
            {
                "kind": "imageLabel",
                "labelUuid": f["label_uuid"],
                "state": state,
                "rollUp": f.get("roll_up", True),
                "negate": negate,
            }
        )

    for f in params.pop("image_tag_filters", None) or []:
        state, negate = _image_item_state(f, default_status="present")
        terms.append(
            {
                "kind": "imageTag",
                "tagUuid": f["tag_uuid"],
                "state": state,
                "negate": negate,
            }
        )

    label_verification_check = params.pop("label_verification_check", None)
    verification_label_uuids = params.pop("verification_label_uuids", None)
    if label_verification_check:
        targets = verification_label_uuids or label_uuids or []
        terms.append(
            {
                "kind": "labelVerification",
                "check": label_verification_check,
                "labelUuids": targets,
            }
        )

    if terms:
        params["terms"] = terms
    return params


def _image_item_state(f: Dict[str, Any], default_status: Optional[str] = None) -> tuple:
    """Map an ImageLabelFilter/ImageTagFilter dict's require_unset/require_verified/
    status/exclude axes onto (ImageLabelState value, negate).

    require_unset and require_verified both ignore `exclude` entirely (matching
    the old flat-field semantics — see test_review.py's
    test_image_label_filter_require_unset_ignores_exclude).
    """
    if f.get("require_unset"):
        return "unreviewed", False
    if f.get("require_verified"):
        return "any_reviewed", False
    status = f.get("status", default_status)
    return status, bool(f.get("exclude", False))
