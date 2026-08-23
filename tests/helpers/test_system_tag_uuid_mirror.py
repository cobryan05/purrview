"""System tag UUIDs are declared twice, in two languages, with nothing sharing them.

`apps/helpers/db/types.py` seeds the rows; `static/js/shared/api/review.js` builds the
filters that look them up. There is no build step between them, so a divergence is
silent in the worst possible way: the tag exists, the filter runs, and it matches
nothing. The user sees "no results" and concludes the feature does not work.

Per the repo's rule for cross-language mirrors, this asserts the two sides *equal each
other* and never pins a literal — a UUID may legitimately change, but only on both
sides at once.
"""

import re
from pathlib import Path

from apps.helpers.db import types as db_types

_REVIEW_JS = (
    Path(__file__).resolve().parents[2]
    / "apps"
    / "image_tagging_server"
    / "static"
    / "js"
    / "shared"
    / "api"
    / "review.js"
)

_JS_CONST = re.compile(r"^export const ([A-Z_]*TAG_UUID)\s*=\s*'([0-9a-f-]+)'", re.MULTILINE)


def _js_tag_uuids() -> dict:
    return dict(_JS_CONST.findall(_REVIEW_JS.read_text()))


def _python_tag_uuids() -> dict:
    return {
        name: value for name, value in vars(db_types).items() if name.endswith("_TAG_UUID") and isinstance(value, str)
    }


class TestSystemTagUuidMirror:
    def test_every_uuid_the_frontend_names_matches_the_backend(self):
        """A typo on either side produces a filter that quietly matches nothing."""
        js = _js_tag_uuids()
        python = _python_tag_uuids()
        assert js, "no tag UUID constants found in review.js — the parse has gone stale"

        for name, value in js.items():
            assert name in python, f"{name} exists in review.js but not in apps/helpers/db/types.py"
            assert python[name] == value, f"{name} disagrees: python={python[name]} js={value}"

    def test_the_frontend_knows_about_every_backend_tag(self):
        """The direction that catches a *new* tag. Adding one to the seed list without
        mirroring it is how a feature ships with a filter that can never be built —
        which is exactly what "harvested frames vanish" looked like."""
        js = _js_tag_uuids()
        python = _python_tag_uuids()

        missing = set(python) - set(js)
        assert not missing, f"declared in Python but not mirrored into review.js: {sorted(missing)}"

    def test_no_two_tags_share_a_uuid(self):
        """Two names on one row means the second tag silently *is* the first."""
        python = _python_tag_uuids()
        assert len(set(python.values())) == len(python), python
