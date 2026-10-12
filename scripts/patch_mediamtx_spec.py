#!/usr/bin/env python3
"""Make MediaMTX's OpenAPI spec generatable before the client is built from it.

``PathTrackCodecProps`` is a ``oneOf`` over nine codec variants with no
discriminator, and ``PathTrackCodecPropsH264``/``...H265`` are structurally
identical — same four optional properties, no ``required``, no
``additionalProperties: false``. Any H264 payload therefore validates against
both, and openapi-generator's Python client raises "Multiple matches found when
deserializing" for every path that has a track. The discriminating value lives
on ``PathTrack.codec``, a sibling of ``codecProps``, which an OpenAPI
discriminator cannot reach, so the spec cannot be fixed from inside that schema.

Collapsing it to a permissive object keeps the field readable as a plain dict.
Nothing in this tree reads codec properties; ``Path.tracks`` (codec names) is
what the status panel shows, and it stays a string enum. ``tracks2`` is kept
rather than deleted because ``tracks`` is already marked deprecated upstream —
when it goes, the move to ``tracks2[].codec`` should not need spec surgery.

PATCHING SOMEONE ELSE'S SPEC IS COUPLED TO THEIR VERSION. This is written
against the schema as MEDIAMTX_VERSION in the Dockerfile pins it, and upstream
owes us nothing here: the union may gain a discriminator, the variants may become
disjoint, the schema may be renamed or the whole shape may move again (the spec
file itself moved from apidocs/ to api/ in 1.19.2). Any of those makes this patch
wrong, unnecessary, or both.

So on every MediaMTX bump, treat this file as part of the bump:

1. Re-read the new ``PathTrackCodecProps``. If the union is discriminated or its
   variants are disjoint, DELETE this script and its Dockerfile step — the
   workaround has outlived the bug.
2. If it still needs patching, re-check that the pattern matches the new shape.
   A mismatch fails the build loudly (see ``patch``), which is the intended
   behaviour — do not loosen the pattern to make the build pass without first
   confirming what changed upstream.
3. Validate against the live relay, not the test suite. Nothing downstream of
   this is covered: generated-client deserialization is mocked everywhere in
   ``tests/stream_server/``, so a client this patch breaks still goes green.

Tracking: the union needs a discriminator upstream; until then this stays.
"""

import re
import sys
from pathlib import Path

SCHEMA = "PathTrackCodecProps"

# The schema as a block: its name at four-space indent, then a `oneOf:` whose
# every entry is a $ref. Anchored to the block so a sibling schema that happens
# to use oneOf is never touched.
_ONE_OF_BLOCK = re.compile(
    rf"^(?P<indent> {{4}}){SCHEMA}:\n" rf"(?P=indent)  oneOf:\n" rf"(?:(?P=indent)    - \$ref: .*\n)+",
    re.MULTILINE,
)


def patch(text: str) -> str:
    """Replace the undiscriminated union with an object that accepts anything.

    Raises if the union is not found, so a spec whose shape moved fails the build
    here rather than producing a client that breaks on first use.
    """
    patched, count = _ONE_OF_BLOCK.subn(
        lambda m: f"{m.group('indent')}{SCHEMA}:\n"
        f"{m.group('indent')}  type: object\n"
        f"{m.group('indent')}  additionalProperties: true\n",
        text,
    )
    if count != 1:
        raise SystemExit(
            f"{SCHEMA}: expected exactly one undiscriminated oneOf block, found {count}. "
            f"Upstream's spec has changed shape — re-check whether this patch is still "
            f"needed before adjusting the pattern."
        )
    return patched


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(f"usage: {argv[0]} <path-to-openapi.yaml>")
    spec = Path(argv[1])
    original = spec.read_text()
    spec.write_text(patch(original))
    print(f"{spec}: collapsed {SCHEMA} oneOf to a permissive object")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
