"""Tests for the MediaMTX spec patch the relay client is generated from.

The patch runs at image build time, and nothing downstream of it is covered by
this suite: `tests/stream_server/` mocks `ConfigurationApi`/`PathsApi`, so a
generated client that cannot deserialize a single real response still passes
every other test here. That is exactly how an unusable client shipped once. These
tests cover the one piece of logic that can be checked without a build — which
schema the patch rewrites, and which it must leave alone.
"""

import pytest

from scripts.patch_mediamtx_spec import patch

# Shaped like the real spec: the target union, a sibling union that must survive,
# and one variant definition the union referenced.
SPEC = """\
components:
  schemas:
    PathTrackCodecProps:
      oneOf:
        - $ref: "#/components/schemas/PathTrackCodecPropsAV1"
        - $ref: "#/components/schemas/PathTrackCodecPropsH264"
        - $ref: "#/components/schemas/PathTrackCodecPropsH265"

    ForwardDestTypeSpecific:
      oneOf:
        - $ref: "#/components/schemas/ForwardDestTypeSpecificRTMP"
        - $ref: "#/components/schemas/ForwardDestTypeSpecificRTSP"

    PathTrackCodecPropsH264:
      type: object
      properties:
        level:
          type: string
"""


class TestPatchingTheCodecPropsUnion:
    def test_the_union_becomes_an_object_that_accepts_anything(self):
        """H264 and H265 are structurally identical, so the union can never be
        resolved by shape — a permissive object is the only form the generator can
        turn into something that deserializes a real response."""
        out = patch(SPEC)

        assert "    PathTrackCodecProps:\n      type: object\n      additionalProperties: true\n" in out
        assert "PathTrackCodecPropsAV1" not in out.split("ForwardDestTypeSpecific")[0]

    def test_a_sibling_union_is_left_alone(self):
        """The pattern is anchored to one schema. A patch that collapsed every
        oneOf would flatten unrelated models and lose their typing silently."""
        out = patch(SPEC)

        assert '        - $ref: "#/components/schemas/ForwardDestTypeSpecificRTMP"' in out
        assert "    ForwardDestTypeSpecific:\n      oneOf:\n" in out

    def test_a_spec_without_the_union_is_an_error_not_a_no_op(self):
        """When upstream gives the union a discriminator the patch stops matching,
        and that has to fail the build loudly: silently passing the spec through is
        how a stale workaround outlives the bug it was written for."""
        with pytest.raises(SystemExit) as exc:
            patch(SPEC.replace("PathTrackCodecProps:\n      oneOf:", "PathTrackCodecProps:\n      type: object"))

        assert "found 0" in str(exc.value)
