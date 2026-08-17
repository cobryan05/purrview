"""
Box colours in the NVR overlay (app/nvr/overlay.js).

A watcher's preview used to stroke every track the same green, which made the overlay
unable to answer the first question anyone asks of a box — *what* is that? The boxes now
carry the label's own colour, and these tests pin the two halves of that:

  - the label's colour wins when the watcher knows one, and a missing or malformed one
    falls back to the semantic colour rather than to whatever the previous box was
    stroked in (canvas ignores an invalid `strokeStyle` silently, so the fallback has to
    be explicit or boxes inherit each other's colours with no error anywhere);
  - the event/anchor status the diagnostic view depends on survives the change, as a
    glow around a label-coloured box rather than as the box's own colour.

Run with:
    pytest playwright_tests/test_nvr_overlay_colours.py -p no:asyncio -v
"""

from playwright.sync_api import Page

_IMPORT = "const O = await import('/app-static/js/app/nvr/overlay.js');"

# A canvas big enough that a box has an interior, small enough to read pixels from.
_CANVAS = """
    const canvas = document.createElement('canvas');
    canvas.width = 40;
    canvas.height = 40;
    const ctx = canvas.getContext('2d');
    const pixel = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data);
"""


class TestBoxColour:
    def test_label_colour_wins_over_the_fallback(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            return O.boxColour({{ labelColor: '#ff00ff' }}, O.TRACK_COLOUR);
        }}""")
        assert result == "#ff00ff"

    def test_a_missing_label_colour_falls_back(self, shared_page: Page):
        """A detection whose class has no label mapping still has to be visible."""
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            return [
                O.boxColour({{ labelColor: null }}, O.DETECTION_COLOUR),
                O.boxColour({{}}, O.DETECTION_COLOUR),
                O.boxColour(null, O.DETECTION_COLOUR),
            ];
        }}""")
        assert result == ["#ffb347", "#ffb347", "#ffb347"]

    def test_a_malformed_label_colour_falls_back(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            return [
                O.boxColour({{ labelColor: '' }}, O.TRACK_COLOUR),
                O.boxColour({{ labelColor: '   ' }}, O.TRACK_COLOUR),
                O.boxColour({{ labelColor: 'not a colour' }}, O.TRACK_COLOUR),
                O.boxColour({{ labelColor: '#12' }}, O.TRACK_COLOUR),
                O.boxColour({{ labelColor: 42 }}, O.TRACK_COLOUR),
            ];
        }}""")
        assert result == ["#4ade80"] * 5

    def test_recognised_colour_notations(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            return ['#abc', '#aabbcc', '#aabbccdd', 'rgb(1, 2, 3)', 'rgba(1,2,3,0.5)',
                    'hsl(10, 20%, 30%)', 'tomato', ' #ff0000 ']
                .map(c => O.isDrawableColour(c));
        }}""")
        assert result == [True] * 8


class TestTrackStatusColour:
    def test_status_colour_still_ranks_event_over_anchor(self, shared_page: Page):
        """The status colour is now the glow, but its precedence is unchanged."""
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            return [
                O.trackColour({{ eventRules: ['rule'], isAnchor: true }}),
                O.trackColour({{ isAnchor: true }}),
                O.trackColour({{}}),
            ];
        }}""")
        assert result == ["#f87171", "#60a5fa", "#4ade80"]


class TestDrawnPixels:
    def test_the_stroke_is_the_colour_passed(self, shared_page: Page):
        """Magenta in, magenta on the canvas — not the green the boxes used to be."""
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            {_CANVAS}
            O.drawBox(ctx, [5, 5, 35, 35], '#ff00ff', null, {{ lineWidth: 4 }});
            return pixel(5, 5);
        }}""")
        red, green, blue, alpha = result
        assert alpha > 200
        assert red > 200 and blue > 200
        assert green < 60

    def test_a_glow_does_not_repaint_the_stroke(self, shared_page: Page):
        """The halo carries the status; the box itself must stay the label's colour."""
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            {_CANVAS}
            O.drawBox(ctx, [10, 10, 30, 30], '#ff00ff', null, {{
                lineWidth: 4, emphasis: true, glow: '#f87171',
            }});
            return pixel(10, 10);
        }}""")
        red, green, blue, _alpha = result
        # The red glow bleeds outward, so the *stroke* pixel is what is checked: still
        # magenta (blue channel high) rather than the glow's red.
        assert red > 150 and blue > 150
        assert green < 90

    def test_a_plain_track_gets_no_glow(self, shared_page: Page):
        """Haloing every box would say nothing — outside the stroke stays transparent."""
        result = shared_page.evaluate(f"""async () => {{
            {_IMPORT}
            {_CANVAS}
            O.drawBox(ctx, [12, 12, 28, 28], '#ff00ff', null, {{ lineWidth: 2 }});
            // Well clear of the 2px stroke centred on the box edge.
            return pixel(6, 6);
        }}""")
        assert result[3] == 0
