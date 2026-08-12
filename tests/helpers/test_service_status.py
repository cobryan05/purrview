"""The shared shape every service's /status details are built from."""

from apps.helpers.serviceStatus import fmt_duration, row, section


class TestRow:
    def test_value_is_stringified(self):
        assert row("Pins", 3) == {"label": "Pins", "value": "3"}

    def test_none_becomes_empty_rather_than_the_word_none(self):
        assert row("Source", None)["value"] == ""


class TestSection:
    def test_rows_are_materialised_from_any_iterable(self):
        sec = section("Streams", (row(n, n) for n in ("a", "b")))
        assert [r["label"] for r in sec["rows"]] == ["a", "b"]

    def test_empty_key_is_omitted_when_not_supplied(self):
        """The frontend falls back to its own placeholder; an absent key and an
        empty string would render differently."""
        assert "empty" not in section("Streams", [])

    def test_empty_key_is_kept_when_supplied(self):
        assert section("Streams", [], empty="No streams")["empty"] == "No streams"


class TestFmtDuration:
    def test_seconds(self):
        assert fmt_duration(42) == "42s"

    def test_minutes_pad_their_seconds(self):
        assert fmt_duration(90) == "1m 30s"
        assert fmt_duration(305) == "5m 05s"

    def test_hours(self):
        assert fmt_duration(7380) == "2h 03m"

    def test_negative_clamps_to_zero(self):
        """An already-expired pin reports 0s, not a negative countdown."""
        assert fmt_duration(-5) == "0s"

    def test_none_is_an_em_dash(self):
        assert fmt_duration(None) == "—"
