from apps.helpers.tracking.tracker import Track
from apps.helpers.watch.engine import WatchEngine
from apps.helpers.watch.events import EventPhase
from apps.helpers.watch.rules import RuleSpec

PERSON = "label-person"
CHRIS = "label-chris"
COAT_RACK = (100.0, 100.0, 160.0, 300.0)
DOORWAY = (600.0, 100.0, 680.0, 320.0)
HOUR = 3600.0


def _rule(ignore_after_s=HOUR) -> RuleSpec:
    return RuleSpec(
        uuid="rule-away",
        name="person inside while away",
        subject_a_kind="label",
        subject_a_refs=[PERSON, CHRIS],
        relation="present",
        subject_b_kind="none",
        threshold=0.5,
        min_duration_s=1.0,
        rearm_delay_s=10.0,
        max_duration_s=300.0,
        stationary_ignore_after_s=ignore_after_s,
    )


def _engine(rule: RuleSpec) -> WatchEngine:
    return WatchEngine(
        rules=[rule],
        source_uuid="cam",
        frame_width=1000.0,
        frame_height=1000.0,
        label_groups={CHRIS: PERSON},
    )


def _person(track_id, bbox, label=PERSON, matched=True) -> Track:
    return Track(track_id=track_id, bbox=bbox, label_uuid=label, confidence=0.6, matched_this_update=matched)


def _flicker(engine, start, end, bbox=COAT_RACK, on_s=30.0, off_s=20.0, dt=1.0):
    """A false positive seen for `on_s`, missed for `off_s`, repeating; returns START times."""
    starts, t, track_id = [], start, 1
    while t < end:
        visible = (t - start) % (on_s + off_s) < on_s
        if visible and (t - start) % (on_s + off_s) == 0:
            track_id += 1  # the tracker loses it while missed and mints a new id
        tracks = [_person(track_id, bbox)] if visible else []
        starts += [t for s in engine.step(tracks, t) if s.phase is EventPhase.START]
        t += dt
    return starts


class TestAStationaryFalsePositiveStopsFiring:
    def test_a_flickering_box_stops_opening_events_after_the_ignore_time(self):
        engine = _engine(_rule())

        starts = _flicker(engine, 0.0, 3 * HOUR)

        assert any(t < HOUR for t in starts), "it must fire normally while still being learned"
        assert [t for t in starts if t > HOUR + 60] == []

    def test_without_the_setting_the_same_flicker_keeps_firing(self):
        engine = _engine(_rule(ignore_after_s=0.0))

        starts = _flicker(engine, 0.0, 3 * HOUR)

        assert [t for t in starts if t > 2 * HOUR]

    def test_a_label_flip_inside_the_group_is_the_same_spot(self):
        engine = _engine(_rule())
        _flicker(engine, 0.0, HOUR + 120)

        t = HOUR + 200
        starts = [s for dt in range(30) for s in engine.step([_person(99, COAT_RACK, label=CHRIS)], t + dt)]

        assert not [s for s in starts if s.phase is EventPhase.START]


class TestOnlyTheStationarySpotIsIgnored:
    def test_a_person_elsewhere_still_fires(self):
        engine = _engine(_rule())
        _flicker(engine, 0.0, HOUR + 120)

        t = HOUR + 200
        signals = [s for dt in range(5) for s in engine.step([_person(1, COAT_RACK), _person(50, DOORWAY)], t + dt)]

        assert [s for s in signals if s.phase is EventPhase.START]

    def test_a_spot_unseen_for_the_ignore_time_is_forgotten(self):
        engine = _engine(_rule())
        _flicker(engine, 0.0, HOUR + 120)
        for dt in range(0, int(HOUR) + 60, 30):
            engine.step([], HOUR + 120 + dt)

        t = 2 * HOUR + 300
        signals = [s for dt in range(5) for s in engine.step([_person(77, COAT_RACK)], t + dt)]

        assert [s for s in signals if s.phase is EventPhase.START]

    def test_a_coasting_track_does_not_keep_a_spot_alive(self):
        """Only a real detection counts as the box still being there."""
        engine = _engine(_rule())
        engine.step([_person(1, COAT_RACK)], 0.0)
        for t in range(1, int(HOUR) + 60):
            engine.step([_person(1, COAT_RACK, matched=False)], float(t))

        signals = [s for dt in range(5) for s in engine.step([_person(1, COAT_RACK)], HOUR + 100 + dt)]

        assert [s for s in signals if s.phase is EventPhase.START]
