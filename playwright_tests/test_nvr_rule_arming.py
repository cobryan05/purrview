"""Arming a rule from the NVR → Live watcher card.

The rule rows on that card used to be names and nothing else, which is why they were
hard to justify taking up space. They are now the place arming is both read and changed
— chosen over the Watchers editor because that holds `enabled`, which is config saved
with a form, while arming is a live control on a watcher that is currently running.

The backend halves are covered in `tests/` (the engine gate, the bridge's switch and the
`set_rule_armed` command). What needs a browser is the wiring: that the row reflects the
watcher's reported state, that clicking sends the command for the right rule, and that a
rule the watcher is not running offers no control at all.

Run with:
    pytest playwright_tests/test_nvr_rule_arming.py -p no:asyncio -v
"""

import json
import re

from playwright.sync_api import Page, expect

from playwright_tests.test_nvr_live_view import PROFILE_UUID, TASK_UUID, _profile, _task
from playwright_tests.waits import wait_until

RULE_A = "rule-aaaa"
RULE_B = "rule-bbbb"


def _profile_with_rules() -> dict:
    profile = _profile()
    profile["rules"] = [
        {"uuid": RULE_A, "name": "person in house", "enabled": True},
        {"uuid": RULE_B, "name": "cat at bowl", "enabled": True},
    ]
    return profile


def _rules_table(armed_a: bool = True, armed_b: bool = True, include_b: bool = True) -> list:
    rows = [
        {
            "ruleUuid": RULE_A,
            "rule": "person in house",
            "armed": "Armed" if armed_a else "Disarmed",
            "cameras": "Hall cam",
        }
    ]
    if include_b:
        rows.append(
            {
                "ruleUuid": RULE_B,
                "rule": "cat at bowl",
                "armed": "Armed" if armed_b else "Disarmed",
                "cameras": "Hall cam",
            }
        )
    return rows


def _mock(page: Page, *, running=True, rules_table=None):
    """Wire the Live pane with a running watcher reporting `rules_table`.

    Returns the list of `set_rule_armed` commands the page sent.
    """
    commands = []

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/watch-profiles"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "profiles": [_profile_with_rules()]}),
        ),
    )
    tasks = {TASK_UUID: _task("running")} if running else {}
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/get"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "tasks": tasks}),
        ),
    )
    table = _rules_table() if rules_table is None else rules_table
    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/status-data"),
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "statusData": {TASK_UUID: {"sources": [], "rulesTable": table}} if running else {},
                }
            ),
        ),
    )
    page.route(
        lambda url: "/live-preview" in url,
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "preview": None}),
        ),
    )

    def _record(route):
        commands.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))

    page.route(
        lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/command"),
        _record,
    )
    return commands


def _open_live_pane(page: Page, base_url: str, *, expand_rules: bool = True):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-nvr']").click()
    page.wait_for_selector("#nvr-live-box .nvr-profile-card")
    # The list is a collapsed disclosure by default — the card is mostly camera
    # thumbnails and the rule names would push them off the screen. Everything below is
    # about what is *inside* it, so open it first.
    if expand_rules:
        _expand_rules(page)


def _expand_rules(page: Page):
    """Open the rule disclosure, re-clicking until its rows are actually visible.

    The pane rebuilds every profile card whenever any watcher changes state, so a click
    can land on a summary that is on its way out: the `toggle` never fires, nothing is
    persisted, and the rows sit inside a collapsed <details> for the full timeout. That
    is the "123 x locator resolved to 2 elements" shape in FLAKY_TESTS.md — the rows
    exist, they are just hidden, and a single lost click is never retried. The open
    state is persisted per profile, so clicking again is idempotent.
    """
    rows = page.locator("[data-testid='nvr-rule-row']")

    def _opened():
        if rows.first.is_visible():
            return True
        try:
            page.locator("[data-testid='nvr-rule-summary']").first.click(timeout=2000)
        except Exception:
            return False  # the card was mid-rebuild; the next poll finds the new one
        return rows.first.is_visible()

    wait_until(page, _opened, "the rule list never opened")


class TestTheCardShowsArming:
    def test_an_armed_rule_and_a_disarmed_one_read_differently(self, page: Page, base_url: str):
        """The pair, because a row that always printed one of them would satisfy either
        assertion alone — and it is the disarmed reading that has to be right."""
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=False))
        _open_live_pane(page, base_url)

        rows = page.locator("[data-testid='nvr-rule-row']")
        expect(rows).to_have_count(2)
        expect(rows.nth(0)).to_contain_text("Armed")
        expect(rows.nth(1)).to_contain_text("Disarmed")
        # The dim applies only to the disarmed row, so scanning a list finds it.
        # Regex, not a literal: `to_have_class` matches the whole attribute.
        expect(rows.nth(0)).not_to_have_class(re.compile(r"nvr-rule-row--disarmed"))
        expect(rows.nth(1)).to_have_class(re.compile(r"nvr-rule-row--disarmed"))

    def test_the_button_offers_the_action_not_the_state(self, page: Page, base_url: str):
        """A button reading "Armed" is ambiguous about whether it reports or acts. The
        pill states, the button acts."""
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=False))
        _open_live_pane(page, base_url)

        toggles = page.locator("[data-testid='nvr-rule-arm-toggle']")
        expect(toggles.nth(0)).to_have_text("Disarm")
        expect(toggles.nth(1)).to_have_text("Arm")

    def test_a_rule_the_watcher_is_not_running_gets_no_control(self, page: Page, base_url: str):
        """A rule scoped to other cameras, or disabled, is absent from the watcher's
        report. Offering a control for it would claim to govern something that is not
        running whatever it said."""
        _mock(page, rules_table=_rules_table(include_b=False))
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-rule-row']")).to_have_count(2)
        expect(page.locator("[data-testid='nvr-rule-arm-toggle']")).to_have_count(1)

    def test_a_stopped_watcher_lists_rules_without_controls(self, page: Page, base_url: str):
        """Arming applies to a running watcher. A stopped card still names its rules —
        that is the profile's shape — but has nothing to arm."""
        _mock(page, running=False)
        _open_live_pane(page, base_url)

        expect(page.locator("[data-testid='nvr-rule-row']")).to_have_count(2)
        expect(page.locator("[data-testid='nvr-rule-arm-toggle']")).to_have_count(0)


class TestTheCollapsedRuleList:
    """What a closed list still has to say.

    Collapsing the rules buys the cameras their space, but a rule that will not fire is
    exactly the thing nobody should have to expand nine cards to discover.
    """

    def test_the_summary_counts_disarmed_rules_while_closed(self, page: Page, base_url: str):
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=False))
        _open_live_pane(page, base_url, expand_rules=False)

        summary = page.locator("[data-testid='nvr-rule-summary']")
        expect(page.locator("[data-testid='nvr-rule-row']").first).not_to_be_visible()
        expect(summary).to_have_text("2 rules · 1 disarmed")

    def test_a_fully_armed_profile_says_nothing_alarming(self, page: Page, base_url: str):
        """The negative half of the pair: a summary that always warned would be ignored."""
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=True))
        _open_live_pane(page, base_url, expand_rules=False)

        expect(page.locator("[data-testid='nvr-rule-summary']")).to_have_text("2 rules")

    def test_disarming_from_the_open_list_updates_the_summary(self, page: Page, base_url: str):
        """The summary is the closed view of the list, so it cannot go stale when the
        list is used — that would report "all armed" over a rule just disarmed."""
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=True))
        _open_live_pane(page, base_url)

        page.locator("[data-testid='nvr-rule-row']").nth(1).locator("[data-testid='nvr-rule-arm-toggle']").click()

        expect(page.locator("[data-testid='nvr-rule-summary']")).to_have_text("2 rules · 1 disarmed")


class TestTogglingFromTheCard:
    def test_clicking_disarm_sends_that_rule_and_repaints(self, page: Page, base_url: str):
        """The request matters as much as the pixels: a toggle that repainted without
        naming the right rule would look correct and arm the wrong thing."""
        commands = _mock(page, rules_table=_rules_table(armed_a=True, armed_b=True))
        _open_live_pane(page, base_url)
        second = page.locator("[data-testid='nvr-rule-row']").nth(1)

        second.locator("[data-testid='nvr-rule-arm-toggle']").click()

        expect(second.locator("[data-testid='nvr-rule-arm-toggle']")).to_have_text("Arm")
        expect(second).to_contain_text("Disarmed")
        assert commands == [
            {
                "taskUuid": TASK_UUID,
                "command": "set_rule_armed",
                "params": {"ruleUuid": RULE_B, "armed": False},
            }
        ], commands

    def test_a_failed_command_puts_the_row_back(self, page: Page, base_url: str):
        """The row repaints before the round trip, because the watcher only applies
        arming on its next loop pass and waiting would leave the button looking dead. The
        cost of that choice is that a failure must be undone, or the card would claim a
        change the watcher never made.
        """
        _mock(page, rules_table=_rules_table(armed_a=True, armed_b=True))
        _open_live_pane(page, base_url)
        # Registered after the mock's, and so matched first.
        page.route(
            lambda url: url.split("?")[0].rstrip("/").endswith("/api/tasks/instances/command"),
            lambda route: route.abort(),
        )
        first = page.locator("[data-testid='nvr-rule-row']").first

        first.locator("[data-testid='nvr-rule-arm-toggle']").click()

        expect(first.locator("[data-testid='nvr-rule-arm-toggle']")).to_have_text("Disarm")
        expect(first).to_contain_text("Armed")
