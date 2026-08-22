"""Tests for ui/utils/restartAffectedTasks.js.

A task reads its config once, in `_init`. Editing a watch profile / filter
chain / source therefore does nothing to an already-running instance, and the
UI used to report "Saved" with no hint of that. These tests pin the prompt
that closes the gap, and — more importantly — pin the cases where it must
*not* fire, since a spurious "restart your watcher?" on every save would be
worse than the original silence.

`window.fetch` is stubbed so the whole flow runs against scripted task-server
responses; nothing here starts or cancels a real task.
"""

import pytest
from playwright.sync_api import Page

# Stubs window.fetch, records every call, and returns scripted responses for
# the four task endpoints this module touches. `window.__calls` is the log the
# assertions read back.
_SETUP = """
    const { promptRestartAffectedTasks, findTasksUsing,
            promptRestartOwnTask, findRunningInstancesOfConfig } =
        await import('/app-static/js/ui/utils/restartAffectedTasks.js');

    window.__calls = [];
    const _realFetch = window.fetch;
    window.fetch = async (url, opts) => {
        const body = opts?.body ? JSON.parse(opts.body) : null;
        window.__calls.push({ url: String(url), body });
        let payload = { status: 'success' };
        if (String(url).includes('/api/tasks/instances/get')) {
            payload = { status: 'success', tasks: window.__tasks };
            // After a cancel, report the instance as stopped so waitForStop()
            // resolves instead of spinning until its timeout.
            if (window.__cancelled) {
                payload = { status: 'success', tasks: {} };
            }
        } else if (String(url).includes('/api/tasks/instances/cancel')) {
            window.__cancelled = true;
        } else if (String(url).includes('/api/tasks/instances/start')) {
            payload = { status: 'success', taskUuid: 'new-instance' };
        }
        return new Response(JSON.stringify(payload), {
            status: 200, headers: { 'Content-Type': 'application/json' },
        });
    };
    window.__restoreFetch = () => { window.fetch = _realFetch; };
"""

_TEARDOWN = """
    window.__restoreFetch();
    delete window.__tasks; delete window.__cancelled;
    document.querySelectorAll('[data-testid="modal-shell-overlay"]').forEach(d => d.remove());
"""


def _running_task(params: str, status: str = "running") -> str:
    """One entry for the stubbed /instances/get response."""
    return f"""{{
        'task-1': {{
            config: {{ uuid: 'cfg-1', name: 'Watch Test', typename: 'StreamWatcherTask',
                       params: {params} }},
            instance: {{ uuid: 'task-1', configUuid: 'cfg-1', status: '{status}' }},
        }}
    }}"""


class TestFindTasksUsing:
    def test_finds_task_referencing_uuid_directly(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const found = await findTasksUsing('prof-1');
            {_TEARDOWN}
            return found.map(f => f.taskUuid);
        }}""")
        assert result == ["task-1"]

    def test_finds_uuid_nested_in_array(self, shared_page: Page):
        # The collector stores sources as a list, so a source edit has to match
        # through an array rather than a scalar param.
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ source_uuids: ['a', 'src-9', 'c'] }")};
            const found = await findTasksUsing('src-9');
            {_TEARDOWN}
            return found.length;
        }}""")
        assert result == 1

    def test_finds_uuid_used_as_an_object_key(self, shared_page: Page):
        # model_label_config keys its thresholds *by* label uuid.
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ modelLabelConfig: { labelValues: { 'lbl-7': { min: 0.25 } } } }")};
            const found = await findTasksUsing('lbl-7');
            {_TEARDOWN}
            return found.length;
        }}""")
        assert result == 1

    def test_ignores_unrelated_uuid(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const found = await findTasksUsing('some-other-uuid');
            {_TEARDOWN}
            return found.length;
        }}""")
        assert result == 0

    @pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
    def test_ignores_tasks_that_are_not_live(self, shared_page: Page, status: str):
        # Restarting a finished task would be surprising, not helpful.
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }", "PLACEHOLDER")};
            const found = await findTasksUsing('prof-1');
            {_TEARDOWN}
            return found.length;
        }}""".replace("PLACEHOLDER", status))
        assert result == 0


class TestPromptRestartAffectedTasks:
    def test_no_prompt_when_nothing_is_running(self, shared_page: Page):
        # The common case: this runs on every save, so it must stay silent.
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {{}};
            const n = await promptRestartAffectedTasks('prof-1', {{ label: 'watcher' }});
            const dialogs = document.querySelectorAll('[data-testid="modal-shell-overlay"]').length;
            {_TEARDOWN}
            return {{ n, dialogs }};
        }}""")
        assert result == {"n": 0, "dialogs": 0}

    def test_confirming_cancels_then_starts(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const p = promptRestartAffectedTasks('prof-1', {{ label: 'watcher' }});
            await new Promise(r => setTimeout(r, 150));
            const btn = [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Restart it');
            btn.click();
            const n = await p;
            const urls = window.__calls.map(c => c.url.replace(location.origin, ''));
            {_TEARDOWN}
            return {{ n, urls }};
        }}""")
        assert result["n"] == 1
        assert "/api/tasks/instances/cancel" in result["urls"]
        assert "/api/tasks/instances/start" in result["urls"]
        # Order matters: starting before the old instance released its sources
        # gives two instances competing for the same camera.
        assert result["urls"].index("/api/tasks/instances/cancel") < result["urls"].index("/api/tasks/instances/start")

    def test_declining_leaves_the_task_alone(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const p = promptRestartAffectedTasks('prof-1', {{ label: 'watcher' }});
            await new Promise(r => setTimeout(r, 150));
            const btn = [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Keep running');
            btn.click();
            const n = await p;
            const urls = window.__calls.map(c => c.url);
            {_TEARDOWN}
            return {{ n, cancelled: urls.some(u => u.includes('cancel')),
                      started: urls.some(u => u.includes('start')) }};
        }}""")
        assert result == {"n": 0, "cancelled": False, "started": False}

    def test_prompt_names_the_affected_task(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const p = promptRestartAffectedTasks('prof-1', {{ label: 'watcher' }});
            await new Promise(r => setTimeout(r, 150));
            const text = document.querySelector('[data-testid="modal-shell-dialog"]').textContent;
            [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Keep running').click();
            await p;
            {_TEARDOWN}
            return text;
        }}""")
        assert '"Watch Test"' in result
        assert "watcher" in result

    def test_task_server_failure_does_not_break_the_save(self, shared_page: Page):
        # The save already succeeded by the time this runs; a task-server
        # outage must not surface as a second error on top of it.
        result = shared_page.evaluate("""async () => {
            const { promptRestartAffectedTasks } =
                await import('/app-static/js/ui/utils/restartAffectedTasks.js');
            const _realFetch = window.fetch;
            window.fetch = async () => { throw new Error('task server down'); };
            let threw = false;
            let n = null;
            try { n = await promptRestartAffectedTasks('prof-1', { label: 'watcher' }); }
            catch { threw = true; }
            window.fetch = _realFetch;
            document.querySelectorAll('[data-testid="modal-shell-overlay"]').forEach(d => d.remove());
            return { threw, n };
        }""")
        assert result == {"threw": False, "n": 0}


class TestRestartOwnTask:
    """Saving a *task config* while an instance of it is running.

    Distinct from every other case in this file: the deep params scan cannot see
    this one at all, because a config's params never contain its own uuid.
    """

    def test_the_deep_scan_alone_would_miss_it(self, shared_page: Page):
        """Why the sibling finder exists. Same running task, same uuid — the
        params scan finds nothing and the config scan finds it."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }")};
            const byParams = await findTasksUsing('cfg-1');
            const byConfig = await findRunningInstancesOfConfig('cfg-1');
            {_TEARDOWN}
            return {{ byParams: byParams.length, byConfig: byConfig.length }};
        }}""")
        assert result == {"byParams": 0, "byConfig": 1}

    def test_no_prompt_for_a_different_config(self, shared_page: Page):
        """Saving config B must not offer to restart config A's instance."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }")};
            const n = await promptRestartOwnTask('cfg-2');
            const dialogs = document.querySelectorAll('[data-testid="modal-shell-overlay"]').length;
            {_TEARDOWN}
            return {{ n, dialogs }};
        }}""")
        assert result == {"n": 0, "dialogs": 0}

    @pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
    def test_no_prompt_when_the_instance_is_not_live(self, shared_page: Page, status: str):
        """A finished instance is not using the old settings any more, so the
        prompt would be pure noise on every save."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }", status=status)};
            const n = await promptRestartOwnTask('cfg-1');
            const dialogs = document.querySelectorAll('[data-testid="modal-shell-overlay"]').length;
            {_TEARDOWN}
            return {{ n, dialogs }};
        }}""")
        assert result == {"n": 0, "dialogs": 0}

    def test_confirming_cancels_then_starts_the_same_config(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }")};
            const p = promptRestartOwnTask('cfg-1');
            await new Promise(r => setTimeout(r, 150));
            const btn = [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Restart it');
            btn.click();
            const n = await p;
            const calls = window.__calls.map(c => ({{ url: c.url.replace(location.origin, ''), body: c.body }}));
            {_TEARDOWN}
            return {{ n, calls }};
        }}""")
        assert result["n"] == 1
        urls = [c["url"] for c in result["calls"]]
        assert "/api/tasks/instances/cancel" in urls
        assert urls.index("/api/tasks/instances/cancel") < urls.index("/api/tasks/instances/start")
        # The replacement must be started from the config that was just saved,
        # or the restart would relaunch something else entirely.
        start = next(c for c in result["calls"] if c["url"].endswith("/start"))
        assert start["body"]["configUuid"] == "cfg-1"

    def test_declining_leaves_the_instance_running(self, shared_page: Page):
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }")};
            const p = promptRestartOwnTask('cfg-1');
            await new Promise(r => setTimeout(r, 150));
            [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Keep running').click();
            const n = await p;
            const urls = window.__calls.map(c => c.url);
            {_TEARDOWN}
            return {{ n, cancelled: urls.some(u => u.includes('cancel')),
                      started: urls.some(u => u.includes('start')) }};
        }}""")
        assert result == {"n": 0, "cancelled": False, "started": False}

    def test_prompt_says_the_changes_are_saved_but_not_applied(self, shared_page: Page):
        """The point of the wording: the save *did* work, and the user should not
        be left thinking it failed."""
        result = shared_page.evaluate(f"""async () => {{
            {_SETUP}
            window.__tasks = {_running_task("{ output_dir: 'run_ring' }")};
            const p = promptRestartOwnTask('cfg-1');
            await new Promise(r => setTimeout(r, 150));
            const text = document.querySelector('[data-testid="modal-shell-dialog"]').textContent;
            [...document.querySelectorAll('[data-testid="modal-shell-dialog"] button')]
                .find(b => b.textContent === 'Keep running').click();
            await p;
            {_TEARDOWN}
            return text;
        }}""")
        assert "saved" in result
        assert "restarted" in result


class TestIndirectReferences:
    """A watcher names only its *profile*; the cameras and filter chains it uses
    live in the profile row.

    So the params scan on its own could never see them: editing a camera's URL
    while a watcher was running matched nothing and offered no restart — exactly
    the silence this module exists to break, for the one config kind most likely
    to be edited mid-run.
    """

    _SETUP_WITH_PROFILES = """
        const { findTasksUsing } =
            await import('/app-static/js/ui/utils/restartAffectedTasks.js');
        const { clearWatchProfilesCache } =
            await import('/app-static/js/shared/api/watchProfiles.js');
        // The fetcher caches across tests on a shared page, and other tests have
        // already primed it with an empty list.
        clearWatchProfilesCache();

        const _realFetch = window.fetch;
        window.fetch = async (url, opts) => {
            let payload = { status: 'success' };
            if (String(url).includes('/api/watch-profiles')) {
                payload = { status: 'success', profiles: window.__profiles || [] };
            } else if (String(url).includes('/api/tasks/instances/get')) {
                payload = { status: 'success', tasks: window.__tasks };
            }
            return new Response(JSON.stringify(payload), {
                status: 200, headers: { 'Content-Type': 'application/json' },
            });
        };
        window.__restoreFetch = () => { window.fetch = _realFetch; clearWatchProfilesCache(); };
    """

    def _run(self, page: Page, profiles: str, needle: str) -> list:
        return page.evaluate(f"""async () => {{
            {self._SETUP_WITH_PROFILES}
            window.__profiles = {profiles};
            window.__tasks = {_running_task("{ watch_profile_uuid: 'prof-1' }")};
            const found = await findTasksUsing('{needle}');
            window.__restoreFetch();
            delete window.__tasks; delete window.__profiles;
            return found.map(f => f.taskUuid);
        }}""")

    def test_a_source_used_through_a_watch_profile_is_found(self, shared_page: Page):
        profiles = "[{ uuid: 'prof-1', name: 'P', sources: [{ sourceUuid: 'src-9' }] }]"
        assert self._run(shared_page, profiles, "src-9") == ["task-1"]

    def test_a_filter_chain_referenced_by_an_action_is_found(self, shared_page: Page):
        """The chain hangs off an action's params, deeper than the source list."""
        profiles = (
            "[{ uuid: 'prof-1', name: 'P', sources: [], "
            "actions: [{ kind: 'apply_filter_chain', paramsJson: { filterChainUuid: 'chain-7' } }] }]"
        )
        assert self._run(shared_page, profiles, "chain-7") == ["task-1"]

    def test_a_source_in_some_other_profile_is_not_matched(self, shared_page: Page):
        """A spurious "restart your watcher?" on an unrelated save would be worse
        than the silence this replaced."""
        profiles = "[{ uuid: 'prof-2', name: 'Other', sources: [{ sourceUuid: 'src-9' }] }]"
        assert self._run(shared_page, profiles, "src-9") == []

    def test_the_profile_itself_still_matches_directly(self, shared_page: Page):
        profiles = "[{ uuid: 'prof-1', name: 'P', sources: [{ sourceUuid: 'src-9' }] }]"
        assert self._run(shared_page, profiles, "prof-1") == ["task-1"]
