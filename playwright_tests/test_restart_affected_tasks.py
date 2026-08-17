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
    const { promptRestartAffectedTasks, findTasksUsing } =
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
