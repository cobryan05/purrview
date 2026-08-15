"""Importing images from the user's machine, from the Files browser.

The destination is asked for once and then remembered: picking a folder on
every drop is friction on the common path (everything lands in one place),
while never asking would scatter files wherever you happened to be browsing.

Run with:
    pytest playwright_tests/test_image_upload_ui.py -p no:asyncio -n 0 -v
"""

import base64
import json

from playwright.sync_api import Page, expect

from playwright_tests.waits import wait_until

# A 1x1 PNG — the server decodes uploads for real, but these tests mock the
# endpoint, so it only has to be plausible file content.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _mock_settings(page: Page, stored: dict):
    """Stand in for the user_settings store, which drives the ask/remember flow."""
    state = dict(stored)

    def handle(route):
        key = route.request.url.rstrip("/").split("/")[-1]
        if route.request.method == "PUT":
            state[key] = json.loads(route.request.post_data or "{}").get("value")
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "success"}))
            return
        if key in state:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"status": "success", "value": state[key]}),
            )
        else:
            # An unset key 404s — that is how "never chosen" is represented.
            route.fulfill(status=404, content_type="application/json", body=json.dumps({"status": "failure"}))

    page.route("**/api/settings/*", handle)
    return state


def _mock_upload(page: Page, results=None):
    calls: list[dict] = []

    def handle(route):
        # post_data_buffer, not post_data: a multipart body carries raw image
        # bytes, and post_data tries to UTF-8 decode it (blowing up on the PNG
        # header). Decoded leniently here because the assertions only look at
        # the text form fields around the binary part.
        raw = route.request.post_data_buffer or b""
        calls.append({"url": route.request.url, "body": raw.decode("utf-8", errors="replace")})
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "success",
                    "results": (
                        results
                        if results is not None
                        else [{"filename": "cat.png", "ok": True, "path": "cat.png", "uuid": "img-1"}]
                    ),
                }
            ),
        )

    page.route("**/api/images/upload", handle)
    return calls


def _mock_file_list(page: Page):
    page.route(
        "**/api/images/list**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "files": [], "has_more": False}),
        ),
    )


def _open_files_tab(page: Page, base_url: str):
    page.goto(base_url)
    page.wait_for_load_state("networkidle")
    page.locator(".main-tab-button[data-tab='tab-image-tagging']").click()
    page.locator("#tab-image-tagging .section-tab-button[data-tab='tab-annotate-workspace']").click()
    page.locator("#tab-image-tagging .workspace-tab-button[data-tab='tab-files']").click()
    page.wait_for_load_state("networkidle")


def _choose_file(page: Page, name: str = "cat.png"):
    """Answer the OS file picker without one ever opening."""
    with page.expect_file_chooser() as fc:
        page.get_by_test_id("file-upload-button").click()
    fc.value.set_files({"name": name, "mimeType": "image/png", "buffer": TINY_PNG})


class TestAffordance:
    def test_the_files_toolbar_offers_an_import(self, page: Page, base_url: str):
        _mock_file_list(page)
        _mock_settings(page, {})
        _open_files_tab(page, base_url)

        expect(page.get_by_test_id("file-upload-button")).to_be_visible()

    def test_the_upload_folder_can_be_changed_without_uploading(self, page: Page, base_url: str):
        """The remembered folder would otherwise be unreachable once set."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "incoming", "upload_target_remembered": "true"})
        _open_files_tab(page, base_url)

        expect(page.get_by_test_id("file-upload-target-button")).to_be_visible()


class TestAskThenRemember:
    def test_the_first_upload_asks_where_to_put_it(self, page: Page, base_url: str):
        """Nothing stored yet, so it must not guess."""
        _mock_file_list(page)
        _mock_settings(page, {})
        _mock_upload(page)
        _open_files_tab(page, base_url)

        _choose_file(page)

        # The directory picker comes first, then the remember prompt.
        expect(page.get_by_test_id("modal-shell-dialog").first).to_be_visible()

    def test_a_remembered_folder_uploads_without_asking(self, page: Page, base_url: str):
        """The whole point of remembering: no dialog on the common path."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "incoming", "upload_target_remembered": "true"})
        calls = _mock_upload(page)
        _open_files_tab(page, base_url)

        _choose_file(page)
        page.wait_for_timeout(1200)

        assert calls, "Expected the upload to go straight out with no prompt"
        assert 'name="path"' in (calls[0]["body"] or "")
        assert "incoming" in (calls[0]["body"] or "")

    def test_a_stored_folder_that_was_not_remembered_still_asks(self, page: Page, base_url: str):
        """Unticking "don't ask again" has to keep asking, or the checkbox lies."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "incoming", "upload_target_remembered": "false"})
        calls = _mock_upload(page)
        _open_files_tab(page, base_url)

        _choose_file(page)
        page.wait_for_timeout(800)

        expect(page.get_by_test_id("modal-shell-dialog").first).to_be_visible()
        assert not calls, "Must not upload before the destination is confirmed"

    def test_confirming_the_prompt_actually_uploads(self, page: Page, base_url: str):
        """The whole ask path end to end. Every other test in this class stops
        at the dialog, so nothing caught that answering it uploaded nothing."""
        _mock_file_list(page)
        _mock_settings(page, {})
        calls = _mock_upload(page)
        _open_files_tab(page, base_url)

        _choose_file(page)

        # Directory picker first…
        page.locator(".fb-select-btn").first.click()
        # …then "Upload to this folder?".
        confirm = page.get_by_test_id("upload-remember-confirm")
        expect(confirm).to_be_visible()
        confirm.click()

        wait_until(page, lambda: bool(calls), "Expected confirming the prompt to upload the file")

    def test_cancelling_the_prompt_uploads_nothing(self, page: Page, base_url: str):
        _mock_file_list(page)
        _mock_settings(page, {})
        calls = _mock_upload(page)
        _open_files_tab(page, base_url)

        _choose_file(page)

        page.locator(".fb-select-btn").first.click()
        page.get_by_test_id("upload-remember-dialog").locator("button", has_text="Cancel").click()
        page.wait_for_timeout(800)

        assert not calls, "Cancelling the destination prompt must not upload"


class TestPickerDismissal:
    def test_a_picker_that_fires_no_event_still_frees_the_import_button(self, page: Page, base_url: str):
        """`<input type=file>` does not fire 'cancel' everywhere, and a dismissed
        picker fires no 'change' either — without a fallback the promise never
        settles and Import stays disabled for the life of the pane."""
        _mock_file_list(page)
        _mock_settings(page, {})
        _open_files_tab(page, base_url)

        button = page.get_by_test_id("file-upload-button")
        with page.expect_file_chooser():
            button.click()
        # Deliberately never answer the chooser: no files, no 'change', and no
        # 'cancel'. Regaining focus is all the app gets.
        expect(button).to_be_disabled()
        page.evaluate("window.dispatchEvent(new Event('focus'))")

        expect(button).to_be_enabled()


class TestReporting:
    def test_a_rejected_file_is_named_in_the_error(self, page: Page, base_url: str):
        """ "1 of 3 failed" without saying which is not actionable."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "", "upload_target_remembered": "true"})
        _mock_upload(
            page,
            results=[{"filename": "notes.pdf", "ok": False, "message": "not an image this app reads"}],
        )
        _open_files_tab(page, base_url)

        _choose_file(page, "notes.pdf")

        expect(page.locator("#toast-container")).to_contain_text("notes.pdf")

    def test_a_duplicate_is_reported_but_not_treated_as_a_failure(self, page: Page, base_url: str):
        """Advisory, not a veto — the file was imported either way."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "", "upload_target_remembered": "true"})
        _mock_upload(
            page,
            results=[
                {
                    "filename": "again.png",
                    "ok": True,
                    "path": "again.png",
                    "uuid": "img-2",
                    "duplicateOf": "img-1",
                    "duplicateDistance": 0,
                }
            ],
        )
        _open_files_tab(page, base_url)

        _choose_file(page, "again.png")

        toasts = page.locator("#toast-container")
        expect(toasts).to_contain_text("already in the database")
        expect(toasts).to_contain_text("imported anyway")

    def test_a_renamed_file_says_so(self, page: Page, base_url: str):
        """Otherwise the user goes looking for a name that is not there."""
        _mock_file_list(page)
        _mock_settings(page, {"upload_target_dir": "", "upload_target_remembered": "true"})
        _mock_upload(
            page,
            results=[
                {
                    "filename": "snapshot.png",
                    "ok": True,
                    "path": "snapshot_1.png",
                    "uuid": "img-3",
                    "renamedFrom": "snapshot.png",
                }
            ],
        )
        _open_files_tab(page, base_url)

        _choose_file(page, "snapshot.png")

        expect(page.locator("#toast-container")).to_contain_text("snapshot_1.png")


class TestCheckboxSizing:
    """Checkboxes must not collapse in a flex row — a global rule, tested here
    because this dialog is where it last surfaced.

    As flex children they default to `flex-shrink: 1`, so a long label squeezes
    them toward zero and Pico fills a checked one, leaving a ~2px sliver that
    reads as a rendering bug. It had been patched locally three times
    (.ms-item, .panel-field-row, .inference-model-item) before the base-layer
    rule; this guards against a fourth.
    """

    PROBE = """
    <div id="cb-probe">
      <label class="upload-remember__check" style="display:flex;gap:.5rem;align-items:center">
        <input type="checkbox"><span>Always upload here, a long label to squeeze it</span>
      </label>
      <div class="panel-field-row" style="display:flex;gap:.5rem">
        <input type="checkbox"><span>a long trailing label to squeeze it</span>
      </div>
      <div class="ms-item" style="display:flex;gap:.5rem">
        <input type="checkbox"><span>a long trailing label to squeeze it</span>
      </div>
    </div>
    """

    def test_checkboxes_keep_their_width_beside_a_long_label(self, page: Page, base_url: str):
        page.goto(base_url)
        page.wait_for_load_state("networkidle")
        page.evaluate(
            "html => document.querySelector('main.container').insertAdjacentHTML('afterbegin', html)",
            self.PROBE,
        )

        measured = page.evaluate("""() => [...document.querySelectorAll('#cb-probe input[type=checkbox]')].map(e => ({
                parent: e.parentElement.className,
                width: e.getBoundingClientRect().width,
            }))""")

        assert measured, "probe rendered no checkboxes"
        for box in measured:
            assert box["width"] >= 8, f"{box['parent']} checkbox collapsed to {box['width']}px"
