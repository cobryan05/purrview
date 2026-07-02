"""Regression tests for the thumbnail file-resolution path-traversal guard.

`_resolve_image_file_path` must never return a path outside FILES_ROOT.
Previously a fallback branch re-opened the raw user-supplied path when
`get_safe_path` rejected it, allowing arbitrary file reads via the
`/api/images/{image_path:path}/thumbnail` route.
"""

import pytest

from apps.db_server.manager import Manager


@pytest.fixture
async def manager(db_client, files_root):
    m = Manager(db_client=db_client, files_root=files_root)
    m._db_ready = True
    return m


class TestResolveImageFilePathTraversal:
    def test_absolute_path_outside_root_is_rejected(self, manager, tmp_path):
        # A real file that exists OUTSIDE files_root.
        secret = tmp_path / "secret_outside_root.png"
        secret.write_bytes(b"not really a png but a real file")
        assert secret.is_file()

        resolved = manager._resolve_image_file_path(str(secret))
        assert resolved is None, "path outside FILES_ROOT must not resolve"

    def test_relative_traversal_outside_root_is_rejected(self, manager, tmp_path):
        secret = tmp_path / "secret_outside_root.png"
        secret.write_bytes(b"real file")
        # ../secret_outside_root.png relative to files_root escapes the root.
        traversal = "../secret_outside_root.png"
        resolved = manager._resolve_image_file_path(traversal)
        assert resolved is None

    def test_file_under_root_still_resolves(self, manager, files_root):
        # The trusted caller passes an absolute path under FILES_ROOT (produced
        # by get_safe_path at import time) — that must still resolve.
        img = files_root / "sub" / "img.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"real image bytes")

        assert manager._resolve_image_file_path(str(img)) == img.resolve()
        # A relative path under the root must also resolve.
        assert manager._resolve_image_file_path("sub/img.jpg") == img.resolve()
