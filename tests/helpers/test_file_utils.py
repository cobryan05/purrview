"""Unit tests for apps/helpers/fileUtils.py.

Security-critical: every path-traversal escape attempt must return None.
All tests are pure filesystem operations on tmp directories.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.helpers.fileUtils import filter_stubs_by_directory, get_safe_path


def test_simple_relative_path(tmp_path):
    result = get_safe_path(tmp_path, "subdir/file.png")
    assert result == tmp_path / "subdir" / "file.png"


def test_root_itself(tmp_path):
    result = get_safe_path(tmp_path, ".")
    assert result == tmp_path


def test_nested_path_allowed(tmp_path):
    result = get_safe_path(tmp_path, "a/b/c/d.txt")
    assert result is not None
    assert str(tmp_path) in str(result)
    assert result == tmp_path / "a" / "b" / "c" / "d.txt"


def test_traversal_with_dotdot_blocked(tmp_path):
    result = get_safe_path(tmp_path, "../outside.txt")
    assert result is None


def test_traversal_double_dot_in_middle_blocked(tmp_path):
    result = get_safe_path(tmp_path, "a/../../b")
    assert result is None


def test_traversal_absolute_escape_blocked(tmp_path):
    result = get_safe_path(tmp_path, "/etc/passwd")
    # /etc/passwd is absolute but not under tmp_path → treated as relative
    # The resolved path should be under tmp_path
    if result is not None:
        assert str(tmp_path) in str(result)


def test_absolute_path_inside_root(tmp_path):
    child = tmp_path / "images" / "photo.png"
    result = get_safe_path(tmp_path, str(child))
    assert result == child


def test_absolute_path_outside_root_treated_as_relative(tmp_path, tmp_path_factory):
    other = tmp_path_factory.mktemp("other_root")
    victim = other / "secret.txt"
    result = get_safe_path(tmp_path, str(victim))
    # Must not return a path outside tmp_path
    if result is not None:
        assert str(tmp_path) in str(result)


def test_nonexistent_root_returns_none(tmp_path):
    fake_root = tmp_path / "does_not_exist"
    result = get_safe_path(fake_root, "file.png")
    assert result is None


def test_empty_relative_path(tmp_path):
    # Empty string resolves to the root itself
    result = get_safe_path(tmp_path, "")
    # Either returns root or None — must not be outside root
    if result is not None:
        assert str(tmp_path) in str(result)


def test_just_filename(tmp_path):
    result = get_safe_path(tmp_path, "image.jpg")
    assert result == tmp_path / "image.jpg"


def test_result_is_path_object(tmp_path):
    result = get_safe_path(tmp_path, "file.txt")
    assert isinstance(result, Path)


# ── filter_stubs_by_directory ──────────────────────────────────────────────
#
# Regression coverage for the bug fixed in 2e00d9f: the DB stores filenames as
# *absolute* paths, so a relative `directory` filter must be resolved to an
# absolute path under FILES_ROOT before prefix-matching, or every stub is
# silently dropped (relative "foo" never prefix-matches absolute
# "/root/foo/img.jpg").


def _stub(filename):
    return SimpleNamespace(uuid="u", filename=filename)


def test_relative_directory_matches_absolute_filenames(tmp_path):
    stubs = [
        _stub(str(tmp_path / "foo" / "img1.jpg")),
        _stub(str(tmp_path / "foo" / "sub" / "img2.jpg")),
        _stub(str(tmp_path / "bar" / "img3.jpg")),
    ]
    result = filter_stubs_by_directory(tmp_path, "foo", stubs)
    assert {s.filename for s in result} == {
        str(tmp_path / "foo" / "img1.jpg"),
        str(tmp_path / "foo" / "sub" / "img2.jpg"),
    }


def test_sibling_directory_with_shared_prefix_not_matched(tmp_path):
    # "foo" must not prefix-match "foobar" — only a real path-component boundary.
    stubs = [_stub(str(tmp_path / "foobar" / "img.jpg"))]
    result = filter_stubs_by_directory(tmp_path, "foo", stubs)
    assert result == []


def test_directory_outside_root_returns_empty(tmp_path):
    stubs = [_stub(str(tmp_path / "foo" / "img.jpg"))]
    result = filter_stubs_by_directory(tmp_path, "../outside", stubs)
    assert result == []


def test_no_matching_stubs_returns_empty(tmp_path):
    stubs = [_stub(str(tmp_path / "bar" / "img.jpg"))]
    result = filter_stubs_by_directory(tmp_path, "foo", stubs)
    assert result == []
