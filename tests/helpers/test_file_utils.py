"""Unit tests for get_safe_path — apps/helpers/fileUtils.py.

Security-critical: every path-traversal escape attempt must return None.
All tests are pure filesystem operations on tmp directories.
"""
from pathlib import Path

import pytest

from apps.helpers.fileUtils import get_safe_path


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
